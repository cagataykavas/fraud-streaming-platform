from __future__ import annotations

import heapq
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class FraudSignal:
    case_id: str
    risk_score: float
    confidence: float
    customer_value: float
    amount: float
    evidence_count: int


@dataclass(frozen=True)
class ReviewPolicy:
    """Operational bounds for the in-memory review queue."""

    max_cases: int = 10_000
    review_sla: timedelta = timedelta(minutes=30)
    max_future_skew: timedelta = timedelta(seconds=30)

    def __post_init__(self) -> None:
        if isinstance(self.max_cases, bool) or not isinstance(self.max_cases, int):
            raise TypeError("max_cases must be an integer")
        if self.max_cases <= 0:
            raise ValueError("max_cases must be positive")
        if not isinstance(self.review_sla, timedelta):
            raise TypeError("review_sla must be a timedelta")
        if self.review_sla <= timedelta(0):
            raise ValueError("review_sla must be positive")
        if not isinstance(self.max_future_skew, timedelta):
            raise TypeError("max_future_skew must be a timedelta")
        if self.max_future_skew < timedelta(0):
            raise ValueError("max_future_skew cannot be negative")


@dataclass(frozen=True)
class QueueSnapshot:
    size: int
    capacity: int
    utilization: float
    overdue_cases: int
    oldest_age_seconds: float
    earliest_deadline: datetime | None


@dataclass(frozen=True)
class _QueuedCase:
    signal: FraudSignal
    enqueued_at: datetime
    deadline: datetime
    priority: float
    sequence: int


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _validate_signal(signal: FraudSignal) -> None:
    if not isinstance(signal, FraudSignal):
        raise TypeError("signal must be a FraudSignal")
    if not signal.case_id or signal.case_id.strip() != signal.case_id:
        raise ValueError("case_id must be non-empty and cannot have surrounding whitespace")
    if len(signal.case_id) > 128:
        raise ValueError("case_id cannot exceed 128 characters")

    for name in ("risk_score", "confidence", "customer_value", "amount"):
        value = getattr(signal, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if not 0.0 <= signal.risk_score <= 1.0:
        raise ValueError("risk_score must be between 0 and 1")
    if not 0.0 <= signal.confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if signal.customer_value < 0:
        raise ValueError("customer_value cannot be negative")
    if signal.amount < 0:
        raise ValueError("amount cannot be negative")
    if isinstance(signal.evidence_count, bool) or not isinstance(signal.evidence_count, int):
        raise TypeError("evidence_count must be an integer")
    if signal.evidence_count < 0:
        raise ValueError("evidence_count cannot be negative")


def triage_priority(signal: FraudSignal) -> float:
    """Return a validated materiality-and-uncertainty priority in ``[0, 1]``."""

    _validate_signal(signal)
    uncertainty = 1.0 - signal.confidence
    monetary_impact = min(signal.amount / 10_000.0, 1.0)
    customer_impact = min(signal.customer_value / 20_000.0, 1.0)
    evidence_strength = min(signal.evidence_count / 5.0, 1.0)

    return (
        0.40 * signal.risk_score
        + 0.25 * uncertainty
        + 0.15 * monetary_impact
        + 0.10 * customer_impact
        + 0.10 * evidence_strength
    )


class ReviewQueue:
    """Bounded, thread-safe review queue with deadline-based starvation control.

    Before the review SLA expires, cases are ordered by model/business priority.
    Once a case is overdue, the oldest deadline wins so continuously arriving
    high-risk work cannot leave an older case waiting forever.
    """

    def __init__(
        self,
        policy: ReviewPolicy | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = policy or ReviewPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._priority_heap: list[tuple[float, datetime, str, int]] = []
        self._deadline_heap: list[tuple[datetime, float, datetime, str, int]] = []
        self._active: dict[str, _QueuedCase] = {}
        self._sequence = 0
        self._lock = threading.Lock()

    def push(self, signal: FraudSignal, *, enqueued_at: datetime | None = None) -> None:
        priority = triage_priority(signal)
        accepted_at = _utc(self._clock(), field="clock")
        queued_at = accepted_at if enqueued_at is None else _utc(enqueued_at, field="enqueued_at")
        if queued_at > accepted_at and queued_at - accepted_at > self.policy.max_future_skew:
            raise ValueError("enqueued_at exceeds the allowed future clock skew")

        with self._lock:
            if signal.case_id in self._active:
                raise ValueError(f"case {signal.case_id!r} is already queued")
            if len(self._active) >= self.policy.max_cases:
                raise OverflowError("review queue capacity is exhausted")

            self._sequence += 1
            try:
                deadline = queued_at + self.policy.review_sla
            except OverflowError as error:
                raise ValueError("review_sla produces an invalid deadline") from error
            queued = _QueuedCase(
                signal=signal,
                enqueued_at=queued_at,
                deadline=deadline,
                priority=priority,
                sequence=self._sequence,
            )
            self._active[signal.case_id] = queued
            heapq.heappush(
                self._priority_heap,
                (-priority, queued_at, signal.case_id, queued.sequence),
            )
            heapq.heappush(
                self._deadline_heap,
                (queued.deadline, -priority, queued_at, signal.case_id, queued.sequence),
            )

    def pop(self, *, now: datetime | None = None) -> FraudSignal:
        observed_at = _utc(self._clock() if now is None else now, field="now")
        with self._lock:
            if not self._active:
                raise IndexError("review queue is empty")

            self._discard_stale_deadlines()
            if self._deadline_heap[0][0] <= observed_at:
                *_, case_id, sequence = heapq.heappop(self._deadline_heap)
            else:
                self._discard_stale_priorities()
                *_, case_id, sequence = heapq.heappop(self._priority_heap)

            queued = self._active.get(case_id)
            if queued is None or queued.sequence != sequence:  # pragma: no cover - guarded above
                raise RuntimeError("review queue heap lost synchronization")
            del self._active[case_id]
            self._compact_heaps_if_needed()
            return queued.signal

    def snapshot(self, *, now: datetime | None = None) -> QueueSnapshot:
        observed_at = _utc(self._clock() if now is None else now, field="now")
        with self._lock:
            cases = tuple(self._active.values())
            size = len(cases)
            overdue = sum(item.deadline <= observed_at for item in cases)
            oldest_age = max(
                (max(0.0, (observed_at - item.enqueued_at).total_seconds()) for item in cases),
                default=0.0,
            )
            earliest_deadline = min((item.deadline for item in cases), default=None)
            return QueueSnapshot(
                size=size,
                capacity=self.policy.max_cases,
                utilization=round(size / self.policy.max_cases, 6),
                overdue_cases=overdue,
                oldest_age_seconds=oldest_age,
                earliest_deadline=earliest_deadline,
            )

    def _discard_stale_deadlines(self) -> None:
        while self._deadline_heap:
            *_, case_id, sequence = self._deadline_heap[0]
            queued = self._active.get(case_id)
            if queued is not None and queued.sequence == sequence:
                return
            heapq.heappop(self._deadline_heap)

    def _discard_stale_priorities(self) -> None:
        while self._priority_heap:
            *_, case_id, sequence = self._priority_heap[0]
            queued = self._active.get(case_id)
            if queued is not None and queued.sequence == sequence:
                return
            heapq.heappop(self._priority_heap)

    def _compact_heaps_if_needed(self) -> None:
        """Bound lazy-deletion evidence during long-running queue churn."""

        maximum_entries = max(64, len(self._active) * 2)
        if (
            len(self._priority_heap) <= maximum_entries
            and len(self._deadline_heap) <= maximum_entries
        ):
            return

        cases = tuple(self._active.values())
        self._priority_heap = [
            (-item.priority, item.enqueued_at, item.signal.case_id, item.sequence) for item in cases
        ]
        self._deadline_heap = [
            (
                item.deadline,
                -item.priority,
                item.enqueued_at,
                item.signal.case_id,
                item.sequence,
            )
            for item in cases
        ]
        heapq.heapify(self._priority_heap)
        heapq.heapify(self._deadline_heap)

    def __len__(self) -> int:
        with self._lock:
            return len(self._active)


if __name__ == "__main__":
    queue = ReviewQueue()
    queue.push(FraudSignal("case-1", 0.82, 0.91, 3200, 500, 4))
    queue.push(FraudSignal("case-2", 0.60, 0.52, 16000, 9000, 2))
    queue.push(FraudSignal("case-3", 0.93, 0.98, 800, 50, 5))

    while queue:
        case = queue.pop()
        print(case.case_id, round(triage_priority(case), 3))
