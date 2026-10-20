from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from case_triage import FraudSignal, ReviewPolicy, ReviewQueue, triage_priority


NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def signal(
    case_id: str,
    *,
    risk: float = 0.5,
    confidence: float = 0.8,
    value: float = 1000,
    amount: float = 100,
    evidence: int = 1,
) -> FraudSignal:
    return FraudSignal(case_id, risk, confidence, value, amount, evidence)


def test_material_priority_wins_before_deadline() -> None:
    queue = ReviewQueue(ReviewPolicy(review_sla=timedelta(minutes=30)), clock=lambda: NOW)
    queue.push(signal("low", risk=0.1, confidence=0.99))
    queue.push(signal("high", risk=0.95, confidence=0.4, amount=9000))

    assert queue.pop(now=NOW + timedelta(minutes=5)).case_id == "high"


def test_oldest_overdue_case_wins_over_new_high_risk_case() -> None:
    policy = ReviewPolicy(review_sla=timedelta(minutes=10))
    queue = ReviewQueue(policy, clock=lambda: NOW)
    queue.push(signal("old-low", risk=0.05), enqueued_at=NOW - timedelta(minutes=20))
    queue.push(signal("new-high", risk=1.0), enqueued_at=NOW)

    assert queue.pop(now=NOW).case_id == "old-low"


def test_overdue_cases_are_ordered_by_deadline_then_priority() -> None:
    queue = ReviewQueue(ReviewPolicy(review_sla=timedelta(minutes=5)), clock=lambda: NOW)
    queued_at = NOW - timedelta(minutes=10)
    queue.push(signal("low", risk=0.1), enqueued_at=queued_at)
    queue.push(signal("high", risk=0.9), enqueued_at=queued_at)

    assert queue.pop(now=NOW).case_id == "high"
    assert queue.pop(now=NOW).case_id == "low"


def test_duplicate_case_is_rejected_without_consuming_capacity() -> None:
    queue = ReviewQueue(ReviewPolicy(max_cases=2), clock=lambda: NOW)
    queue.push(signal("case-1"))

    with pytest.raises(ValueError, match="already queued"):
        queue.push(signal("case-1"))

    assert len(queue) == 1


def test_queue_capacity_fails_closed() -> None:
    queue = ReviewQueue(ReviewPolicy(max_cases=1), clock=lambda: NOW)
    queue.push(signal("case-1"))

    with pytest.raises(OverflowError, match="capacity"):
        queue.push(signal("case-2"))


def test_case_id_can_be_reused_without_stale_heap_corruption() -> None:
    queue = ReviewQueue(clock=lambda: NOW)
    queue.push(signal("reused", risk=0.9))
    assert queue.pop(now=NOW).case_id == "reused"
    queue.push(signal("other", risk=0.8))
    queue.push(signal("reused", risk=0.1))

    assert queue.pop(now=NOW).case_id == "other"
    assert queue.pop(now=NOW).case_id == "reused"


def test_repeated_churn_keeps_lazy_heaps_bounded() -> None:
    queue = ReviewQueue(ReviewPolicy(max_cases=2), clock=lambda: NOW)

    for index in range(1000):
        queue.push(signal(f"case-{index}"))
        assert queue.pop(now=NOW).case_id == f"case-{index}"

    assert len(queue._priority_heap) <= 64
    assert len(queue._deadline_heap) <= 64


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (signal("bad", risk=float("nan")), "risk_score must be finite"),
        (signal("bad", confidence=float("inf")), "confidence must be finite"),
        (signal("bad", amount=-1), "amount cannot be negative"),
        (signal("bad", value=-1), "customer_value cannot be negative"),
        (signal("bad", evidence=-1), "evidence_count cannot be negative"),
        (signal(" bad"), "surrounding whitespace"),
    ],
)
def test_invalid_signal_evidence_fails_closed(item: FraudSignal, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        triage_priority(item)


def test_naive_enqueue_timestamp_is_rejected() -> None:
    queue = ReviewQueue(clock=lambda: NOW)

    with pytest.raises(ValueError, match="timezone-aware"):
        queue.push(signal("case-1"), enqueued_at=NOW.replace(tzinfo=None))


def test_excessive_future_timestamp_is_rejected() -> None:
    queue = ReviewQueue(ReviewPolicy(max_future_skew=timedelta(seconds=5)), clock=lambda: NOW)

    with pytest.raises(ValueError, match="clock skew"):
        queue.push(signal("case-1"), enqueued_at=NOW + timedelta(seconds=6))


def test_deadline_overflow_is_rejected_without_queue_mutation() -> None:
    queue = ReviewQueue(ReviewPolicy(review_sla=timedelta.max), clock=lambda: NOW)

    with pytest.raises(ValueError, match="invalid deadline"):
        queue.push(signal("case-1"))

    assert len(queue) == 0


def test_snapshot_exposes_capacity_and_deadline_pressure() -> None:
    queue = ReviewQueue(
        ReviewPolicy(max_cases=4, review_sla=timedelta(minutes=10)),
        clock=lambda: NOW,
    )
    queue.push(signal("overdue"), enqueued_at=NOW - timedelta(minutes=20))
    queue.push(signal("fresh"), enqueued_at=NOW - timedelta(minutes=2))

    snapshot = queue.snapshot(now=NOW)

    assert snapshot.size == 2
    assert snapshot.capacity == 4
    assert snapshot.utilization == 0.5
    assert snapshot.overdue_cases == 1
    assert snapshot.oldest_age_seconds == 1200
    assert snapshot.earliest_deadline == NOW - timedelta(minutes=10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_cases": 0}, "positive"),
        ({"review_sla": timedelta(0)}, "positive"),
        ({"max_future_skew": timedelta(seconds=-1)}, "negative"),
    ],
)
def test_invalid_policy_fails_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReviewPolicy(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_cases": 1.5}, "integer"),
        ({"review_sla": 30}, "timedelta"),
        ({"max_future_skew": 5}, "timedelta"),
    ],
)
def test_invalid_policy_types_fail_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ReviewPolicy(**kwargs)


def test_concurrent_pushes_preserve_uniqueness_and_capacity() -> None:
    queue = ReviewQueue(ReviewPolicy(max_cases=500), clock=lambda: NOW)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda index: queue.push(signal(f"case-{index}")), range(500)))

    assert len(queue) == 500
    assert queue.snapshot(now=NOW).utilization == 1.0


def test_empty_queue_pop_is_explicit() -> None:
    with pytest.raises(IndexError, match="empty"):
        ReviewQueue(clock=lambda: NOW).pop(now=NOW)
