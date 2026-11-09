from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from case_triage import FraudSignal, ReviewPolicy
from durable_review_queue import (
    DurableQueueError,
    DurableReviewPolicy,
    LeaseConflict,
    QueueEmpty,
    SQLiteReviewQueue,
)

NOW = datetime(2026, 11, 9, 5, 0, tzinfo=timezone.utc)


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


def queue_at(
    tmp_path: Path,
    *,
    max_cases: int = 100,
    review_sla: timedelta = timedelta(minutes=30),
) -> SQLiteReviewQueue:
    return SQLiteReviewQueue(
        tmp_path / "review.sqlite3",
        DurableReviewPolicy(
            review=ReviewPolicy(max_cases=max_cases, review_sla=review_sla),
            default_lease=timedelta(minutes=5),
            max_lease=timedelta(minutes=30),
        ),
        clock=lambda: NOW,
    )


def test_queue_persists_cases_across_instances(tmp_path: Path) -> None:
    first = queue_at(tmp_path)
    assert first.enqueue(signal("case-1", risk=0.9))

    reopened = queue_at(tmp_path)
    lease = reopened.claim_next("analyst-1", now=NOW)

    assert lease.signal.case_id == "case-1"
    assert lease.worker_id == "analyst-1"
    assert lease.token
    assert lease.token not in repr(lease)


def test_enqueue_is_idempotent_but_rejects_conflicting_evidence(
    tmp_path: Path,
) -> None:
    queue = queue_at(tmp_path)
    original = signal("case-1", risk=0.7)

    assert queue.enqueue(original)
    assert not queue.enqueue(original)
    with pytest.raises(DurableQueueError, match="different evidence"):
        queue.enqueue(replace(original, risk_score=0.8))

    assert queue.snapshot(now=NOW).active_cases == 1


def test_priority_before_sla_and_deadline_after_sla(tmp_path: Path) -> None:
    queue = queue_at(tmp_path, review_sla=timedelta(minutes=10))
    queue.enqueue(
        signal("old-low", risk=0.05),
        enqueued_at=NOW - timedelta(minutes=20),
    )
    queue.enqueue(signal("new-high", risk=1.0), enqueued_at=NOW)

    overdue = queue.claim_next("analyst-1", now=NOW)
    assert overdue.signal.case_id == "old-low"
    queue.complete(overdue, completion_id="completion-old", completed_at=NOW)

    assert queue.claim_next("analyst-1", now=NOW).signal.case_id == "new-high"


def test_material_priority_wins_when_no_case_is_overdue(tmp_path: Path) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("low", risk=0.1, confidence=0.99))
    queue.enqueue(signal("high", risk=0.95, confidence=0.4, amount=9000))

    assert queue.claim_next("analyst-1", now=NOW).signal.case_id == "high"


def test_concurrent_workers_cannot_claim_the_same_case(tmp_path: Path) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("case-1"))

    def claim(worker: str) -> str:
        try:
            return queue.claim_next(worker, now=NOW).signal.case_id
        except QueueEmpty:
            return "empty"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(claim, [f"analyst-{index}" for index in range(8)]))

    assert results.count("case-1") == 1
    assert results.count("empty") == 7


def test_expired_lease_is_reclaimed_with_a_new_token(tmp_path: Path) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("case-1"))
    first = queue.claim_next("analyst-1", now=NOW, lease_duration=timedelta(seconds=30))

    reclaimed = queue.claim_next("analyst-2", now=NOW + timedelta(seconds=31))

    assert reclaimed.signal.case_id == "case-1"
    assert reclaimed.worker_id == "analyst-2"
    assert reclaimed.token != first.token
    with pytest.raises(LeaseConflict):
        queue.complete(
            first,
            completion_id="stale-completion",
            completed_at=NOW + timedelta(seconds=31),
        )


def test_renewal_extends_exclusive_lease(tmp_path: Path) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("case-1"))
    lease = queue.claim_next("analyst-1", now=NOW, lease_duration=timedelta(seconds=30))

    renewed = queue.renew(
        lease,
        now=NOW + timedelta(seconds=10),
        lease_duration=timedelta(minutes=2),
    )

    assert renewed.expires_at == NOW + timedelta(minutes=2, seconds=30)
    with pytest.raises(QueueEmpty):
        queue.claim_next("analyst-2", now=NOW + timedelta(minutes=1))


def test_completion_is_durable_and_exact_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    queue = queue_at(tmp_path, max_cases=1)
    queue.enqueue(signal("case-1"))
    lease = queue.claim_next("analyst-1", now=NOW)

    assert queue.complete(lease, completion_id="review-123", completed_at=NOW)
    assert not queue.complete(lease, completion_id="review-123", completed_at=NOW)
    with pytest.raises(LeaseConflict, match="different completion_id"):
        queue.complete(lease, completion_id="review-456", completed_at=NOW)

    assert queue.enqueue(signal("case-2"))
    snapshot = queue.snapshot(now=NOW)
    assert snapshot.active_cases == 1
    assert snapshot.completed_cases == 1


def test_wrong_worker_or_token_cannot_renew_or_complete(tmp_path: Path) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("case-1"))
    lease = queue.claim_next("analyst-1", now=NOW)

    with pytest.raises(LeaseConflict, match="owner or token"):
        queue.renew(replace(lease, worker_id="analyst-2"), now=NOW)
    with pytest.raises(LeaseConflict, match="owner or token"):
        queue.complete(
            replace(lease, token="wrong-token"),
            completion_id="review-123",
            completed_at=NOW,
        )


def test_completion_ids_are_unique_across_cases(tmp_path: Path) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("case-1"))
    queue.enqueue(signal("case-2"))
    first = queue.claim_next("analyst-1", now=NOW)
    queue.complete(first, completion_id="review-123", completed_at=NOW)
    second = queue.claim_next("analyst-2", now=NOW)

    with pytest.raises(LeaseConflict, match="another case"):
        queue.complete(second, completion_id="review-123", completed_at=NOW)


def test_snapshot_exposes_pressure_without_case_identifiers(tmp_path: Path) -> None:
    queue = queue_at(tmp_path, max_cases=4, review_sla=timedelta(minutes=10))
    queue.enqueue(
        signal("overdue"),
        enqueued_at=NOW - timedelta(minutes=20),
    )
    queue.enqueue(signal("fresh"), enqueued_at=NOW - timedelta(minutes=2))
    queue.claim_next(
        "analyst-1",
        now=NOW - timedelta(minutes=1),
        lease_duration=timedelta(seconds=30),
    )

    snapshot = queue.snapshot(now=NOW)

    assert snapshot.active_cases == 2
    assert snapshot.queued_cases == 1
    assert snapshot.leased_cases == 1
    assert snapshot.expired_leases == 1
    assert snapshot.overdue_cases == 1
    assert snapshot.utilization == 0.5
    assert snapshot.oldest_age_seconds == 1200
    assert "fresh" not in repr(snapshot)
    assert "case_id" not in repr(snapshot)


def test_capacity_check_is_transactional_under_concurrent_enqueue(
    tmp_path: Path,
) -> None:
    queue = queue_at(tmp_path, max_cases=10)

    def enqueue(index: int) -> str:
        try:
            queue.enqueue(signal(f"case-{index}"))
            return "accepted"
        except OverflowError:
            return "full"

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(enqueue, range(20)))

    assert results.count("accepted") == 10
    assert results.count("full") == 10
    assert queue.snapshot(now=NOW).active_cases == 10


@pytest.mark.parametrize(
    "duration",
    [timedelta(0), timedelta(seconds=-1), timedelta(hours=1)],
)
def test_unsafe_lease_durations_are_rejected(tmp_path: Path, duration: timedelta) -> None:
    queue = queue_at(tmp_path)
    queue.enqueue(signal("case-1"))

    with pytest.raises(ValueError, match="max_lease"):
        queue.claim_next("analyst-1", now=NOW, lease_duration=duration)


def test_private_storage_permissions_are_enforced(tmp_path: Path) -> None:
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o755)

    with pytest.raises(DurableQueueError, match="owner-only"):
        SQLiteReviewQueue(unsafe_parent / "queue.sqlite3")

    safe_parent = tmp_path / "safe"
    safe_parent.mkdir(mode=0o700)
    database = safe_parent / "queue.sqlite3"
    SQLiteReviewQueue(database)
    database.chmod(0o644)

    with pytest.raises(DurableQueueError, match="owner-only"):
        SQLiteReviewQueue(database)


def test_symlink_database_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    target.chmod(0o600)
    link = tmp_path / "queue.sqlite3"
    os.symlink(target, link)

    with pytest.raises(DurableQueueError, match="single-link regular file"):
        SQLiteReviewQueue(link)
