from datetime import datetime, timedelta, timezone

import pytest

from case_triage import FraudSignal, ReviewQueue
from src.scoring import FraudAssessment, StreamingFraudScorer, Transaction
from src.worker import ProcessingResult, Worker


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.returned_id: str | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, _query: str, parameters: tuple[object, ...]) -> None:
        transaction_id = str(parameters[0])
        if transaction_id in self.connection.persisted_ids:
            self.returned_id = None
        else:
            self.returned_id = transaction_id
            self.connection.pending_id = transaction_id

    def fetchone(self) -> tuple[str] | None:
        if self.returned_id is None:
            return None
        return (self.returned_id,)


class FakeConnection:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.persisted_ids: set[str] = set()
        self.pending_id: str | None = None
        self.commit_count = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commit_count += 1
        if self.fail_commit:
            raise RuntimeError("database commit failed")
        if self.pending_id is not None:
            self.persisted_ids.add(self.pending_id)
            self.pending_id = None


def worker() -> Worker:
    instance = Worker.__new__(Worker)
    instance.scorer = StreamingFraudScorer()
    return instance


def transaction(
    index: int, time: datetime, *, country: str = "TR", amount: float = 100.0
) -> Transaction:
    return Transaction(
        transaction_id=f"tx-{index}",
        customer_id="cust-1",
        event_time=time.isoformat().replace("+00:00", "Z"),
        amount=amount,
        merchant_category="grocery",
        country_code=country,
    )


def test_velocity_signal_is_stateful() -> None:
    scorer = StreamingFraudScorer()
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    result = None
    for index in range(6):
        result = scorer.score(transaction(index, now + timedelta(seconds=index * 20)))
    assert result is not None
    assert "transaction_velocity_5m" in {item.name for item in result.signals}
    assert result.risk_score > 0


def test_assess_is_side_effect_free_until_commit() -> None:
    scorer = StreamingFraudScorer()
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    item = transaction(1, now)

    first = scorer.assess(item)
    second = scorer.assess(item)

    assert first == second
    assert len(scorer.history["cust-1"]) == 0
    scorer.commit(item)
    assert len(scorer.history["cust-1"]) == 1


def test_score_compatibility_path_commits_exactly_once() -> None:
    scorer = StreamingFraudScorer()
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)

    scorer.score(transaction(1, now))

    assert len(scorer.history["cust-1"]) == 1


def test_worker_commits_state_only_after_durable_insert() -> None:
    instance = worker()
    connection = FakeConnection()
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    item = transaction(1, now)

    result = instance.process(connection, item.__dict__)

    assert result.inserted is True
    assert connection.persisted_ids == {"tx-1"}
    assert len(instance.scorer.history["cust-1"]) == 1


def test_duplicate_replay_does_not_mutate_state() -> None:
    instance = worker()
    connection = FakeConnection()
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    item = transaction(1, now)

    first = instance.process(connection, item.__dict__)
    replay = instance.process(connection, item.__dict__)

    assert first.inserted is True
    assert replay.inserted is False
    assert connection.commit_count == 2
    assert len(instance.scorer.history["cust-1"]) == 1


def test_duplicate_non_allow_assessment_is_not_eligible_for_alert() -> None:
    item = transaction(1, datetime(2026, 8, 1, tzinfo=timezone.utc))
    assessment = FraudAssessment(
        transaction_id=item.transaction_id,
        customer_id=item.customer_id,
        event_time=item.event_time,
        risk_score=0.8,
        confidence=0.9,
        action="block_and_review",
        signals=(),
    )

    assert ProcessingResult(item, assessment, inserted=True).should_emit_alert is True
    assert ProcessingResult(item, assessment, inserted=False).should_emit_alert is False


def test_failed_database_commit_leaves_state_unchanged() -> None:
    instance = worker()
    connection = FakeConnection(fail_commit=True)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="database commit failed"):
        instance.process(connection, transaction(1, now).__dict__)

    assert connection.persisted_ids == set()
    assert len(instance.scorer.history["cust-1"]) == 0


def test_invalid_history_size_fails_closed() -> None:
    with pytest.raises(ValueError, match="history_size must be positive"):
        StreamingFraudScorer(history_size=0)


def test_rapid_country_change_is_detected() -> None:
    scorer = StreamingFraudScorer()
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    scorer.score(transaction(1, now, country="TR"))
    result = scorer.score(transaction(2, now + timedelta(minutes=2), country="US"))
    assert "rapid_country_change" in {item.name for item in result.signals}


def test_review_queue_prioritizes_material_uncertainty() -> None:
    queue = ReviewQueue()
    queue.push(FraudSignal("low", 0.30, 0.95, 1000, 50, 1))
    queue.push(FraudSignal("high", 0.85, 0.60, 18000, 9000, 4))
    assert queue.pop().case_id == "high"
