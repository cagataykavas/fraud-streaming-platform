from datetime import datetime, timedelta, timezone

from case_triage import FraudSignal, ReviewQueue
from src.scoring import StreamingFraudScorer, Transaction


def transaction(index: int, time: datetime, *, country: str = "TR", amount: float = 100.0) -> Transaction:
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
