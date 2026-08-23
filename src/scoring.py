from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Deque


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    customer_id: str
    event_time: str
    amount: float
    merchant_category: str
    country_code: str


@dataclass(frozen=True)
class Signal:
    name: str
    weight: float
    value: float
    explanation: str


@dataclass(frozen=True)
class FraudAssessment:
    transaction_id: str
    customer_id: str
    event_time: str
    risk_score: float
    confidence: float
    action: str
    signals: tuple[Signal, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["signals"] = [asdict(item) for item in self.signals]
        return payload


@dataclass(frozen=True)
class HistoricTransaction:
    event_time: datetime
    amount: float
    country_code: str


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class StreamingFraudScorer:
    """Stateful reference scorer for event-by-event fraud feature computation.

    The scorer keeps a bounded per-customer history. It is intentionally
    deterministic so it can be unit tested and compared with Spark/Kafka
    implementations of the same features.
    """

    def __init__(self, history_size: int = 200) -> None:
        self.history_size = history_size
        self.history: dict[str, Deque[HistoricTransaction]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def score(self, transaction: Transaction) -> FraudAssessment:
        now = parse_time(transaction.event_time)
        history = self.history[transaction.customer_id]
        recent_5m = [item for item in history if now - item.event_time <= timedelta(minutes=5)]
        recent_15m = [item for item in history if now - item.event_time <= timedelta(minutes=15)]
        amounts = [item.amount for item in history]
        signals: list[Signal] = []

        velocity_count = len(recent_5m) + 1
        if velocity_count >= 6:
            severity = min(1.0, (velocity_count - 5) / 8)
            signals.append(
                Signal(
                    "transaction_velocity_5m",
                    0.30,
                    severity,
                    f"{velocity_count} transactions observed in a 5-minute window",
                )
            )

        if len(amounts) >= 10:
            sigma = pstdev(amounts)
            z_score = (transaction.amount - mean(amounts)) / sigma if sigma > 1e-9 else 0.0
            if z_score >= 3.0:
                signals.append(
                    Signal(
                        "amount_outlier",
                        0.28,
                        min(1.0, z_score / 6.0),
                        f"amount is {z_score:.2f} standard deviations above customer history",
                    )
                )
        elif transaction.amount >= 5000:
            signals.append(
                Signal("large_amount_cold_start", 0.18, min(1.0, transaction.amount / 20000), "large amount before sufficient customer history exists")
            )

        recent_countries = {item.country_code for item in recent_15m}
        if recent_countries and transaction.country_code not in recent_countries:
            signals.append(
                Signal(
                    "rapid_country_change",
                    0.24,
                    1.0,
                    f"country changed from {sorted(recent_countries)} to {transaction.country_code} within 15 minutes",
                )
            )

        if transaction.merchant_category.lower() in {"crypto", "cash_equivalent", "gambling"}:
            signals.append(
                Signal(
                    "high_risk_merchant_category",
                    0.16,
                    0.85,
                    f"merchant category {transaction.merchant_category!r} is configured as elevated risk",
                )
            )

        weighted = sum(item.weight * item.value for item in signals)
        # Saturating transformation makes several moderate signals compound while
        # keeping the public score in [0, 1].
        risk_score = 1.0 - math.exp(-2.4 * weighted)
        history_factor = min(1.0, len(history) / 25)
        evidence_factor = min(1.0, len(signals) / 3)
        confidence = min(0.99, 0.45 + 0.35 * history_factor + 0.20 * evidence_factor)

        if risk_score >= 0.78:
            action = "block_and_review"
        elif risk_score >= 0.52:
            action = "step_up_and_review"
        elif risk_score >= 0.30:
            action = "monitor"
        else:
            action = "allow"

        history.append(HistoricTransaction(now, transaction.amount, transaction.country_code))
        return FraudAssessment(
            transaction_id=transaction.transaction_id,
            customer_id=transaction.customer_id,
            event_time=transaction.event_time,
            risk_score=round(risk_score, 6),
            confidence=round(confidence, 6),
            action=action,
            signals=tuple(signals),
        )
