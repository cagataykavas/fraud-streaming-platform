from __future__ import annotations

from dataclasses import dataclass
import heapq


@dataclass(frozen=True)
class FraudSignal:
    case_id: str
    risk_score: float
    confidence: float
    customer_value: float
    amount: float
    evidence_count: int


def triage_priority(signal: FraudSignal) -> float:
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
    def __init__(self) -> None:
        self._heap: list[tuple[float, str, FraudSignal]] = []

    def push(self, signal: FraudSignal) -> None:
        priority = triage_priority(signal)
        heapq.heappush(self._heap, (-priority, signal.case_id, signal))

    def pop(self) -> FraudSignal:
        if not self._heap:
            raise IndexError("review queue is empty")
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)


if __name__ == "__main__":
    queue = ReviewQueue()
    queue.push(FraudSignal("case-1", 0.82, 0.91, 3200, 500, 4))
    queue.push(FraudSignal("case-2", 0.60, 0.52, 16000, 9000, 2))
    queue.push(FraudSignal("case-3", 0.93, 0.98, 800, 50, 5))

    while queue:
        case = queue.pop()
        print(case.case_id, round(triage_priority(case), 3))
