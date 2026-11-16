from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, TypeVar


EVALUATION_VERSION = "delay-aware-fraud-evaluation/v1"
T = TypeVar("T")


@dataclass(frozen=True)
class Prediction:
    transaction_id: str
    scored_at: datetime
    risk_score: float


@dataclass(frozen=True)
class Outcome:
    transaction_id: str
    observed_at: datetime
    is_fraud: bool


@dataclass(frozen=True)
class EvaluationPolicy:
    as_of: datetime
    window_start: datetime
    window_end: datetime
    label_maturity_delay: timedelta
    decision_threshold: float = 0.5
    minimum_mature_predictions: int = 100
    minimum_positive_labels: int = 1
    minimum_negative_labels: int = 1
    maximum_missing_label_rate: float = 0.05
    maximum_records: int = 1_000_000


@dataclass(frozen=True)
class EvaluationReport:
    version: str
    accepted: bool
    evidence_sha256: str
    as_of: str
    window_start: str
    window_end: str
    maturity_cutoff: str
    decision_threshold: float
    prediction_count: int
    mature_prediction_count: int
    pending_prediction_count: int
    evaluated_label_count: int
    missing_mature_label_count: int
    future_outcome_count: int
    premature_outcome_count: int
    positive_label_count: int
    negative_label_count: int
    label_coverage: float | None
    brier_score: float | None
    average_precision: float | None
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["findings"] = list(self.findings)
        return payload


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _validate_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string of at most 256 characters")


def _validate_policy(policy: EvaluationPolicy) -> tuple[datetime, datetime, datetime]:
    as_of = _utc(policy.as_of, "policy.as_of")
    window_start = _utc(policy.window_start, "policy.window_start")
    window_end = _utc(policy.window_end, "policy.window_end")
    if not window_start < window_end <= as_of:
        raise ValueError("evaluation window must be non-empty and end no later than as_of")
    if policy.label_maturity_delay < timedelta(0):
        raise ValueError("label_maturity_delay must be non-negative")
    if not 0.0 <= policy.decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be in [0, 1]")
    if policy.minimum_mature_predictions < 1:
        raise ValueError("minimum_mature_predictions must be positive")
    if policy.minimum_positive_labels < 0 or policy.minimum_negative_labels < 0:
        raise ValueError("minimum label counts must be non-negative")
    if not 0.0 <= policy.maximum_missing_label_rate <= 1.0:
        raise ValueError("maximum_missing_label_rate must be in [0, 1]")
    if policy.maximum_records < 1:
        raise ValueError("maximum_records must be positive")
    return as_of, window_start, window_end


def _average_precision(rows: list[tuple[float, bool, str]], positives: int) -> float | None:
    if positives == 0:
        return None
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    ordered = sorted(rows, key=lambda row: (-row[0], row[2]))
    while index < len(ordered):
        score = ordered[index][0]
        group_end = index
        group_positives = 0
        while group_end < len(ordered) and ordered[group_end][0] == score:
            group_positives += int(ordered[group_end][1])
            group_end += 1
        seen += group_end - index
        true_positives += group_positives
        recall = true_positives / positives
        result += (recall - previous_recall) * (true_positives / seen)
        previous_recall = recall
        index = group_end
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _bounded_rows(values: Iterable[T], limit: int) -> list[T]:
    rows: list[T] = []
    for value in values:
        if len(rows) >= limit:
            raise ValueError("input exceeds maximum_records")
        rows.append(value)
    return rows


def audit_delayed_labels(
    predictions: Iterable[Prediction],
    outcomes: Iterable[Outcome],
    policy: EvaluationPolicy,
) -> EvaluationReport:
    """Evaluate fraud scores without treating pending labels as negatives.

    Predictions newer than the configured maturity cutoff are always excluded,
    even when an early positive outcome is already known. Outcomes observed after
    ``as_of`` are also excluded. This keeps the evaluated cohort independent of
    label arrival speed and prevents future adjudications from leaking backward.
    """

    as_of, window_start, window_end = _validate_policy(policy)
    prediction_rows = _bounded_rows(predictions, policy.maximum_records)
    outcome_rows = _bounded_rows(outcomes, policy.maximum_records - len(prediction_rows) + 1)
    if len(prediction_rows) + len(outcome_rows) > policy.maximum_records:
        raise ValueError("input exceeds maximum_records")

    prediction_by_id: dict[str, tuple[Prediction, datetime]] = {}
    for prediction in prediction_rows:
        _validate_id(prediction.transaction_id, "prediction.transaction_id")
        if prediction.transaction_id in prediction_by_id:
            raise ValueError(f"duplicate prediction transaction_id: {prediction.transaction_id}")
        scored_at = _utc(prediction.scored_at, "prediction.scored_at")
        if isinstance(prediction.risk_score, bool) or not isinstance(
            prediction.risk_score, (int, float)
        ):
            raise ValueError("prediction.risk_score must be a real number")
        if not math.isfinite(prediction.risk_score) or not 0.0 <= prediction.risk_score <= 1.0:
            raise ValueError("prediction.risk_score must be finite and in [0, 1]")
        prediction_by_id[prediction.transaction_id] = (prediction, scored_at)

    outcome_by_id: dict[str, tuple[Outcome, datetime]] = {}
    for outcome in outcome_rows:
        _validate_id(outcome.transaction_id, "outcome.transaction_id")
        if outcome.transaction_id in outcome_by_id:
            raise ValueError(f"duplicate outcome transaction_id: {outcome.transaction_id}")
        if not isinstance(outcome.is_fraud, bool):
            raise ValueError("outcome.is_fraud must be a boolean")
        observed_at = _utc(outcome.observed_at, "outcome.observed_at")
        prediction_pair = prediction_by_id.get(outcome.transaction_id)
        if prediction_pair is not None and observed_at < prediction_pair[1]:
            raise ValueError("outcome.observed_at cannot precede prediction.scored_at")
        outcome_by_id[outcome.transaction_id] = (outcome, observed_at)

    selected = [
        (prediction, scored_at)
        for prediction, scored_at in prediction_by_id.values()
        if window_start <= scored_at < window_end
    ]
    maturity_cutoff = as_of - policy.label_maturity_delay
    mature = [
        (prediction, scored_at)
        for prediction, scored_at in selected
        if scored_at <= maturity_cutoff
    ]
    pending = [
        (prediction, scored_at) for prediction, scored_at in selected if scored_at > maturity_cutoff
    ]

    evaluated: list[tuple[float, bool, str]] = []
    missing_mature = 0
    future_outcomes = 0
    premature_outcomes = 0
    evidence_rows: list[dict[str, object]] = []

    for prediction, scored_at in mature:
        outcome_pair = outcome_by_id.get(prediction.transaction_id)
        outcome_evidence: dict[str, object] = {}
        if outcome_pair is None:
            missing_mature += 1
            outcome_state = "missing"
        elif outcome_pair[1] > as_of:
            missing_mature += 1
            future_outcomes += 1
            outcome_state = "future"
            outcome_evidence = {
                "outcome_observed_at": _iso(outcome_pair[1]),
                "is_fraud": outcome_pair[0].is_fraud,
            }
        else:
            evaluated.append(
                (float(prediction.risk_score), outcome_pair[0].is_fraud, prediction.transaction_id)
            )
            outcome_state = "observed"
            outcome_evidence = {
                "outcome_observed_at": _iso(outcome_pair[1]),
                "is_fraud": outcome_pair[0].is_fraud,
            }
        evidence_rows.append(
            {
                "transaction_id": prediction.transaction_id,
                "scored_at": _iso(scored_at),
                "risk_score": float(prediction.risk_score),
                "maturity": "mature",
                "outcome_state": outcome_state,
                **outcome_evidence,
            }
        )

    for prediction, scored_at in pending:
        outcome_pair = outcome_by_id.get(prediction.transaction_id)
        if outcome_pair is not None and outcome_pair[1] <= as_of:
            premature_outcomes += 1
        row: dict[str, object] = {
            "transaction_id": prediction.transaction_id,
            "scored_at": _iso(scored_at),
            "risk_score": float(prediction.risk_score),
            "maturity": "pending",
        }
        if outcome_pair is not None:
            row.update(
                outcome_observed_at=_iso(outcome_pair[1]),
                is_fraud=outcome_pair[0].is_fraud,
            )
        evidence_rows.append(row)

    positive_count = sum(int(label) for _, label, _ in evaluated)
    negative_count = len(evaluated) - positive_count
    predicted_positive = sum(int(score >= policy.decision_threshold) for score, _, _ in evaluated)
    true_positive = sum(
        int(score >= policy.decision_threshold and label) for score, label, _ in evaluated
    )
    false_positive = predicted_positive - true_positive

    findings: list[str] = []
    if len(mature) < policy.minimum_mature_predictions:
        findings.append("insufficient_mature_predictions")
    missing_rate = _ratio(missing_mature, len(mature))
    if missing_rate is not None and missing_rate > policy.maximum_missing_label_rate:
        findings.append("missing_label_rate_exceeded")
    if positive_count < policy.minimum_positive_labels:
        findings.append("insufficient_positive_labels")
    if negative_count < policy.minimum_negative_labels:
        findings.append("insufficient_negative_labels")

    canonical_evidence = {
        "version": EVALUATION_VERSION,
        "policy": {
            "as_of": _iso(as_of),
            "window_start": _iso(window_start),
            "window_end": _iso(window_end),
            "label_maturity_delay_seconds": policy.label_maturity_delay.total_seconds(),
            "decision_threshold": policy.decision_threshold,
            "minimum_mature_predictions": policy.minimum_mature_predictions,
            "minimum_positive_labels": policy.minimum_positive_labels,
            "minimum_negative_labels": policy.minimum_negative_labels,
            "maximum_missing_label_rate": policy.maximum_missing_label_rate,
        },
        "rows": sorted(evidence_rows, key=lambda row: str(row["transaction_id"])),
    }
    evidence_sha256 = hashlib.sha256(
        json.dumps(canonical_evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    brier_score = (
        sum((score - int(label)) ** 2 for score, label, _ in evaluated) / len(evaluated)
        if evaluated
        else None
    )
    return EvaluationReport(
        version=EVALUATION_VERSION,
        accepted=not findings,
        evidence_sha256=evidence_sha256,
        as_of=_iso(as_of),
        window_start=_iso(window_start),
        window_end=_iso(window_end),
        maturity_cutoff=_iso(maturity_cutoff),
        decision_threshold=policy.decision_threshold,
        prediction_count=len(selected),
        mature_prediction_count=len(mature),
        pending_prediction_count=len(pending),
        evaluated_label_count=len(evaluated),
        missing_mature_label_count=missing_mature,
        future_outcome_count=future_outcomes,
        premature_outcome_count=premature_outcomes,
        positive_label_count=positive_count,
        negative_label_count=negative_count,
        label_coverage=_ratio(len(evaluated), len(mature)),
        brier_score=brier_score,
        average_precision=_average_precision(evaluated, positive_count),
        precision=_ratio(true_positive, predicted_positive),
        recall=_ratio(true_positive, positive_count),
        false_positive_rate=_ratio(false_positive, negative_count),
        findings=tuple(findings),
    )
