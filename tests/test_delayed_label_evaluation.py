from datetime import datetime, timedelta, timezone

import pytest

from src.delayed_label_evaluation import (
    EvaluationPolicy,
    Outcome,
    Prediction,
    audit_delayed_labels,
)


AS_OF = datetime(2026, 11, 16, tzinfo=timezone.utc)


def policy(**changes: object) -> EvaluationPolicy:
    values = {
        "as_of": AS_OF,
        "window_start": AS_OF - timedelta(days=30),
        "window_end": AS_OF,
        "label_maturity_delay": timedelta(days=7),
        "decision_threshold": 0.5,
        "minimum_mature_predictions": 2,
        "minimum_positive_labels": 1,
        "minimum_negative_labels": 1,
        "maximum_missing_label_rate": 0.0,
    }
    values.update(changes)
    return EvaluationPolicy(**values)


def prediction(transaction_id: str, *, age_days: int, score: float) -> Prediction:
    return Prediction(transaction_id, AS_OF - timedelta(days=age_days), score)


def outcome(transaction_id: str, *, age_days: int, fraud: bool) -> Outcome:
    return Outcome(transaction_id, AS_OF - timedelta(days=age_days), fraud)


def test_metrics_use_only_point_in_time_mature_labels() -> None:
    predictions = [
        prediction("tp", age_days=12, score=0.9),
        prediction("fp", age_days=11, score=0.7),
        prediction("tn", age_days=10, score=0.2),
        prediction("pending-positive", age_days=2, score=0.8),
    ]
    outcomes = [
        outcome("tp", age_days=9, fraud=True),
        outcome("fp", age_days=8, fraud=False),
        outcome("tn", age_days=8, fraud=False),
        outcome("pending-positive", age_days=1, fraud=True),
    ]

    report = audit_delayed_labels(predictions, outcomes, policy())

    assert report.accepted
    assert report.mature_prediction_count == 3
    assert report.pending_prediction_count == 1
    assert report.premature_outcome_count == 1
    assert report.evaluated_label_count == 3
    assert report.precision == 0.5
    assert report.recall == 1.0
    assert report.false_positive_rate == 0.5
    assert report.brier_score == pytest.approx((0.01 + 0.49 + 0.04) / 3)
    assert report.average_precision == 1.0


def test_future_outcome_is_not_leaked_into_metrics() -> None:
    predictions = [
        prediction("future", age_days=12, score=0.9),
        prediction("negative", age_days=11, score=0.1),
    ]
    outcomes = [
        Outcome("future", AS_OF + timedelta(hours=1), True),
        outcome("negative", age_days=8, fraud=False),
    ]

    report = audit_delayed_labels(
        predictions,
        outcomes,
        policy(maximum_missing_label_rate=0.5, minimum_positive_labels=0),
    )

    assert report.accepted
    assert report.future_outcome_count == 1
    assert report.missing_mature_label_count == 1
    assert report.evaluated_label_count == 1
    assert report.brier_score == pytest.approx(0.01)


def test_missing_mature_labels_fail_coverage_gate() -> None:
    predictions = [
        prediction("missing", age_days=12, score=0.9),
        prediction("positive", age_days=11, score=0.8),
        prediction("negative", age_days=10, score=0.2),
    ]
    outcomes = [
        outcome("positive", age_days=9, fraud=True),
        outcome("negative", age_days=8, fraud=False),
    ]

    report = audit_delayed_labels(predictions, outcomes, policy())

    assert not report.accepted
    assert report.label_coverage == pytest.approx(2 / 3)
    assert report.findings == ("missing_label_rate_exceeded",)


def test_class_support_is_an_explicit_admission_requirement() -> None:
    predictions = [
        prediction("one", age_days=12, score=0.2),
        prediction("two", age_days=11, score=0.3),
    ]
    outcomes = [
        outcome("one", age_days=10, fraud=False),
        outcome("two", age_days=9, fraud=False),
    ]

    report = audit_delayed_labels(predictions, outcomes, policy())

    assert not report.accepted
    assert report.average_precision is None
    assert report.recall is None
    assert report.findings == ("insufficient_positive_labels",)


def test_evidence_digest_is_order_independent() -> None:
    predictions = [
        prediction("positive", age_days=12, score=0.8),
        prediction("negative", age_days=11, score=0.2),
    ]
    outcomes = [
        outcome("positive", age_days=10, fraud=True),
        outcome("negative", age_days=9, fraud=False),
    ]

    first = audit_delayed_labels(predictions, outcomes, policy())
    second = audit_delayed_labels(reversed(predictions), reversed(outcomes), policy())

    assert first.evidence_sha256 == second.evidence_sha256
    assert len(first.evidence_sha256) == 64

    changed_label = [
        outcome("positive", age_days=10, fraud=False),
        outcome("negative", age_days=9, fraud=False),
    ]
    changed = audit_delayed_labels(
        predictions,
        changed_label,
        policy(minimum_positive_labels=0),
    )
    assert changed.evidence_sha256 != first.evidence_sha256


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_scores_fail_closed(score: float) -> None:
    with pytest.raises(ValueError, match="risk_score"):
        audit_delayed_labels([prediction("bad", age_days=12, score=score)], [], policy())


def test_duplicate_ids_fail_closed() -> None:
    duplicate = prediction("duplicate", age_days=12, score=0.5)
    with pytest.raises(ValueError, match="duplicate prediction"):
        audit_delayed_labels([duplicate, duplicate], [], policy())


def test_outcome_cannot_predate_its_prediction() -> None:
    scored = prediction("causality", age_days=10, score=0.5)
    earlier_outcome = outcome("causality", age_days=11, fraud=True)
    with pytest.raises(ValueError, match="cannot precede"):
        audit_delayed_labels([scored], [earlier_outcome], policy())


def test_naive_timestamps_and_unbounded_inputs_fail_closed() -> None:
    naive = Prediction("naive", datetime(2026, 1, 1), 0.5)
    with pytest.raises(ValueError, match="timezone-aware"):
        audit_delayed_labels([naive], [], policy())

    with pytest.raises(ValueError, match="maximum_records"):
        audit_delayed_labels(
            [prediction("one", age_days=10, score=0.5)],
            [outcome("one", age_days=9, fraud=False)],
            policy(maximum_records=1),
        )
