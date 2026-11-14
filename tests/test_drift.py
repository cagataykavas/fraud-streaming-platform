import json
import math
from pathlib import Path

import pytest

from src.drift import DriftPolicy, audit_score_drift, main


def repeated(values: list[float], count: int) -> list[float]:
    return (values * math.ceil(count / len(values)))[:count]


def test_identical_populations_are_stable() -> None:
    scores = repeated([0.02, 0.15, 0.28, 0.43, 0.67, 0.91], 120)

    report = audit_score_drift(scores, scores)

    assert report.status == "stable"
    assert report.population_stability_index == pytest.approx(0.0)
    assert sum(item.reference_count for item in report.bins) == 120
    assert sum(item.current_count for item in report.bins) == 120


def test_material_shift_crosses_alert_policy() -> None:
    reference = repeated([0.05, 0.12, 0.18, 0.25], 200)
    current = repeated([0.72, 0.84, 0.93], 100)

    report = audit_score_drift(reference, current)

    assert report.status == "alert"
    assert report.population_stability_index >= report.alert_threshold
    assert all(math.isfinite(item.psi_contribution) for item in report.bins)


def test_thresholds_are_policy_owned() -> None:
    reference = repeated([0.05, 0.15], 20)
    current = repeated([0.05, 0.25], 20)
    policy = DriftPolicy(
        minimum_reference_size=20,
        minimum_current_size=20,
        warning_threshold=0.0,
        alert_threshold=10.0,
    )

    report = audit_score_drift(reference, current, policy)

    assert report.status == "warning"


def test_boundary_scores_are_counted_once() -> None:
    policy = DriftPolicy(
        bin_edges=(0.0, 0.5, 1.0),
        minimum_reference_size=3,
        minimum_current_size=3,
    )

    report = audit_score_drift([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], policy)

    assert [item.reference_count for item in report.bins] == [1, 2]


@pytest.mark.parametrize("invalid", [[0.1, float("nan")], [0.1, 1.01], [0.1, True]])
def test_invalid_scores_fail_closed(invalid: list[float]) -> None:
    policy = DriftPolicy(minimum_reference_size=2, minimum_current_size=2)

    with pytest.raises(ValueError):
        audit_score_drift(invalid, [0.1, 0.2], policy)


def test_undersized_population_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        audit_score_drift([0.1], repeated([0.1], 50))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bin_edges": (0.0, 0.5, 0.5, 1.0)},
        {"bin_edges": (0.1, 1.0)},
        {"smoothing_count": 0.0},
        {"warning_threshold": 0.3, "alert_threshold": 0.2},
    ],
)
def test_invalid_policy_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DriftPolicy(**kwargs)


def write_scores(path: Path, values: list[float]) -> None:
    path.write_text(json.dumps({"scores": values}), encoding="utf-8")


def test_cli_writes_deterministic_report(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    current = tmp_path / "current.json"
    output = tmp_path / "report.json"
    values = repeated([0.05, 0.15, 0.35, 0.75], 12)
    write_scores(reference, values)
    write_scores(current, values)

    exit_code = main(
        [
            str(reference),
            str(current),
            "--output",
            str(output),
            "--minimum-reference-size",
            "12",
            "--minimum-current-size",
            "12",
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "stable"


def test_cli_returns_distinct_alert_exit(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    current = tmp_path / "current.json"
    output = tmp_path / "report.json"
    write_scores(reference, repeated([0.05, 0.15], 20))
    write_scores(current, repeated([0.85, 0.95], 20))

    exit_code = main(
        [
            str(reference),
            str(current),
            "--output",
            str(output),
            "--minimum-reference-size",
            "20",
            "--minimum-current-size",
            "20",
        ]
    )

    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "alert"


def test_cli_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    current = tmp_path / "current.json"
    output = tmp_path / "report.json"
    reference.write_text('{"scores":[0.1],"scores":[0.2]}', encoding="utf-8")
    write_scores(current, [0.1])

    exit_code = main(
        [
            str(reference),
            str(current),
            "--output",
            str(output),
            "--minimum-reference-size",
            "1",
            "--minimum-current-size",
            "1",
        ]
    )

    assert exit_code == 3
    assert not output.exists()
