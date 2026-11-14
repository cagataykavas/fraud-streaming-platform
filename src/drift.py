"""Deterministic population-stability audit for fraud risk scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_BIN_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MAX_INPUT_BYTES = 5 * 1024 * 1024
MAX_SCORES = 1_000_000


@dataclass(frozen=True)
class DriftPolicy:
    """Admission and decision policy for a score-drift comparison.

    PSI thresholds are operating-policy defaults, not universal statistical
    significance levels. Teams should calibrate them to their scorecard and
    review capacity.
    """

    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES
    minimum_reference_size: int = 100
    minimum_current_size: int = 50
    smoothing_count: float = 0.5
    warning_threshold: float = 0.10
    alert_threshold: float = 0.25

    def __post_init__(self) -> None:
        edges = self.bin_edges
        if not 2 <= len(edges) <= 101:
            raise ValueError("bin_edges must contain between 2 and 101 values")
        if any(
            isinstance(edge, bool) or not isinstance(edge, (int, float)) or not math.isfinite(edge)
            for edge in edges
        ):
            raise ValueError("bin_edges must contain only finite numbers")
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("bin_edges must start at 0.0 and end at 1.0")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise ValueError("bin_edges must be strictly increasing")
        if self.minimum_reference_size < 1 or self.minimum_current_size < 1:
            raise ValueError("minimum sample sizes must be positive")
        if not math.isfinite(self.smoothing_count) or self.smoothing_count <= 0:
            raise ValueError("smoothing_count must be finite and positive")
        if not (
            math.isfinite(self.warning_threshold)
            and math.isfinite(self.alert_threshold)
            and 0 <= self.warning_threshold < self.alert_threshold
        ):
            raise ValueError("thresholds must be finite and satisfy 0 <= warning < alert")


@dataclass(frozen=True)
class DriftBin:
    lower: float
    upper: float
    reference_count: int
    current_count: int
    reference_share: float
    current_share: float
    psi_contribution: float


@dataclass(frozen=True)
class ScoreDriftReport:
    status: str
    population_stability_index: float
    reference_size: int
    current_size: int
    warning_threshold: float
    alert_threshold: float
    smoothing_count: float
    bins: tuple[DriftBin, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["bins"] = [asdict(item) for item in self.bins]
        return payload


def _validate_scores(name: str, scores: Sequence[float], minimum_size: int) -> tuple[float, ...]:
    if isinstance(scores, (str, bytes)):
        raise ValueError(f"{name} scores must be a sequence of numbers")
    if len(scores) < minimum_size:
        raise ValueError(f"{name} requires at least {minimum_size} scores")
    if len(scores) > MAX_SCORES:
        raise ValueError(f"{name} exceeds the {MAX_SCORES} score limit")

    validated: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{name}[{index}] must be a number")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{index}] must be finite and within [0, 1]")
        validated.append(value)
    return tuple(validated)


def _counts(scores: Sequence[float], edges: tuple[float, ...]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for score in scores:
        index = min(bisect_right(edges, score) - 1, len(counts) - 1)
        counts[index] += 1
    return counts


def audit_score_drift(
    reference_scores: Sequence[float],
    current_scores: Sequence[float],
    policy: DriftPolicy | None = None,
) -> ScoreDriftReport:
    """Compare two admitted score populations with fixed-bin PSI.

    A Jeffreys-style pseudo-count keeps empty bins finite without hiding their
    raw counts. Fixed, policy-owned edges make repeated audits comparable and
    avoid deriving bins from the population being evaluated.
    """

    policy = policy or DriftPolicy()
    reference = _validate_scores("reference", reference_scores, policy.minimum_reference_size)
    current = _validate_scores("current", current_scores, policy.minimum_current_size)
    reference_counts = _counts(reference, policy.bin_edges)
    current_counts = _counts(current, policy.bin_edges)
    number_of_bins = len(reference_counts)
    reference_denominator = len(reference) + policy.smoothing_count * number_of_bins
    current_denominator = len(current) + policy.smoothing_count * number_of_bins

    bins: list[DriftBin] = []
    psi = 0.0
    for index, (reference_count, current_count) in enumerate(
        zip(reference_counts, current_counts, strict=True)
    ):
        reference_share = (reference_count + policy.smoothing_count) / reference_denominator
        current_share = (current_count + policy.smoothing_count) / current_denominator
        contribution = (current_share - reference_share) * math.log(current_share / reference_share)
        psi += contribution
        bins.append(
            DriftBin(
                lower=policy.bin_edges[index],
                upper=policy.bin_edges[index + 1],
                reference_count=reference_count,
                current_count=current_count,
                reference_share=reference_share,
                current_share=current_share,
                psi_contribution=contribution,
            )
        )

    if psi >= policy.alert_threshold:
        status = "alert"
    elif psi >= policy.warning_threshold:
        status = "warning"
    else:
        status = "stable"

    return ScoreDriftReport(
        status=status,
        population_stability_index=psi,
        reference_size=len(reference),
        current_size=len(current),
        warning_threshold=policy.warning_threshold,
        alert_threshold=policy.alert_threshold,
        smoothing_count=policy.smoothing_count,
        bins=tuple(bins),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_scores(path: Path) -> list[float]:
    if path.is_symlink():
        raise ValueError(f"refusing symlink input: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    scores = payload.get("scores") if isinstance(payload, dict) else payload
    if not isinstance(scores, list):
        raise ValueError(f"{path} must contain a JSON array or an object with a scores array")
    return scores


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit fraud-score population drift")
    parser.add_argument("reference", type=Path, help="JSON reference score population")
    parser.add_argument("current", type=Path, help="JSON current score population")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON report path")
    parser.add_argument("--minimum-reference-size", type=int, default=100)
    parser.add_argument("--minimum-current-size", type=int, default=50)
    parser.add_argument("--warning-threshold", type=float, default=0.10)
    parser.add_argument("--alert-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)

    try:
        policy = DriftPolicy(
            minimum_reference_size=args.minimum_reference_size,
            minimum_current_size=args.minimum_current_size,
            warning_threshold=args.warning_threshold,
            alert_threshold=args.alert_threshold,
        )
        report = audit_score_drift(_load_scores(args.reference), _load_scores(args.current), policy)
        _write_atomic(args.output, report.to_dict())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"score drift audit failed: {exc}", file=sys.stderr)
        return 3
    return 2 if report.status == "alert" else 0


if __name__ == "__main__":
    raise SystemExit(main())
