"""Optimistic DEV-only ranking/threshold diagnostic; never a deployed threshold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import time
from pathlib import Path
from typing import Any


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Positive-vs-negative concordance; each equal-score pair gets half credit."""
    validate(scores, labels)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels, strict=True))
    wins, negatives_before, index = 0.0, 0, 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        pos = sum(label for _, label in group)
        neg = len(group) - pos
        wins += pos * (negatives_before + neg / 2)
        negatives_before += neg
        index = end
    return wins / (positives * negatives)


def validate(scores: list[float], labels: list[bool]) -> None:
    if not scores or len(scores) != len(labels):
        raise ValueError("nonempty_equal_lengths")
    if any(type(y) is not bool for y in labels) or any(
        not math.isfinite(x) or not 0 <= x <= 1 for x in scores
    ):
        raise ValueError("finite_probability_boolean_labels")


def oracle(scores: list[float], labels: list[bool]) -> dict[str, Any]:
    """Fit yes iff score>=threshold; endpoints included, tied scores never split."""
    validate(scores, labels)
    correct = len(labels) - sum(labels)  # all-no endpoint
    best = correct
    ordered = sorted(zip(scores, labels, strict=True), reverse=True)
    index, cuts = 0, 1
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        positive = sum(label for _, label in group)
        correct += positive - (len(group) - positive)
        best = max(best, correct)
        cuts += 1
        index = end
    return {
        "best_correct": best,
        "accuracy_upper_bound": best / len(labels),
        "tie_preserving_cuts_tested": cuts,
        "threshold_exported_or_adopted": False,
    }


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    result = {}
    for name, q in [
        ("min", 0.0),
        ("p10", 0.1),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
        ("max", 1.0),
    ]:
        position = q * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [r["probabilities"]["yes"] for r in records]
    labels = [r["target"] == "yes" for r in records]
    current = sum(r["canonical_prediction"] == r["target"] for r in records)
    fitted = oracle(scores, labels)
    return {
        "count": len(records),
        "yes": sum(labels),
        "no": len(labels) - sum(labels),
        "p_yes_roc_auc": auc(scores, labels),
        "current_argmax_correct": current,
        "current_argmax_accuracy": current / len(records),
        "oracle_threshold_dev_fit": fitted,
        "optimistic_gap_correct": fitted["best_correct"] - current,
        "optimistic_gap_pp": 100 * (fitted["accuracy_upper_bound"] - current / len(records)),
        "residual_errors_after_oracle_fit": len(records) - fitted["best_correct"],
        "fn": {
            "count": sum(
                r["target"] == "yes" and r["canonical_prediction"] == "no" for r in records
            ),
            "p_yes_quantiles": quantiles(
                [
                    r["probabilities"]["yes"]
                    for r in records
                    if r["target"] == "yes" and r["canonical_prediction"] == "no"
                ]
            ),
        },
        "fp": {
            "count": sum(
                r["target"] == "no" and r["canonical_prediction"] == "yes" for r in records
            ),
            "p_yes_quantiles": quantiles(
                [
                    r["probabilities"]["yes"]
                    for r in records
                    if r["target"] == "no" and r["canonical_prediction"] == "yes"
                ]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    helper = globals().get("BOUND_HELPER")
    if helper is None:
        path = Path(__file__).with_name("diagnose-video-coverage.py")
        spec = importlib.util.spec_from_file_location("bound_video_helper", path)
        assert spec and spec.loader
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
    report, samples, binding = helper.load(args.root, "s4-coverage-video-full-001", 1)
    records = binding.pop("records")
    require = helper.require
    require(
        all(r["prediction"] == r["canonical_prediction"] for r in records.values()),
        "canonical_prediction_match",
    )
    output = {
        "scope": (
            "DEV-only optimistic fit; not calibration, deployed threshold or "
            "generalization evidence"
        ),
        "all": summarize(list(records.values())),
        "heuristic_groups": {
            kind: summarize(
                [r for ident, r in records.items() if helper.kind(samples[ident]) == kind]
            )
            for kind in ["existence", "motion", "temporal"]
        },
        "group_fit_boundary": (
            "Each subgroup oracle is separately fit to its own DEV labels; these are "
            "extra optimistic upper bounds, not a joint policy."
        ),
        "same_input_boundary": (
            "Same data + recorded config as prior diagnostic; prepared tensor/token "
            "input_sha absent. No GPU rerun."
        ),
        "coverage_binding": binding,
        "existing_report_accuracy": report["metrics"]["product_accuracy"],
        "quantile_method": "linear interpolation at q*(n-1)",
        "gpu_used": False,
        "threshold_changed": False,
        "cost": None,
        "duration_seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "script_sha256": globals().get("SOURCE_SHA") or helper.sha(Path(__file__)),
        "helper_sha256": globals().get("HELPER_SHA")
        or helper.sha(Path(__file__).with_name("diagnose-video-coverage.py")),
        "exit_code": 0,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
