"""Fail-closed numerical gate. This does not attest dataset/model provenance.

CLI: python -m eval.gate metrics.json thresholds.json
Thresholds: {"accuracy": {"min": 0.98}, "ece15": {"max": 0.06}}.
Exit 0 only for nonempty valid passing checks; 1 for failed checks, 2 for bad files.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def evaluate_gate(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not thresholds:
        return {
            "passed": False,
            "checks": [],
            "error": "nonempty threshold mapping and metric mapping required",
        }
    for name, bounds in thresholds.items():
        reason = None
        value = metrics.get(name)
        if not isinstance(name, str) or not name:
            reason = "metric names must be nonempty strings"
        elif (
            not isinstance(bounds, dict)
            or not bounds
            or set(bounds) - {"min", "max", "min_exclusive", "max_exclusive"}
        ):
            reason = "bounds require min/max or exclusive variants only"
        elif any(not _finite(bound) for bound in bounds.values()):
            reason = "bounds must be finite numbers"
        elif ("min" in bounds and "min_exclusive" in bounds) or (
            "max" in bounds and "max_exclusive" in bounds
        ):
            reason = "inclusive and exclusive bounds cannot share a side"
        elif any(
            bounds[lo] > bounds[hi] or (bounds[lo] == bounds[hi] and "exclusive" in lo + hi)
            for lo in ("min", "min_exclusive")
            for hi in ("max", "max_exclusive")
            if lo in bounds and hi in bounds
        ):
            reason = "lower bound exceeds upper bound or interval is empty"
        elif not _finite(value):
            reason = "metric is missing or not finite"
        elif "min" in bounds and value < bounds["min"]:
            reason = "metric below minimum"
        elif "max" in bounds and value > bounds["max"]:
            reason = "metric above maximum"
        elif "min_exclusive" in bounds and value <= bounds["min_exclusive"]:
            reason = "metric at or below exclusive minimum"
        elif "max_exclusive" in bounds and value >= bounds["max_exclusive"]:
            reason = "metric at or above exclusive maximum"
        checks.append(
            {
                "metric": str(name),
                "value": value if _finite(value) else None,
                "passed": reason is None,
                "reason": reason,
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("thresholds", type=Path)
    args = parser.parse_args()
    try:
        metrics, thresholds = [
            json.loads(
                p.read_text(), parse_constant=_reject_constant, object_pairs_hook=_unique_object
            )
            for p in (args.metrics, args.thresholds)
        ]
        result = evaluate_gate(metrics, thresholds)
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
