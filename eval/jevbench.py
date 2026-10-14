"""Pure JevBench public scoring semantics, reproduced from the locked MIT source."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from data.jevbench import JevBenchProtocolError, JevBenchTask

SUM_TOL = 1e-3
RENORM_TOL = 2e-2
ECE_BINS = 10


class InvalidDistribution(ValueError):
    """Probability map cannot be scored under the frozen public protocol."""


@dataclass(frozen=True)
class ScoredTask:
    id: str
    family: str
    valid: bool
    strict_valid: bool
    renormalized: bool
    correct: bool
    predicted: str | None
    probabilities: dict[str, float] | None
    error: str | None = None
    ordinal_ev: float | None = None


def validate_probs(
    probabilities: Mapping[str, float], labels: Sequence[str], *, sum_tol: float = SUM_TOL
) -> dict[str, float]:
    if not isinstance(probabilities, Mapping):
        raise InvalidDistribution("probs is not a dict")
    want = set(labels)
    got = set(probabilities)
    if got != want:
        raise InvalidDistribution(
            f"label keys mismatch: missing={sorted(want - got)} extra={sorted(got - want)}"
        )
    out: dict[str, float] = {}
    total = 0.0
    for key, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidDistribution(f"prob[{key!r}] is not a number: {value!r}")
        prob = float(value)
        if not math.isfinite(prob):
            raise InvalidDistribution(f"prob[{key!r}] is not finite")
        if prob < 0.0 or prob > 1.0:
            raise InvalidDistribution(f"prob[{key!r}] out of [0,1]: {prob}")
        out[str(key)] = prob
        total += prob
    if abs(total - 1.0) > sum_tol:
        raise InvalidDistribution(f"probs sum to {total}, tolerance {sum_tol}")
    return out


def argmax_label(probabilities: Mapping[str, float]) -> str:
    best_label = ""
    best_probability = -1.0
    for label in sorted(probabilities):
        if probabilities[label] > best_probability:
            best_label = label
            best_probability = probabilities[label]
    return best_label


def expected_value(probabilities: Mapping[str, float]) -> float:
    return sum(float(label) * probability for label, probability in probabilities.items())


def brier_score(
    probabilities: Mapping[str, float], expected_label: str, labels: Sequence[str]
) -> float:
    total = 0.0
    for label in labels:
        target = 1.0 if label == expected_label else 0.0
        total += (float(probabilities.get(label, 0.0)) - target) ** 2
    return total


def ece_top_label(pairs: Sequence[tuple[float, bool]], *, n_bins: int = ECE_BINS) -> dict[str, Any]:
    bins: list[dict[str, float | int]] = [
        {"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0, "conf_sum": 0.0, "correct": 0}
        for i in range(n_bins)
    ]
    for confidence, correct in pairs:
        clipped = min(max(float(confidence), 0.0), 1.0)
        index = min(int(clipped * n_bins), n_bins - 1)
        bucket = bins[index]
        bucket["n"] = int(bucket["n"]) + 1
        bucket["conf_sum"] = float(bucket["conf_sum"]) + clipped
        bucket["correct"] = int(bucket["correct"]) + (1 if correct else 0)
    total = sum(int(bucket["n"]) for bucket in bins)
    ece = 0.0
    rendered = []
    for bucket in bins:
        count = int(bucket["n"])
        if count:
            accuracy = int(bucket["correct"]) / count
            mean_confidence = float(bucket["conf_sum"]) / count
            ece += (count / total) * abs(accuracy - mean_confidence)
        else:
            accuracy = None
            mean_confidence = None
        rendered.append(
            {
                "lo": bucket["lo"],
                "hi": bucket["hi"],
                "n": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {"ece": ece, "n": total, "bins": rendered}


def score_task(task: JevBenchTask, probabilities: Mapping[str, float]) -> ScoredTask:
    try:
        clean = validate_probs(probabilities, task.labels)
        strict_valid = True
        renormalized = False
    except InvalidDistribution as strict_error:
        try:
            clean = validate_probs(probabilities, task.labels, sum_tol=RENORM_TOL)
        except InvalidDistribution:
            return ScoredTask(
                id=task.id,
                family=task.family,
                valid=False,
                strict_valid=False,
                renormalized=False,
                correct=False,
                predicted=None,
                probabilities=None,
                error=str(strict_error),
            )
        total = math.fsum(clean.values())
        if total <= 0.0:
            return ScoredTask(
                id=task.id,
                family=task.family,
                valid=False,
                strict_valid=False,
                renormalized=False,
                correct=False,
                predicted=None,
                probabilities=None,
                error="probabilities sum to zero",
            )
        clean = {label: probability / total for label, probability in clean.items()}
        strict_valid = False
        renormalized = True

    predicted = argmax_label(clean)
    correct = predicted == str(task.expected)
    ordinal_ev = expected_value(clean) if task.primitive == "score" else None
    return ScoredTask(
        id=task.id,
        family=task.family,
        valid=True,
        strict_valid=strict_valid,
        renormalized=renormalized,
        correct=correct,
        predicted=predicted,
        probabilities=clean,
        ordinal_ev=ordinal_ev,
    )


def macro_accuracy(per_family: Mapping[str, Mapping[str, Any]]) -> float:
    values = [
        float(family["accuracy"]) for family in per_family.values() if int(family["n_scorable"]) > 0
    ]
    return math.fsum(values) / len(values) if values else 0.0


def ordinal_mae(pairs: Sequence[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    return math.fsum(abs(expected - predicted) for expected, predicted in pairs) / len(pairs)


def paraphrase_consistency(
    results_by_id: Mapping[str, ScoredTask], tasks_by_id: Mapping[str, JevBenchTask]
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, task in tasks_by_id.items():
        if task.group:
            groups[task.group].append(task_id)
    pairs = 0
    both_answered = 0
    both_correct = 0
    unscorable = 0
    for ids in groups.values():
        if len(ids) < 2:
            continue
        pairs += 1
        outcomes = [results_by_id.get(task_id) for task_id in ids]
        if any(outcome is None or not outcome.valid for outcome in outcomes):
            continue
        both_answered += 1
        if any(outcome.correct is None for outcome in outcomes if outcome is not None):
            unscorable += 1
            continue
        if all(outcome is not None and outcome.correct for outcome in outcomes):
            both_correct += 1
    return {
        "pairs": pairs,
        "both_answered_valid": both_answered,
        "both_correct": both_correct,
        "unscorable_expected_none": unscorable,
        "consistency": (both_correct / both_answered) if both_answered else None,
    }


def _prediction_map(raw: Any) -> dict[str, Mapping[str, float]]:
    if isinstance(raw, Mapping) and "predictions" in raw:
        raw = raw["predictions"]
    if isinstance(raw, Mapping):
        out = {}
        for task_id, value in raw.items():
            if not isinstance(task_id, str) or not isinstance(value, Mapping):
                raise JevBenchProtocolError("prediction mapping must be id -> probability object")
            probs = value.get("probabilities", value)
            if not isinstance(probs, Mapping):
                raise JevBenchProtocolError("prediction probabilities must be an object")
            out[task_id] = probs
        return out
    if isinstance(raw, list):
        out = {}
        for row in raw:
            if not isinstance(row, Mapping):
                raise JevBenchProtocolError("prediction rows must be objects")
            task_id = row.get("id")
            probs = row.get("probabilities")
            if not isinstance(task_id, str) or not isinstance(probs, Mapping):
                raise JevBenchProtocolError("prediction rows require id and probabilities")
            if task_id in out:
                raise JevBenchProtocolError(f"duplicate prediction id: {task_id}")
            out[task_id] = probs
        return out
    raise JevBenchProtocolError("predictions must be a list or mapping")


def score_predictions(tasks: Sequence[JevBenchTask], raw_predictions: Any) -> dict[str, Any]:
    predictions = _prediction_map(raw_predictions)
    tasks_by_id = {task.id: task for task in tasks}
    missing = sorted(set(tasks_by_id) - set(predictions))
    extra = sorted(set(predictions) - set(tasks_by_id))
    if missing or extra:
        raise JevBenchProtocolError(
            f"prediction ids mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    results = {task.id: score_task(task, predictions[task.id]) for task in tasks}
    families: dict[str, dict[str, int | float]] = {}
    for family in sorted({task.family for task in tasks}):
        family_results = [result for result in results.values() if result.family == family]
        scorable = len(family_results)
        correct = sum(1 for result in family_results if result.correct)
        families[family] = {
            "n_scorable": scorable,
            "correct": correct,
            "accuracy": correct / scorable if scorable else 0.0,
        }
    ece_pairs: list[tuple[float, bool]] = []
    briers: list[float] = []
    ordinal_pairs: list[tuple[int, float]] = []
    for task in tasks:
        result = results[task.id]
        if not result.valid or result.probabilities is None:
            continue
        confidence = max(result.probabilities.values())
        ece_pairs.append((confidence, result.correct))
        briers.append(brier_score(result.probabilities, str(task.expected), task.labels))
        if task.primitive == "score" and result.ordinal_ev is not None:
            ordinal_pairs.append((int(task.expected), result.ordinal_ev))
    correct = sum(1 for result in results.values() if result.correct)
    invalid = sum(1 for result in results.values() if not result.valid)
    return {
        "schema_version": 1,
        "status": "passed",
        "public_decisions": len(tasks),
        "accuracy": correct / len(tasks) if tasks else 0.0,
        "macro_accuracy": macro_accuracy(families),
        "per_family": families,
        "invalid_distributions": invalid,
        "strict_invalid_renormalized": sum(1 for result in results.values() if result.renormalized),
        "brier": math.fsum(briers) / len(briers) if briers else None,
        "ece": ece_top_label(ece_pairs),
        "ordinal_mae": ordinal_mae(ordinal_pairs),
        "paraphrase_consistency": paraphrase_consistency(results, tasks_by_id),
        "results": {
            task_id: {
                "valid": result.valid,
                "strict_valid": result.strict_valid,
                "renormalized": result.renormalized,
                "correct": result.correct,
                "predicted": result.predicted,
                "error": result.error,
                "ordinal_ev": result.ordinal_ev,
            }
            for task_id, result in results.items()
        },
        "official_semantics": {
            "headline": "argmax source label; ties lexicographic; score uses argmax",
            "macro": "unweighted source family mean",
            "sum_tol": SUM_TOL,
            "renorm_tol": RENORM_TOL,
            "brier": "one-hot expected over source labels",
            "ece": "top-label confidence, 10 equal-width bins",
        },
        "p_insufficient_in_official_scorer": False,
        "teacher_matched_protocol_verified": False,
        "gpu_used": False,
    }
