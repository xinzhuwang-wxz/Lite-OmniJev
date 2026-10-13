"""Metrics from design §8, operating on answer distributions with U removed.

Evaluate each modality and hard/teacher target cohort separately. Insufficient
examples require a separate abstention-quality report, not an invented answer.
"""

import math
from collections.abc import Sequence


def evaluate(
    probabilities: Sequence[Sequence[float]], targets: Sequence[int], *, max_error: float = 0.05
) -> dict[str, float | int]:
    """Top-label ECE15, multiclass sum Brier and longest-prefix coverage.

    Confidence for coverage is (p_max - 1/K)/(1 - 1/K). A confidence threshold
    cannot split equal-confidence samples: only whole tie groups are considered.
    This offline diagnostic cannot select a production threshold on test labels.
    """
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("nonempty predictions and targets must have equal length")
    if isinstance(max_error, bool) or not math.isfinite(max_error) or not 0 <= max_error <= 1:
        raise ValueError("max_error must lie in [0, 1]")
    correct: list[int] = []
    maxima: list[float] = []
    confidence: list[float] = []
    brier: list[float] = []
    bins: list[list[int]] = [[] for _ in range(15)]
    for index, (row, target) in enumerate(zip(probabilities, targets, strict=True)):
        if not 2 <= len(row) <= 20:
            raise ValueError("each answer distribution requires 2–20 probabilities")
        if any(
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or not math.isfinite(p)
            or not 0 <= p <= 1
            for p in row
        ):
            raise ValueError("probabilities must be finite numbers in [0, 1]")
        if not math.isclose(math.fsum(row), 1.0, abs_tol=1e-9):
            raise ValueError("answer probabilities must sum to 1")
        if type(target) is not int or not 0 <= target < len(row):
            raise ValueError("target must be a valid integer answer index")
        maximum = max(row)
        prediction = next(i for i, p in enumerate(row) if p == maximum)
        correct.append(int(prediction == target))
        maxima.append(maximum)
        confidence.append((maximum - 1 / len(row)) / (1 - 1 / len(row)))
        brier.append(math.fsum((p - int(i == target)) ** 2 for i, p in enumerate(row)))
        bins[min(int(maximum * 15), 14)].append(index)
    count = len(correct)
    ece = math.fsum(
        abs(math.fsum(correct[i] - maxima[i] for i in bucket)) / count for bucket in bins if bucket
    )
    errors, covered = 0, 0
    order = sorted(range(count), key=lambda i: -confidence[i])
    for rank, index in enumerate(order, 1):
        errors += 1 - correct[index]
        group_end = rank == count or confidence[order[rank]] != confidence[index]
        # A longer complete group prefix may recover after an earlier violation.
        if group_end and errors / rank <= max_error:
            covered = rank
    return {
        "count": count,
        "accuracy": math.fsum(correct) / count,
        "ece15": ece,
        "brier": math.fsum(brier) / count,
        "coverage_at_error": covered / count,
        "max_error": max_error,
    }
