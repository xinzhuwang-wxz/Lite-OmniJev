"""Program-fixture truth metrics, separate from numerical engine parity."""

from __future__ import annotations

import math
from typing import Any

from data.schema import HardTarget, InsufficientTarget, Sample, answer_tokens
from eval.metrics import evaluate
from scripts.p0_parity import digest


def truth_metrics(samples: list[Sample], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != len(rows):
        raise ValueError("complete predictions required")
    known: dict[str, list[tuple[list[float], int]]] = {k: [] for k in ("choice", "noul", "score")}
    correct, brier = [], []
    u_squared: dict[str, list[float]] = {"sufficient": [], "insufficient": []}
    score_errors = []
    for sample, row in zip(samples, rows, strict=True):
        if row["id_sha256"] != digest(sample.id):
            raise ValueError("prediction identity mismatch")
        question = sample.questions[0]
        ca, u = row["conditional"], row["p_U"]
        if len(ca) != len(answer_tokens(question)):
            raise ValueError("answer count mismatch")
        target = question.target
        insufficient = isinstance(target, InsufficientTarget)
        if isinstance(target, HardTarget):
            index = answer_tokens(question).index(str(target.value))
            known[question.type].append((ca, index))
            if question.type == "score":
                score_errors.append(abs(math.fsum(i * p for i, p in enumerate(ca)) - index))
        elif insufficient:
            index = len(ca)
        else:
            raise ValueError("hard truth required")
        joint = [*(p * (1 - u) for p in ca), u]
        winners = [i for i, p in enumerate(joint) if p == max(joint)]
        correct.append(int(winners == [index]))
        brier.append(math.fsum((p - int(i == index)) ** 2 for i, p in enumerate(joint)))
        u_squared["insufficient" if insufficient else "sufficient"].append((u - insufficient) ** 2)
    return {
        "scope": "program fixture only; no production accuracy or calibrated abstention claim",
        "joint_unique_top1_accuracy": math.fsum(correct) / len(correct),
        "joint_brier": math.fsum(brier) / len(brier),
        "conditional_known_target_by_type": {
            k: evaluate([p for p, _ in group], [t for _, t in group]) if group else None
            for k, group in known.items()
        },
        "p_U_mse_by_truth": {
            k: {"count": len(values), "mse": math.fsum(values) / len(values) if values else None}
            for k, values in u_squared.items()
        },
        "conditional_score_mae": math.fsum(score_errors) / len(score_errors)
        if score_errors
        else None,
    }
