"""Explain paired P0 probability reports without changing or replacing any gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.p0_parity import row


def analyze(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if (
        reference.get("status") != "reference-ready"
        or reference.get("action") != "hf-reference"
        or candidate.get("action") != "compare-vllm"
        or candidate.get("status") not in {"passed", "failed"}
        or not reference.get("provenance")
        or reference["provenance"] != candidate.get("provenance")
    ):
        raise ValueError("paired completed reports with matching provenance required")
    left, right = reference.get("rows", []), candidate.get("rows", [])
    if len(left) != 100 or len(right) != 100:
        raise ValueError("all 100 P0 rows required")
    seen: set[str] = set()
    details = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if any(a[key] != b[key] for key in ("id_sha256", "input_sha256", "token_ids")):
            raise ValueError("row identity mismatch")
        if a["id_sha256"] in seen:
            raise ValueError("duplicate sample identity")
        seen.add(a["id_sha256"])
        for item in (a, b):
            ids = item["token_ids"]
            if (
                len(ids) < 3
                or any(type(token) is not int for token in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("invalid legal token IDs")
            rebuilt = row(
                {"id": "probability-check", "input_ids": [], "token_ids": ids},
                item["raw_logprobs"],
            )
            expected = [*rebuilt["raw_probabilities"], *rebuilt["conditional"], rebuilt["p_U"]]
            actual = [*item["raw_probabilities"], *item["conditional"], item["p_U"]]
            # Linux/macOS libm can differ by an ulp; this is not an engine acceptance budget.
            if any(
                type(y) not in (int, float) or not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15)
                for x, y in zip(expected, actual, strict=True)
            ):
                raise ValueError("probabilities do not match raw logprobs")
        ca, cb = a["conditional"], b["conditional"]
        ua, ub = a["p_U"], b["p_U"]
        joint_a = [*(p * (1 - ua) for p in ca), ua]
        joint_b = [*(p * (1 - ub) for p in cb), ub]
        # Include every co-maximal answer; a changed tie is not silently a stable argmax.
        top_a = [i for i, p in enumerate(ca) if p == max(ca)]
        top_b = [i for i, p in enumerate(cb) if p == max(cb)]
        details.append(
            {
                "index": index,
                "id_sha256": a["id_sha256"],
                "conditional_max": max(abs(x - y) for x, y in zip(ca, cb, strict=True)),
                "raw_max": max(
                    abs(x - y)
                    for x, y in zip(a["raw_probabilities"], b["raw_probabilities"], strict=True)
                ),
                "p_U_difference": abs(ua - ub),
                "joint_TV": sum(abs(x - y) for x, y in zip(joint_a, joint_b, strict=True)) / 2,
                "top_answer_set_changed": top_a != top_b,
                "reference_p_U": ua,
                "candidate_p_U": ub,
            }
        )
    metrics = {}
    for key in ("conditional_max", "raw_max", "p_U_difference", "joint_TV"):
        values = sorted(float(item[key]) for item in details)
        metrics[key] = {
            "median": (values[49] + values[50]) / 2,
            "p95_nearest_rank": values[math.ceil(len(values) * 0.95) - 1],
            "max": values[-1],
        }
    slices = {}
    # Fixed descriptive bins, not a fitted abstention or business acceptance policy.
    for name, lo, hi in (("low_U", 0, 0.5), ("middle_U", 0.5, 0.99), ("high_U", 0.99, 1.01)):
        selected = [d for d in details if lo <= d["reference_p_U"] < hi]
        slices[name] = {
            "count": len(selected),
            "conditional_max": max((d["conditional_max"] for d in selected), default=None),
            "top_answer_set_changes": sum(d["top_answer_set_changed"] for d in selected),
        }
    return {
        "status": "analysis-only",
        "gate_changed": False,
        "quality_verified": False,
        "scope": "paired program fixtures; no labels or calibrated decision policy evaluated",
        "slice_semantics": "fixed reference p(U) bins, descriptive only; all rows retained",
        "joint_formula": "[(1-p_U)*conditional_i for each answer] + [p_U]",
        "sample_count": len(details),
        "top_answer_set_changes": sum(d["top_answer_set_changed"] for d in details),
        "metrics": metrics,
        "slices": slices,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [args.reference.read_bytes(), args.candidate.read_bytes()]
    result = analyze(*(json.loads(source) for source in sources))
    result["source_sha256"] = dict(
        zip(
            ("reference", "candidate"),
            (hashlib.sha256(s).hexdigest() for s in sources),
            strict=True,
        )
    )
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "sample_count", "top_answer_set_changes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
