"""Frozen low-precision shadow assessment; never replaces a release or stage gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from data.parity_fixture import load_confirmation
from data.schema import Sample
from eval.program_truth import truth_metrics
from scripts.parity_impact import analyze


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_policy(policy: dict[str, Any]) -> None:
    if (
        policy.get("schema_version") != 1
        or policy.get("status") != "candidate_shadow_not_release_gate"
        or policy.get("release_or_stage_acceptance") is not False
        or policy.get("dtype") != "float16"
        or policy.get("backend") != "TRITON_ATTN"
        or policy.get("confirmation_must_be_complete_no_posthoc_filtering") is not True
        or policy.get("policy_changes_after_confirmation_require_new_independent_cohort")
        is not True
    ):
        raise ValueError("frozen non-release FP16/Triton shadow policy required")
    requirements = policy["requirements"]
    if (
        requirements.get("complete_identity_and_provenance") is not True
        or requirements.get("finite_complete_raw_probabilities") is not True
        or type(requirements.get("joint_top_answer_set_changes_max")) is not int
        or not 0 <= requirements["joint_top_answer_set_changes_max"] <= 100
        or policy["manual_review_triggers"].get("any_conditional_top_answer_set_change") is not True
    ):
        raise ValueError("complete evidence and explicit decision policy required")
    limits = [
        requirements[key]
        for key in (
            "joint_accuracy_degradation_max",
            "mean_joint_TV_max",
            "p95_joint_TV_max",
            "joint_brier_degradation_max",
            "known_target_ece15_degradation_per_type_max",
        )
    ] + [
        policy["manual_review_triggers"][key]
        for key in ("any_joint_TV_above", "any_normalized_expected_score_delta_above")
    ]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in limits):
        raise ValueError("finite probability-scale shadow budgets required")


def load_registered_confirmation(
    directory: Path, policy_path: Path, registration_path: Path
) -> tuple[list[Sample], dict[str, Any]]:
    policy = json.loads(policy_path.read_text())
    validate_policy(policy)
    registration = json.loads(registration_path.read_text())
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    expected = {
        "policy_sha256": sha(policy_path),
        "fixture_manifest_sha256": sha(manifest_path),
        "calibration_sha256": sha(directory / "calibration.jsonl"),
        "confirmation_sha256": sha(directory / "confirmation.jsonl"),
    }
    if (
        registration.get("schema_version") != 1
        or registration.get("purpose") != "preregistered-shadow-confirmation"
        or any(registration.get(k) != v for k, v in expected.items())
        or not registration.get("policy_git_commit")
        or not registration.get("registered_at_utc")
        or files["confirmation.jsonl"]["sha256"] != expected["confirmation_sha256"]
        or files["calibration.jsonl"]["sha256"] != expected["calibration_sha256"]
        or files["confirmation.jsonl"]["pool"] != "dev"
        or files["calibration.jsonl"]["pool"] != "calibration"
        or files["confirmation.jsonl"]["rows"] != 100
        or policy["cohort_for_confirmation"] != manifest["data_version"] + "/confirmation"
        or policy["cohort_for_selection"] != manifest["data_version"] + "/calibration"
    ):
        raise ValueError("preregistration or confirmation manifest mismatch")
    selection_families = set(files["calibration.jsonl"]["generator_families"])
    confirmation_families = set(files["confirmation.jsonl"]["generator_families"])
    if (
        not selection_families
        or not confirmation_families
        or selection_families & confirmation_families
    ):
        raise ValueError("distinct nonempty generator families required")
    samples = load_confirmation(directory / "confirmation.jsonl")
    if any(
        sample.questions[0].target is None or sample.questions[0].target.kind == "soft"
        for sample in samples
    ):
        raise ValueError("program hard or insufficient truth required")
    return samples, policy


def assess(
    samples: list[Sample],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    validate_policy(policy)
    if (
        reference.get("provenance", {}).get("purpose") != "independent-confirmation"
        or reference["provenance"].get("dtype") != policy["dtype"]
        or len(samples) != 100
        or any(s.meta.pool != "dev" for s in samples)
    ):
        raise ValueError("complete registered FP16 confirmation reports required")
    impact = analyze(reference, candidate)
    left, right = reference["rows"], candidate["rows"]
    hf_truth, vllm_truth = truth_metrics(samples, left), truth_metrics(samples, right)
    details = impact["details"]
    joint_changes = []
    score_changes = []
    for sample, a, b in zip(samples, left, right, strict=True):
        ja = [*(p * (1 - a["p_U"]) for p in a["conditional"]), a["p_U"]]
        jb = [*(p * (1 - b["p_U"]) for p in b["conditional"]), b["p_U"]]
        wa = [i for i, p in enumerate(ja) if p == max(ja)]
        wb = [i for i, p in enumerate(jb) if p == max(jb)]
        if wa != wb:
            joint_changes.append(
                {"id_sha256": a["id_sha256"], "reference_winners": wa, "candidate_winners": wb}
            )
        if sample.questions[0].type == "score":
            score_changes.append(
                {
                    "id_sha256": a["id_sha256"],
                    "normalized_delta": abs(
                        math.fsum(
                            i * (x - y)
                            for i, (x, y) in enumerate(
                                zip(a["conditional"], b["conditional"], strict=True)
                            )
                        )
                    )
                    / (len(a["conditional"]) - 1),
                }
            )
    requirements = policy["requirements"]
    ece_deltas = {}
    for kind in ("choice", "noul", "score"):
        h = hf_truth["conditional_known_target_by_type"][kind]
        v = vllm_truth["conditional_known_target_by_type"][kind]
        if h is None or v is None:
            raise ValueError("known-target coverage for every primitive required")
        ece_deltas[kind] = v["ece15"] - h["ece15"]
    observed = {
        "joint_top_answer_set_changes_max": len(joint_changes),
        "joint_accuracy_degradation_max": hf_truth["joint_unique_top1_accuracy"]
        - vllm_truth["joint_unique_top1_accuracy"],
        "mean_joint_TV_max": math.fsum(d["joint_TV"] for d in details) / 100,
        "p95_joint_TV_max": impact["metrics"]["joint_TV"]["p95_nearest_rank"],
        "joint_brier_degradation_max": vllm_truth["joint_brier"] - hf_truth["joint_brier"],
        "known_target_ece15_degradation_per_type_max": max(ece_deltas.values()),
    }
    failures = [k for k, value in observed.items() if value > requirements[k]]
    triggers = policy["manual_review_triggers"]
    tails = [d for d in details if d["joint_TV"] > triggers["any_joint_TV_above"]]
    score_tails = [
        d
        for d in score_changes
        if d["normalized_delta"] > triggers["any_normalized_expected_score_delta_above"]
    ]
    conditional_changes = [d for d in details if d["top_answer_set_changed"]]
    review = bool(tails or score_tails or conditional_changes)
    return {
        "outcome": "reject" if failures else "review_required" if review else "shadow_supported",
        "release_or_stage_acceptance": False,
        "sample_count": 100,
        "scope": "fixed program confirmation only; no production-quality claim or gate change",
        "failed_requirements": failures,
        "observed": observed,
        "budgets": {k: requirements[k] for k in observed},
        "known_target_ece15_deltas": ece_deltas,
        "joint_decision_changes": joint_changes,
        "joint_TV_review_cases": tails,
        "score_review_cases": score_tails,
        "conditional_answer_review_cases": conditional_changes,
        "reference_truth_metrics": hf_truth,
        "candidate_truth_metrics": vllm_truth,
        "impact": impact,
    }
