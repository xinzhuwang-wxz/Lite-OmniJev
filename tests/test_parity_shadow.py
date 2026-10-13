"""Confirmation evidence must stay complete and cannot silently grant stage acceptance."""

import copy
import json
from pathlib import Path

import pytest

from data.parity_fixture import build_fixture, generate_confirmation_samples
from data.schema import HardTarget, answer_tokens
from eval.parity_shadow import assess, load_registered_confirmation, sha
from scripts.p0_parity import row


@pytest.fixture
def policy():
    return json.loads(Path("configs/candidates/parity-shadow-v1.json").read_text())


def reports():
    samples = generate_confirmation_samples()
    rows = []
    for i, sample in enumerate(samples):
        q = sample.questions[0]
        labels = [*answer_tokens(q), "U"]
        target = str(q.target.value) if isinstance(q.target, HardTarget) else "U"
        values = [-10.0 for _ in labels]
        values[labels.index(target)] = -0.01
        rows.append(
            row(
                {"id": sample.id, "input_ids": [i + 1], "token_ids": list(range(len(labels)))},
                values,
            )
        )
    reference = {
        "status": "reference-ready",
        "action": "hf-reference",
        "provenance": {"purpose": "independent-confirmation", "dtype": "float16", "model": "same"},
        "rows": rows,
    }
    candidate = copy.deepcopy(reference)
    candidate.update(status="passed", action="compare-vllm")
    return samples, reference, candidate


def change_row(candidate, index, values):
    old = candidate["rows"][index]
    new = row({"id": "ignored", "input_ids": [], "token_ids": old["token_ids"]}, values)
    for key in ("raw_logprobs", "raw_probabilities", "conditional", "p_U"):
        old[key] = new[key]


def registration(tmp_path, policy):
    build_fixture(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    m = json.loads((tmp_path / "manifest.json").read_text())
    record = {
        "schema_version": 1,
        "purpose": "preregistered-shadow-confirmation",
        "policy_sha256": sha(policy_path),
        "fixture_manifest_sha256": sha(tmp_path / "manifest.json"),
        "calibration_sha256": m["files"]["calibration.jsonl"]["sha256"],
        "confirmation_sha256": sha(tmp_path / "confirmation.jsonl"),
        "policy_git_commit": "frozen-before-model-results",
        "registered_at_utc": "2026-10-13",
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(record))
    return policy_path, path


def test_identical_reports_can_only_support_shadow(policy):
    result = assess(*reports(), policy)
    assert result["outcome"] == "shadow_supported"
    assert result["release_or_stage_acceptance"] is False
    assert result["joint_decision_changes"] == []


def test_joint_flip_rejects_even_when_summary_claims_accuracy_improved(policy):
    samples, reference, candidate = reports()
    old = candidate["rows"][0]["raw_logprobs"]
    values = [-10.0] * len(old)
    values[-1] = -0.01
    change_row(candidate, 0, values)
    candidate["truth_metrics"] = {"joint_unique_top1_accuracy": 1.0}
    result = assess(samples, reference, candidate, policy)
    assert result["outcome"] == "reject"
    assert "joint_top_answer_set_changes_max" in result["failed_requirements"]
    assert "joint_accuracy_degradation_max" in result["failed_requirements"]


def test_single_tail_is_reviewed_instead_of_averaged_away(policy):
    samples, reference, candidate = reports()
    values = list(candidate["rows"][0]["raw_logprobs"])
    wrong = next(i for i, p in enumerate(values[:-1]) if p == -10)
    values[wrong] = -3
    change_row(candidate, 0, values)
    result = assess(samples, reference, candidate, policy)
    assert result["failed_requirements"] == []
    assert result["outcome"] == "review_required"
    assert len(result["joint_TV_review_cases"]) == 1


def test_high_u_conditional_tie_change_is_preserved(policy):
    samples, reference, candidate = reports()
    values = list(candidate["rows"][1]["raw_logprobs"])
    values[0] = -9.9
    change_row(candidate, 1, values)
    result = assess(samples, reference, candidate, policy)
    assert result["outcome"] == "review_required"
    assert result["joint_decision_changes"] == []
    assert len(result["conditional_answer_review_cases"]) == 1


@pytest.mark.parametrize("bad", ["short", "nan", "identity", "provenance"])
def test_bad_evidence_never_supports_shadow(policy, bad):
    samples, reference, candidate = reports()
    if bad == "short":
        candidate["rows"].pop()
    elif bad == "nan":
        candidate["rows"][0]["raw_logprobs"][0] = float("nan")
    elif bad == "identity":
        candidate["rows"][0]["input_sha256"] = "other"
    else:
        candidate["provenance"]["model"] = "other"
    with pytest.raises(ValueError):
        assess(samples, reference, candidate, policy)


def test_confirmation_requires_registered_unchanged_policy_and_bytes(tmp_path, policy):
    policy_path, reg = registration(tmp_path, policy)
    samples, loaded = load_registered_confirmation(tmp_path, policy_path, reg)
    assert len(samples) == 100 and loaded == policy
    policy["requirements"]["mean_joint_TV_max"] = 1.0
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="mismatch"):
        load_registered_confirmation(tmp_path, policy_path, reg)


def test_changed_confirmation_truth_or_holdout_pool_is_rejected(tmp_path, policy):
    policy_path, reg = registration(tmp_path, policy)
    path = tmp_path / "confirmation.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["meta"]["pool"] = "holdout"
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="mismatch"):
        load_registered_confirmation(tmp_path, policy_path, reg)


def test_invalid_numeric_budgets_are_rejected(policy):
    policy["requirements"]["mean_joint_TV_max"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assess(*reports(), policy)


def test_joint_tie_is_a_decision_change(policy):
    samples, reference, candidate = reports()
    values = list(candidate["rows"][0]["raw_logprobs"])
    values[0] = values[1] = -0.01
    change_row(candidate, 0, values)
    result = assess(samples, reference, candidate, policy)
    assert result["outcome"] == "reject"
    assert result["joint_decision_changes"][0]["candidate_winners"] == [0, 1]


def test_score_tail_under_high_u_is_not_hidden_by_joint_tv(policy):
    samples, reference, candidate = reports()
    index = next(
        i
        for i, s in enumerate(samples)
        if s.questions[0].type == "score" and s.questions[0].target.kind == "insufficient"
    )
    values = list(candidate["rows"][index]["raw_logprobs"])
    values[0] = -9.95
    change_row(candidate, index, values)
    result = assess(samples, reference, candidate, policy)
    assert result["outcome"] == "review_required"
    assert result["joint_TV_review_cases"] == []
    assert len(result["score_review_cases"]) == 1


def test_holdout_metadata_rejected_even_if_all_file_hashes_are_rebound(tmp_path, policy):
    policy_path, reg = registration(tmp_path, policy)
    path = tmp_path / "confirmation.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["meta"]["pool"] = "holdout"
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["confirmation.jsonl"]["sha256"] = sha(path)
    manifest_path.write_text(json.dumps(manifest))
    record = json.loads(reg.read_text())
    record["confirmation_sha256"] = sha(path)
    record["fixture_manifest_sha256"] = sha(manifest_path)
    reg.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="pool dev"):
        load_registered_confirmation(tmp_path, policy_path, reg)
