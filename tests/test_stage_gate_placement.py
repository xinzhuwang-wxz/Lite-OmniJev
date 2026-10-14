import json
from pathlib import Path

import pytest

from eval.gate import evaluate_gate


def stage(name):
    config = Path(__file__).resolve().parents[1] / "configs/gates.json"
    return json.loads(config.read_text())["stages"][name]


def test_s0_can_close_technical_baselines_without_new_human_labels():
    metrics = {
        "teacher_mmau_delta_abs": 0.011,
        "repeat_identical": 1,
        "hf_vllm_max_probability_delta": 0.000032294,
        "baseline_table_complete": 1,
        "teacher_mvbench_baseline_verified": 1,
        "prefix_cache_baseline_verified": 1,
    }
    assert evaluate_gate(metrics, stage("S0"))["passed"]
    assert not evaluate_gate({**metrics, "repeat_identical": 0}, stage("S0"))["passed"]


@pytest.mark.parametrize("provenance,passed", [(None, False), (0, False), (1, True)])
def test_s4_requires_existing_gold_provenance(provenance, passed):
    metrics = {
        "all_modalities_min_teacher_ratio": 0.97,
        "max_holdout_dev_gap": 0.03,
        "decision_index_vs_peer_delta": 0,
        "jevbench_vs_peer_delta": 0,
        "holdout_gate_pass": 1,
    }
    if provenance is not None:
        metrics["holdout_gold_provenance_verified"] = provenance
    assert evaluate_gate(metrics, stage("S4"))["passed"] is passed


def test_gold_provenance_alone_cannot_replace_sealed_gate_result():
    metrics = {
        "all_modalities_min_teacher_ratio": 0.97,
        "max_holdout_dev_gap": 0.03,
        "decision_index_vs_peer_delta": 0,
        "jevbench_vs_peer_delta": 0,
        "holdout_gold_provenance_verified": 1,
    }
    assert not evaluate_gate(metrics, stage("S4"))["passed"]
