"""Paired analysis rejects changed inputs and inconsistent probability sidecars."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from scripts.s0_development_baselines import sha256_json

path = Path(__file__).resolve().parents[1] / ".scratch/s4/compare-native-event-condition-dev50.py"
spec = importlib.util.spec_from_file_location("native_compare", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row():
    summary = {"tensors": {"input_ids": {"sha256": "a" * 64}}}
    result = {
        k: "same"
        for k in [
            "id",
            "family_id",
            "target_index",
            "answer_token_ids",
            "input_sha256",
            "prompt_sha256",
            "payload_sha256",
            "frame_indices",
            "pass_index",
            "sampling_policy",
        ]
    }
    result.update(input_summary=summary, input_summary_sha256=sha256_json(summary))
    return result


def test_strict_input_pair_rejects_tensor_or_digest_change():
    left = row()
    module.verify_pair(left, copy.deepcopy(left))
    right = copy.deepcopy(left)
    right["input_summary"]["tensors"]["input_ids"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="input"):
        module.verify_pair(left, right)
    left["input_summary_sha256"] = "bad"
    with pytest.raises(ValueError, match="digest"):
        module.verify_pair(left, copy.deepcopy(left))


def test_brier_and_yes_false_positive_use_original_gold():
    value = {
        "conditional_without_u": {"yes": 0.75, "no": 0.25},
        "raw_logprobs": {"yes": math.log(0.75), "no": math.log(0.25)},
        "prediction": "yes",
    }
    metrics = module.row_metrics(value, "no")
    assert metrics["yes_fp"] and not metrics["correct"]
    assert metrics["brier"] == pytest.approx(1.125)
    value["conditional_without_u"]["yes"] = 0.7
    with pytest.raises(ValueError):
        module.row_metrics(value, "no")


def test_exact_probability_tie_preserves_saved_legal_readout():
    value = {
        "conditional_without_u": {"no": 0.5, "yes": 0.5},
        "raw_logprobs": {"yes": -1.0, "no": -1.0},
        "prediction": "yes",
    }
    assert module.row_metrics(value, "yes")["correct"]
