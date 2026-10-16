"""Paired TRAIN metrics distinguish learning both answers from one-label prediction."""

import importlib.util
from pathlib import Path

from test_gold_inventory import sample

from data.schema import Sample

path = Path(__file__).resolve().parents[1] / ".scratch/s4/evaluate-paired-train12.py"
spec = importlib.util.spec_from_file_location("paired_train12", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_metrics_detect_constant_answer_and_probability_alignment():
    rows = []
    pairs = []
    for i in range(6):
        pair = []
        for j, gold in enumerate(["yes", "no"]):
            identity = f"{i}:{j}"
            pair.append(identity)
            rows.append(
                {
                    "id": identity,
                    "target": gold,
                    "prediction": "yes",
                    "probabilities": {"yes": 0.8 if j == 0 else 0.6},
                }
            )
        pairs.append(pair)
    result = module.pair_metrics(rows, pairs)
    assert result["pair_both_correct"] == 0
    assert result["pair_yes_probability_aligned"] == 6
    assert result["pair_same_prediction"] == 6
    rows[1]["prediction"] = "no"
    assert module.pair_metrics(rows, pairs)["pair_both_correct"] == 1


def test_nonmedia_contract_drops_gold_but_keeps_state():
    left = Sample.model_validate(sample("one", pool="train", modality="video"))
    right = left.model_copy(deep=True)
    right.questions[0].target.value = "no"
    right.prefix.media[0].sha256 = "b" * 64
    assert module.nonmedia_key(left) == module.nonmedia_key(right)
    right.prefix.state += " different"
    assert module.nonmedia_key(left) != module.nonmedia_key(right)
