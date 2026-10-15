"""CPU public-interface paired metric and evidence-boundary tests."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from data.schema import Sample
from train.readout import read_probabilities


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location(
        "temporal_compare", Path(".scratch/s4/compare-temporal-curriculum.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(index, instruction, gold):
    return Sample.model_validate(
        {
            "id": str(index),
            "prefix": {"state": "state", "media": []},
            "questions": [
                {
                    "qid": "q",
                    "type": "noul",
                    "instruction": instruction,
                    "target": {"kind": "hard", "value": gold, "source": "gold"},
                }
            ],
            "meta": {
                "modality": ["text"],
                "source_dataset": "fixture",
                "source_revision": "r1",
                "source_sha256": "a" * 64,
                "license": "MIT",
                "usage": "research",
                "pool": "dev",
                "family_id": str(index),
                "data_version": "fixture",
            },
        }
    )


def record(sample, yes, no):
    logits = {"yes": yes, "no": no, "U": -20.0}
    result = read_probabilities(logits, ["yes", "no"], temperature=1)
    return {
        "id": sample.id,
        "pool": "dev",
        "family_hash": sample.meta.family_id,
        "pass": 0,
        "variant": "original",
        "target": sample.questions[0].target.value,
        "prediction": result.prediction,
        "probabilities": result.probabilities,
        "raw_logits": logits,
    }


def test_hand_calculated_corrections_regressions_and_probability_loss(helper):
    samples = [sample(0, "Was it before event?", "yes"), sample(1, "Is it moving now", "no")]
    before = [record(samples[0], 0.0, 1.0), record(samples[1], 0.0, 2.0)]
    after = [record(samples[0], 2.0, 0.0), record(samples[1], 1.0, 0.0)]
    result = helper.compare(samples, before, after)
    all_rows = result["summaries"]["all"]
    assert all_rows["baseline_correct"] == all_rows["candidate_correct"] == 1
    assert all_rows["corrected"] == all_rows["regressed"] == 1
    assert all_rows["nll"]["baseline"] == pytest.approx(
        (math.log1p(math.exp(1)) + math.log1p(math.exp(-2))) / 2
    )
    assert all_rows["nll"]["delta"] == pytest.approx(0)
    assert result["summaries"]["temporal"]["candidate_correct"] == 1
    assert result["summaries"]["motion"]["candidate_correct"] == 0
    assert len(result["changed_predictions"]) == 2
    assert result["paired_input_hashes_verified"] is None
    assert "family_hash is not" in result["input_hash_limitation"]


def test_real_input_hashes_are_required_on_every_pair_if_present(helper):
    samples = [sample(0, "Before event?", "yes")]
    rows = [record(samples[0], 1.0, 0.0)]
    for key in ("input_sha256", "input_summary_sha256"):
        rows[0][key] = "a" * 64
    assert (
        helper.compare(samples, rows, copy.deepcopy(rows))["paired_input_hashes_verified"] is True
    )
    other = copy.deepcopy(rows)
    other[0]["input_summary_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="input hashes"):
        helper.compare(samples, rows, other)
    other = copy.deepcopy(rows)
    del other[0]["input_sha256"]
    with pytest.raises(ValueError, match="input hashes"):
        helper.compare(samples, rows, other)


@pytest.mark.parametrize("change", ["id", "target", "probability", "family", "pool", "variant"])
def test_mismatched_evidence_rejected(helper, change):
    samples = [sample(0, "Before event?", "yes")]
    rows = [record(samples[0], 1.0, 0.0)]
    other = copy.deepcopy(rows)
    if change == "id":
        other[0]["id"] = "different"
    elif change == "target":
        other[0]["target"] = "no"
    elif change == "probability":
        other[0]["probabilities"]["yes"] = 0.1
    elif change == "family":
        other[0]["family_hash"] = "different"
    elif change == "pool":
        other[0]["pool"] = "holdout"
    else:
        other[0]["variant"] = "counterfactual"
    with pytest.raises(ValueError):
        helper.compare(samples, rows, other)


def test_zero_probability_does_not_create_nonfinite_nll(helper):
    item = sample(0, "Before event?", "yes")
    row = record(item, -10000.0, 0.0)
    result = helper.compare([item], [row], [copy.deepcopy(row)])
    assert result["summaries"]["all"]["nll"]["baseline"] == 10000
    assert result["summaries"]["all"]["brier"]["baseline"] == 2
