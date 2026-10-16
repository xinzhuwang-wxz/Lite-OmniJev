"""CPU checks for cached-parent comparison; no tensor-parity or GPU claims."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[1] / ".scratch/s4/evaluate-program-stratified-video150.py"
    )
    spec = importlib.util.spec_from_file_location("program_video150", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def rows(tmp_path):
    samples, records = {}, []
    for i in range(150):
        item_id = f"sample-{i}"
        samples[item_id] = {
            "meta": {"pool": "dev", "family_id": item_id},
            "questions": [
                {"instruction": "Is the moving sphere visible?", "target": {"value": "yes"}}
            ],
        }
        records.append(
            {
                "id": item_id,
                "family_hash": item_id,
                "pool": "dev",
                "target": "yes",
                "n_options": 0,
                "pass": 0,
                "variant": "original",
                "prediction": "yes",
                "raw_logits": {"yes": 1.0, "no": 0.0, "U": -4.0},
                "joint_probabilities": {"yes": 0.7, "no": 0.2, "U": 0.1},
                "probabilities": {"yes": 7 / 9, "no": 2 / 9},
                "p_insufficient": 0.1,
            }
        )
    path = tmp_path / "records.jsonl"

    def write():
        path.write_text("\n".join(json.dumps(row) for row in records))
        return path

    return samples, records, write


def test_actual_cached_records_yield_count_probability_and_group_metrics(rows):
    samples, _, write = rows
    helper = module()
    records = helper.original_records(write(), samples)
    assert helper.compare_records(records, records) == 150
    probability = helper.probability_summary(records)
    assert probability["joint_nll"] == pytest.approx(-math.log(0.7))
    assert probability["conditional_nll"] == pytest.approx(-math.log(7 / 9))
    assert helper.group_scores(records, samples) == {"motion": {"count": 150, "correct": 150}}


@pytest.mark.parametrize(
    "change", ["duplicate", "wronggold", "missingU", "nonfinite", "extra_pass"]
)
def test_ordinary_records_cannot_hide_identity_or_readout_changes(rows, change):
    samples, records, write = rows
    if change == "duplicate":
        records[-1] = records[0]
    elif change == "wronggold":
        records[0]["target"] = "no"
    elif change == "missingU":
        records[0]["raw_logits"].pop("U")
    elif change == "nonfinite":
        records[0]["raw_logits"]["yes"] = float("nan")
    else:
        records[0]["pass"] = 1
    with pytest.raises(ValueError):
        module().original_records(write(), samples)


def test_pair_comparison_checks_family_and_gold_not_just_ids(rows):
    samples, _, write = rows
    helper = module()
    parent = helper.original_records(write(), samples)
    candidate = copy.deepcopy(parent)
    candidate["sample-0"]["family_hash"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        helper.compare_records(parent, candidate)


def test_review_cannot_authorize_different_step_or_adapter():
    helper = module()
    review = {
        "decision": "approve_video150",
        "step": 20,
        "training_report_sha256": "a",
        "dev50_report_sha256": "b",
        "adapter_sha256": "c",
        "max_seconds": 180,
    }
    helper.validate_review(review, 20, "a", "b", "c")
    with pytest.raises(ValueError):
        helper.validate_review(review, 10, "a", "b", "c")
    with pytest.raises(ValueError):
        helper.validate_review(review | {"max_seconds": 181}, 20, "a", "b", "c")
