"""Calibration runner cannot silently admit training or confirmation pools."""

import hashlib
import json

import pytest

from data.schema import answer_tokens
from data.synth import generate_samples
from scripts.p0_parity import row
from scripts.parity_calibration import load_calibration, truth_metrics
from train.trainer import data_bytes


def write_fixture(tmp_path, pool="calibration"):
    samples = generate_samples(100, seed=113, pool=pool)
    for sample in samples:
        sample.questions = [sample.questions[0]]
    payload = data_bytes(samples)
    (tmp_path / "calibration.jsonl").write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "calibration.jsonl": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "rows": 100,
                        "pool": "calibration",
                    }
                }
            }
        )
    )
    return samples


def test_loader_only_reads_calibration_and_rejects_modified_bytes(tmp_path):
    write_fixture(tmp_path)
    assert len(load_calibration(tmp_path)) == 100
    # No confirmation file exists: loading calibration must not inspect it.
    with (tmp_path / "calibration.jsonl").open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValueError, match="manifest"):
        load_calibration(tmp_path)


@pytest.mark.parametrize("pool", ["train", "dev", "holdout"])
def test_loader_rejects_other_pools_even_with_matching_hash(tmp_path, pool):
    write_fixture(tmp_path, pool)
    with pytest.raises(ValueError, match="calibration"):
        load_calibration(tmp_path)


def test_truth_metrics_keep_abstention_error_separate_from_answer_accuracy():
    sample = generate_samples(1, seed=17, pool="calibration")[0]
    sample.questions = [sample.questions[0]]
    question = sample.questions[0]
    tokens = answer_tokens(question)
    correct = tokens.index(question.target.value)
    logits = [-25.0] * len(tokens) + [-0.001]
    logits[correct] = -20.0
    predicted = row(
        {"id": sample.id, "input_ids": [1], "token_ids": list(range(len(logits)))}, logits
    )
    report = truth_metrics([sample], [predicted])
    assert report["conditional_known_target_by_type"]["choice"]["accuracy"] == 1
    assert report["joint_unique_top1_accuracy"] == 0
    assert report["p_U_mse_by_truth"]["sufficient"]["mse"] > 0.99
    assert report["p_U_mse_by_truth"]["insufficient"]["count"] == 0
    predicted["id_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        truth_metrics([sample], [predicted])
