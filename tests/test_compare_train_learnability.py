"""Frozen diagnostic thresholds and paired receipt integrity; CPU only."""

import importlib.util
import json
import math
from pathlib import Path

import pytest


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location(
        "learnability_compare", Path(".scratch/s4/compare-train-learnability.py")
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ((12, 0.04, 3.0), (12, 0.01, 4.0), "baseline_near_saturation"),
        ((10, 1.0, 1.0), (11, 0.8, 1.5), "selected_train_examples_learnable"),
        ((10, 1.0, 1.0), (10, 0.96, 1.09), "weak_learning_signal"),
        ((10, 1.0, 1.0), (10, 1.1, 0.9), "weak_learning_signal"),
        ((10, 1.0, 1.0), (11, 0.81, 1.5), "inconclusive"),
        ((10, 1.0, 1.0), (11, 0.8, 1.49), "inconclusive"),
        ((10, 1.0, 1.0), (10, 0.95, 1.1), "inconclusive"),
    ],
)
def test_frozen_thresholds(helper, before, after, expected):
    def metrics(values):
        return dict(zip(("correct", "mean_gold_ce", "mean_gold_margin"), values, strict=True))

    assert helper.classify(metrics(before), metrics(after))["classification"] == expected


def test_nonfinite_metrics_rejected(helper):
    with pytest.raises(ValueError):
        helper.classify(
            {"correct": 10, "mean_gold_ce": float("nan"), "mean_gold_margin": 1.0},
            {"correct": 11, "mean_gold_ce": 0.1, "mean_gold_margin": 2.0},
        )


@pytest.fixture
def completed(tmp_path, helper):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    init = {"weights_sha256": helper.INIT_SHA, "config_sha256": "a" * 64, "path": "source"}
    config = {
        "sample_count": 12,
        "steps": 10,
        "accumulation": 12,
        "seed": 42,
        "learning_rate": 2e-5,
        "video_num_frames": 16,
    }
    prepared = [
        {
            "id": str(i),
            "sample_sha256": f"{i:064x}",
            "input_sha256": f"{i + 12:064x}",
            "answer_token_ids": {"yes": 0, "no": 1, "U": 2},
            "target": "yes" if i % 2 == 0 else "no",
        }
        for i in range(12)
    ]

    def emit(directory, phase, weights):
        rows = []
        for binding in prepared:
            target = binding["target"]
            logits = {"yes": 0.1, "no": 0.0, "U": -20.0}
            if phase == "after":
                logits = {
                    "yes": 2.0 if target == "yes" else 0.0,
                    "no": 2.0 if target == "no" else 0.0,
                    "U": -20.0,
                }
            high = max(logits.values())
            ce = math.log(sum(math.exp(v - high) for v in logits.values())) + high - logits[target]
            rows.append(
                {
                    **binding,
                    "pool": "train",
                    "adapter_sha256": weights,
                    "raw_logits": logits,
                    "prediction": "yes" if logits["yes"] > logits["no"] else "no",
                    "gold_ce": ce,
                    "gold_margin": (logits["yes"] - logits["no"]) * (1 if target == "yes" else -1),
                }
            )
        path = directory / "records.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = {
            "phase": phase,
            "status": "completed",
            "exit_code": 0,
            "adapter": init if phase == "before" else {"weights_sha256": weights},
            "data_sha256": "c" * 64,
            "prepared_bindings": prepared,
            "model_revision": "model",
            "dtype": "bfloat16",
            "video_frames": 16,
            "code_sha256": {"train/trainer.py": "d" * 64},
            "records_sha256": helper.sha(path),
            "correct": sum(r["prediction"] == r["target"] for r in rows),
            "mean_gold_ce": sum(r["gold_ce"] for r in rows) / 12,
            "mean_gold_margin": sum(r["gold_margin"] for r in rows) / 12,
        }
        (directory / "receipt.json").write_text(json.dumps(receipt))

    emit(before, "before", helper.INIT_SHA)
    emit(after, "after", "b" * 64)
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "steps_completed": 10,
                "config": config,
                "init_adapter": init,
                "adapter_sha256": "b" * 64,
                "data_sha256": "c" * 64,
            }
        )
    )
    return before, after, training


def test_completed_hashed_comparison_classifies_learning_without_claiming_dev_gain(
    helper, completed
):
    result = helper.analyze(*completed)
    assert result["classification"] == "selected_train_examples_learnable"
    assert (
        result["corrected"] == 6 and result["regressed"] == 0 and result["adapter_updated"] is True
    )
    assert "no gradient-norm history" in result["gradient_evidence"]
    assert result["dev_generalization"].startswith("not measured")


@pytest.mark.parametrize(
    "field,value", [("steps_completed", 9), ("adapter_sha256", "same"), ("data_sha256", "wrong")]
)
def test_wrong_training_receipt_rejected(helper, completed, field, value):
    before, after, training = completed
    payload = json.loads(training.read_text())
    payload[field] = value
    training.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        helper.analyze(before, after, training)


def test_rebound_records_cannot_change_input_hash(helper, completed):
    before, after, training = completed
    path = after / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["input_sha256"] = "forged"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = json.loads((after / "receipt.json").read_text())
    receipt["records_sha256"] = helper.sha(path)
    (after / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prepared input/gold"):
        helper.analyze(before, after, training)
