"""CPU classification of the frozen same12TRAIN learnability diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from train.readout import read_probabilities

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
INIT_SHA = "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classify(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    ce_before = float(before["mean_gold_ce"])
    ce_after = float(after["mean_gold_ce"])
    margin_before = float(before["mean_gold_margin"])
    margin_after = float(after["mean_gold_margin"])
    require(
        all(math.isfinite(x) for x in (ce_before, ce_after, margin_before, margin_after))
        and ce_before >= 0
        and ce_after >= 0,
        "nonfinite/negative diagnostic metrics",
    )
    reduction = (ce_before - ce_after) / ce_before if ce_before else None
    increase = margin_after - margin_before
    if before["correct"] == 12 and ce_before <= 0.05:
        category = "baseline_near_saturation"
        next_step = "stop_this_diagnostic_as_uninformative"
    elif (
        reduction is not None
        and (reduction >= 0.2 or math.isclose(reduction, 0.2, abs_tol=1e-12))
        and increase >= 0.5
    ):
        category = "selected_train_examples_learnable"
        next_step = "coverage_generalization_first"
    elif reduction is not None and reduction < 0.05 and increase < 0.1:
        category = "weak_learning_signal"
        next_step = "optimization_input_diagnostic_first"
    else:
        category = "inconclusive"
        next_step = "no_automatic_sweep_or_expansion"
    return {
        "classification": category,
        "recommended_next_step": next_step,
        "relative_train_ce_reduction": reduction,
        "mean_gold_margin_increase": increase,
        "strong_selected_train_fit": after["correct"] >= 11 and ce_after <= 0.1,
        "thresholds": {
            "near_saturation_correct": 12,
            "near_saturation_ce": 0.05,
            "learnable_relative_ce_drop": 0.2,
            "learnable_margin_increase": 0.5,
            "weak_relative_ce_drop_below": 0.05,
            "weak_margin_increase_below": 0.1,
        },
        "interpretation_boundary": (
            "selected TRAIN learning is not generalization evidence; weak learning "
            "does not prove a representation or backbone ceiling"
        ),
    }


def validate_records(receipt: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    require(
        receipt["status"] == "completed" and receipt["exit_code"] == 0 and len(rows) == 12,
        "evaluation is not completed12TRAIN",
    )
    require(len({row["id"] for row in rows}) == 12, "duplicate TRAIN evaluation IDs")
    require(
        [row["id"] for row in rows] == [binding["id"] for binding in receipt["prepared_bindings"]],
        "records differ from receipt prepared bindings",
    )
    for row, binding in zip(rows, receipt["prepared_bindings"], strict=True):
        require(
            all(row[key] == value for key, value in binding.items()),
            "record differs from receipt prepared input/gold binding",
        )
        require(
            row["pool"] == "train"
            and row["adapter_sha256"] == receipt["adapter"]["weights_sha256"],
            "record pool/adapter identity mismatch",
        )
        logits = row["raw_logits"]
        require(
            set(logits) == {"yes", "no", "U"}
            and all(math.isfinite(value) for value in logits.values()),
            "invalid raw legal logits",
        )
        readout = read_probabilities(logits, ["yes", "no"], temperature=1)
        require(readout.prediction == row["prediction"], "prediction differs from raw legal logits")
        target = row["target"]
        require(target in ("yes", "no"), "requires Noul gold")
        maximum = max(logits.values())
        ce = (
            math.log(math.fsum(math.exp(value - maximum) for value in logits.values()))
            + maximum
            - logits[target]
        )
        margin = (logits["yes"] - logits["no"]) * (1 if target == "yes" else -1)
        require(
            math.isclose(ce, row["gold_ce"], abs_tol=1e-5, rel_tol=1e-5)
            and math.isclose(margin, row["gold_margin"], abs_tol=1e-9),
            "stored CE/margin differs from logits",
        )
    correct = sum(row["prediction"] == row["target"] for row in rows)
    require(
        correct == receipt["correct"]
        and math.isclose(
            math.fsum(row["gold_ce"] for row in rows) / 12, receipt["mean_gold_ce"], abs_tol=1e-9
        )
        and math.isclose(
            math.fsum(row["gold_margin"] for row in rows) / 12,
            receipt["mean_gold_margin"],
            abs_tol=1e-9,
        ),
        "evaluation aggregate metrics differ from records",
    )


def analyze(before_dir: Path, after_dir: Path, training_path: Path) -> dict[str, Any]:
    before = json.loads((before_dir / "receipt.json").read_bytes())
    after = json.loads((after_dir / "receipt.json").read_bytes())
    trained = json.loads(training_path.read_bytes())
    require(
        trained["status"] == "completed"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 10,
        "diagnostic training has not completed10steps",
    )
    require(before["phase"] == "before" and after["phase"] == "after", "evaluation phase mismatch")
    require(
        before["adapter"]["weights_sha256"] == INIT_SHA
        and before["adapter"] == trained["init_adapter"]
        and after["adapter"]["weights_sha256"] == trained["adapter_sha256"],
        "training/evaluation adapter binding mismatch",
    )
    require(
        before["adapter"]["weights_sha256"] != after["adapter"]["weights_sha256"],
        "adapter weights did not change",
    )
    require(
        before["data_sha256"] == after["data_sha256"] == trained["data_sha256"],
        "same12TRAIN data binding mismatch",
    )
    for key in ("prepared_bindings", "model_revision", "dtype", "video_frames", "code_sha256"):
        require(before[key] == after[key], "before/after input/evaluator identity changed:" + key)
    config = trained["config"]
    require(
        (
            config["sample_count"],
            config["steps"],
            config["accumulation"],
            config["seed"],
            config["learning_rate"],
            config["video_num_frames"],
        )
        == (12, 10, 12, 42, 2e-5, 16)
        and config.get("brier_weight", 0)
        == config.get("reference_kl_weight", 0)
        == config.get("pair_loss_weight", 0)
        == 0,
        "training is not frozen pureCE12TRAIN diagnostic",
    )
    records = []
    for directory, receipt in ((before_dir, before), (after_dir, after)):
        path = directory / "records.jsonl"
        require(sha(path) == receipt["records_sha256"], "records SHA mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        validate_records(receipt, rows)
        records.append(rows)
    for left, right in zip(*records, strict=True):
        for key in ("id", "sample_sha256", "input_sha256", "answer_token_ids", "target"):
            require(left[key] == right[key], "same TRAIN paired input/gold differs:" + key)
    result = classify(before, after)
    result.update(
        status="completed",
        exit_code=0,
        gpu_used=False,
        stage_gate=False,
        before_metrics={
            key: before[key] for key in ("correct", "mean_gold_ce", "mean_gold_margin")
        },
        after_metrics={key: after[key] for key in ("correct", "mean_gold_ce", "mean_gold_margin")},
        corrected=sum(
            a["prediction"] != a["target"] and b["prediction"] == b["target"]
            for a, b in zip(*records, strict=True)
        ),
        regressed=sum(
            a["prediction"] == a["target"] and b["prediction"] != b["target"]
            for a, b in zip(*records, strict=True)
        ),
        adapter_updated=True,
        gradient_evidence=(
            "completed10steps of trainer with nonfinite/nonzero guards; "
            "no gradient-norm history is asserted"
        ),
        dev_generalization="not measured in this12TRAIN comparison",
        bindings={
            "before_receipt_sha256": sha(before_dir / "receipt.json"),
            "after_receipt_sha256": sha(after_dir / "receipt.json"),
            "before_records_sha256": sha(before_dir / "records.jsonl"),
            "after_records_sha256": sha(after_dir / "records.jsonl"),
            "training_report_sha256": sha(training_path),
            "runner_sha256": sha(Path(__file__)),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", type=Path, default=ROOT / "artifacts/s4-train-learnability-before-001"
    )
    parser.add_argument(
        "--after", type=Path, default=ROOT / "artifacts/s4-train-learnability-after-001"
    )
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.before, args.after, args.training_report)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "classification",
                    "recommended_next_step",
                    "relative_train_ce_reduction",
                    "mean_gold_margin_increase",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
