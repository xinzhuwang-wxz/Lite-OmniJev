"""CPU public-dev paired diagnostics; never treats family IDs as tensor hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample, answer_tokens
from scripts.s0_development_baselines import sha256_json
from train.readout import read_probabilities

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
BASELINE = "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"
CANDIDATE = "97201e304d26353833845078d1d392fd5e8d966d8d5e3952b2c4d7efb0ebd170"
DATA_SHA = "e9b91fedecf2ccf7dc30e837ac11bcc4b233624d1f6c16812a40b1319c15b080"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def group(sample: Sample) -> str:
    words = set(sample.questions[0].instruction.lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else ("motion" if words & {"moving", "stationary"} else "existence")
    )


def normalized(row: dict[str, Any], sample: Sample) -> dict[str, Any]:
    question = sample.questions[0]
    require(
        sample.meta.pool == "dev" and isinstance(question.target, HardTarget),
        "requires public dev hard gold",
    )
    require(
        row["id"] == sample.id and row["target"] == question.target.value,
        "row/sample target identity mismatch",
    )
    require(
        row.get("pool") == "dev" and row.get("pass") == 0 and row.get("variant") == "original",
        "only one original dev pass is supported",
    )
    require(row.get("family_hash") == sample.meta.family_id, "row/sample family mismatch")
    tokens = answer_tokens(question)
    readout = read_probabilities(row["raw_logits"], tokens, temperature=1)
    require(
        readout.prediction == row["prediction"], "stored prediction differs from raw-logit readout"
    )
    require(
        set(row["probabilities"]) == set(tokens)
        and all(
            math.isclose(
                row["probabilities"][token],
                readout.probabilities[token],
                abs_tol=1e-9,
                rel_tol=1e-9,
            )
            for token in tokens
        ),
        "stored probabilities differ from raw-logit readout",
    )
    values = [float(row["raw_logits"][token]) for token in tokens]
    largest = max(values)
    nll = (
        math.log(math.fsum(math.exp(value - largest) for value in values))
        + largest
        - float(row["raw_logits"][str(question.target.value)])
    )
    brier = math.fsum(
        (readout.probabilities[token] - float(token == question.target.value)) ** 2
        for token in tokens
    )
    return {
        "id": sample.id,
        "target": question.target.value,
        "prediction": readout.prediction,
        "correct": readout.prediction == question.target.value,
        "nll": nll,
        "brier": brier,
        "p_U": readout.p_insufficient,
        "group": group(sample),
    }


def compare(
    samples: list[Sample], baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    ids = [sample.id for sample in samples]
    require(len(ids) == len(set(ids)) and ids, "samples must have unique IDs")
    require(
        [row["id"] for row in baseline] == ids == [row["id"] for row in candidate],
        "paired IDs/order/count mismatch",
    )
    hash_keys = ("input_sha256", "input_summary_sha256")
    any_hashes = any(key in row for row in [*baseline, *candidate] for key in hash_keys)
    if any_hashes:
        require(
            all(
                all(key in left and key in right and left[key] == right[key] for key in hash_keys)
                for left, right in zip(baseline, candidate, strict=True)
            ),
            "paired input hashes missing or differ",
        )
    pairs = [
        (normalized(left, sample), normalized(right, sample))
        for sample, left, right in zip(samples, baseline, candidate, strict=True)
    ]
    summaries = {}
    for name in ("all", "temporal", "motion", "existence"):
        chosen = [(left, right) for left, right in pairs if name == "all" or left["group"] == name]
        if not chosen:
            continue
        count = len(chosen)
        metrics = {
            "count": count,
            "baseline_correct": sum(left["correct"] for left, _ in chosen),
            "candidate_correct": sum(right["correct"] for _, right in chosen),
            "corrected": sum(not left["correct"] and right["correct"] for left, right in chosen),
            "regressed": sum(left["correct"] and not right["correct"] for left, right in chosen),
            "probability_loss_improved": sum(right["nll"] < left["nll"] for left, right in chosen),
            "probability_loss_worsened": sum(right["nll"] > left["nll"] for left, right in chosen),
        }
        for metric in ("nll", "brier", "p_U"):
            before = math.fsum(left[metric] for left, _ in chosen) / count
            after = math.fsum(right[metric] for _, right in chosen) / count
            metrics[metric] = {"baseline": before, "candidate": after, "delta": after - before}
        for label in ("yes", "no", "U"):
            metrics[label + "_predictions"] = {
                "baseline": sum(left["prediction"] == label for left, _ in chosen),
                "candidate": sum(right["prediction"] == label for _, right in chosen),
            }
        metrics["gold_counts"] = {
            str(label): sum(left["target"] == label for left, _ in chosen)
            for label in {left["target"] for left, _ in chosen}
        }
        summaries[name] = metrics
    changed = [
        {
            "id": left["id"],
            "group": left["group"],
            "target": left["target"],
            "baseline_prediction": left["prediction"],
            "candidate_prediction": right["prediction"],
            "baseline_correct": left["correct"],
            "candidate_correct": right["correct"],
            "nll_delta": right["nll"] - left["nll"],
            "brier_delta": right["brier"] - left["brier"],
        }
        for left, right in pairs
        if left["prediction"] != right["prediction"]
    ]
    return {
        "summaries": summaries,
        "changed_predictions": changed,
        "paired_input_hashes_verified": True if any_hashes else None,
        "input_hash_limitation": None
        if any_hashes
        else (
            "historical sidecars did not record input/tensor hashes; "
            "family_hash is not an input hash"
        ),
        "probability_loss_definition": (
            "conditional gold NLL and summed multiclass Brier without U; p_U separately"
        ),
        "scope": "public dev paired diagnostic; no causal or quality-gate conclusion",
    }


def compare_motion(directory: Path) -> dict[str, Any]:
    preparation = directory / "preparation.json"
    prepared = read(preparation)
    require(
        prepared["status"] == "prepared" and prepared["exit_code"] == 0,
        "motion CPU preparation incomplete",
    )
    source = read(ROOT / "reports/s3-dev-eval-video-stable-001.json")
    command = source["command"]
    data = Path(command[command.index("--data") + 1])
    require(
        sha(data)
        == prepared["identity"]["data_sha256"]
        == "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2",
        "motion source mismatch",
    )
    all_samples = [
        Sample.model_validate_json(line) for line in data.read_text().splitlines() if line.strip()
    ]
    require(
        len(all_samples) == 500 and all(s.meta.pool == "dev" for s in all_samples),
        "motion requires original dev500",
    )
    temporal = [s for s in all_samples if group(s) == "temporal"]
    samples = sorted(
        temporal,
        key=lambda s: hashlib.sha256(("42:s4-motion-coverage:" + s.id).encode()).hexdigest(),
    )[:50]
    require(
        [s.id for s in samples] == [row["id"] for row in prepared["selected"]],
        "motion fixed50selection differs",
    )
    manifests = {}
    for digest, bound in prepared["manifests"].items():
        path = Path(bound["path"])
        require(sha(path) == bound["sha256"], "motion manifest SHA differs")
        manifests[digest] = read(path)
    arms = []
    receipts = []
    changed_pixels = changed_positions = changed_input_ids = 0
    for policy in ("uniform", "motion"):
        receipt = read(directory / policy / "receipt.json")
        records = directory / policy / "records.jsonl"
        require(
            receipt["status"] == "completed"
            and receipt["exit_code"] == 0
            and receipt["identity"] == prepared["identity"]
            and sha(records) == receipt["records_sha256"],
            "motion arm binding failed",
        )
        arms.append([json.loads(line) for line in records.read_text().splitlines() if line.strip()])
        receipts.append(receipt)
    converted = [[], []]
    require(len(arms[0]) == len(arms[1]) == len(samples) == 50, "motion paired count mismatch")
    for sample, left, right in zip(samples, *arms, strict=True):
        tokens = answer_tokens(sample.questions[0])
        target = sample.questions[0].target
        require(isinstance(target, HardTarget), "motion gold must be hard")
        for key in (
            "id",
            "target_index",
            "answer_token_ids",
            "prompt_sha256",
            "payload_sha256",
            "input_sha256",
        ):
            require(left[key] == right[key], "motion text/media/gold identity differs:" + key)
        require(
            left["id"] == sample.id and tokens[left["target_index"]] == target.value,
            "motion source gold identity mismatch",
        )
        summaries = [left["input_summary"], right["input_summary"]]
        for row in (left, right):
            require(
                sha256_json(row["input_summary"]) == row["input_summary_sha256"],
                "motion tensor summary hash invalid",
            )
        for key in ("processor_kwargs", "media", "canonical_media"):
            require(summaries[0][key] == summaries[1][key], "motion preprocessing/source differs")
        tensors = [summary["tensors"] for summary in summaries]
        require(set(tensors[0]) == set(tensors[1]), "motion tensor keys differ")
        for key in tensors[0]:
            if key in ("pixel_values_videos", "video_position_ids", "input_ids"):
                require(
                    tensors[0][key]["shape"] == tensors[1][key]["shape"]
                    and tensors[0][key]["dtype"] == tensors[1][key]["dtype"],
                    "motion tensor budget differs",
                )
            else:
                require(
                    tensors[0][key] == tensors[1][key], "motion non-video tensor differs:" + key
                )
        changed_pixels += int(
            tensors[0]["pixel_values_videos"]["sha256"]
            != tensors[1]["pixel_values_videos"]["sha256"]
        )
        changed_positions += int(
            tensors[0]["video_position_ids"]["sha256"] != tensors[1]["video_position_ids"]["sha256"]
        )
        changed_input_ids += int(
            tensors[0]["input_ids"]["sha256"] != tensors[1]["input_ids"]["sha256"]
        )
        manifest = manifests[sample.prefix.media[0].sha256]
        for arm, row in enumerate((left, right)):
            metadata = row["input_summary"]["video_metadata"][0]
            require(
                row["frame_indices"]
                == metadata["frames_indices"]
                == manifest["uniform_indices" if arm == 0 else "adaptive_indices"],
                "motion selected indices differ",
            )
            require(
                all(
                    metadata[key] == manifest[key]
                    for key in ("fps", "duration", "total_num_frames")
                ),
                "motion temporal metadata changed",
            )
            converted[arm].append(
                {
                    "id": sample.id,
                    "target": target.value,
                    "pool": "dev",
                    "pass": 0,
                    "variant": "original",
                    "family_hash": sample.meta.family_id,
                    "prediction": row["prediction"],
                    "probabilities": row["conditional_without_u"],
                    "raw_logits": row["raw_logprobs"],
                }
            )
        require(
            {k: v for k, v in summaries[0]["video_metadata"][0].items() if k != "frames_indices"}
            == {
                k: v for k, v in summaries[1]["video_metadata"][0].items() if k != "frames_indices"
            },
            "motion decode metadata differs beyond indices",
        )
    result = compare(samples, *converted)
    result.update(
        status="completed",
        exit_code=0,
        gpu_used=False,
        stage_gate=False,
        input_hash_limitation=(
            "video pixels and expanded input IDs may vary with frame selection/time metadata; "
            "source prompt/media/gold identities and attention/modality masks verified; "
            "token-level expanded-ID differences cannot be localized from hashes"
        ),
        matched_attention_and_modality_mask_hashes=50,
        changed_expanded_input_id_hashes=changed_input_ids,
        changed_video_pixel_hashes=changed_pixels,
        changed_video_position_hashes=changed_positions,
        model_adapter_sha256=prepared["identity"]["adapter_sha256"],
        bindings={
            "preparation_sha256": sha(preparation),
            "uniform_receipt_sha256": sha(directory / "uniform/receipt.json"),
            "motion_receipt_sha256": sha(directory / "motion/receipt.json"),
            "uniform_records_sha256": sha(directory / "uniform/records.jsonl"),
            "motion_records_sha256": sha(directory / "motion/records.jsonl"),
            "runner_sha256": sha(Path(__file__)),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary", type=Path, default=ROOT / "reports/s4-claim-ce-retention-001.json"
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=ROOT / "artifacts/s4-temporal-curriculum-video150-001/video/report.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path)
    args = parser.parse_args()
    if args.motion_dir:
        result = compare_motion(args.motion_dir)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result["summaries"], sort_keys=True))
        return 0
    summary = read(args.baseline_summary)["results"]["video"]
    before_path = Path(summary["report"])
    before = read(before_path)
    after = read(args.candidate_report)
    require(sha(before_path) == summary["report_sha256"], "baseline report hash mismatch")
    require(
        before["adapter_sha256"] == BASELINE and after["adapter_sha256"] == CANDIDATE,
        "frozen adapter identities differ",
    )
    for report in (before, after):
        require(
            report["status"] == "passed"
            and report["exit_code"] == 0
            and report["samples"] == 150
            and report["passes"] == 1
            and not report["counterfactual"]
            and report["data_sha256"] == DATA_SHA,
            "requires accepted same dev150 one-pass reports",
        )
    for key in (
        "model_revision",
        "script_sha256",
        "versions",
        "video_num_frames",
        "image_max_soft_tokens",
    ):
        require(before[key] == after[key], "evaluator identity/budget differs:" + key)

    def argument(report, flag, default=None):
        command = report["command"]
        return command[command.index(flag) + 1] if flag in command else default

    for flag, default in (
        ("--model-dir", None),
        ("--data", None),
        ("--media-root", None),
        ("--passes", "2"),
        ("--video-frames", "16"),
        ("--max-length", "2048"),
        ("--image-max-soft-tokens", None),
    ):
        require(
            argument(before, flag, default) == argument(after, flag, default),
            "paired evaluator argument differs:" + flag,
        )
    data = Path(argument(before, "--data"))
    require(sha(data) == DATA_SHA, "dev150source changed")
    samples = [
        Sample.model_validate_json(line) for line in data.read_text().splitlines() if line.strip()
    ]
    sidecars = [Path(summary["sidecar"]), Path(argument(after, "--output-dir")) / "records.jsonl"]
    for path, report in zip(sidecars, (before, after), strict=True):
        require(sha(path) == report["sidecar_sha256"], "records sidecar changed")
    rows = [
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for path in sidecars
    ]
    result = compare(samples, *rows)
    result.update(
        status="completed",
        exit_code=0,
        gpu_used=False,
        stage_gate=False,
        bindings={
            "baseline_summary_sha256": sha(args.baseline_summary),
            "baseline_report_sha256": sha(before_path),
            "candidate_report_sha256": sha(args.candidate_report),
            "data_sha256": sha(data),
            "baseline_records_sha256": sha(sidecars[0]),
            "candidate_records_sha256": sha(sidecars[1]),
            "runner_sha256": sha(Path(__file__)),
        },
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summaries"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
