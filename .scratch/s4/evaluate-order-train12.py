"""Bounded same12TRAIN before/after logits and complete prepared tensor hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.check_tokens import OfficialTokenizerAdapter
from scripts.s1_text_pools import canonical_sample_sha
from scripts.train_s1 import verify_data_is_train_pool
from train.checkpoint import file_sha256
from train.multimodal import configure_image_processor
from train.readout import read_probabilities
from train.trainer import (
    MEDIA_FORWARD_KEYS,
    LoopConfig,
    data_bytes,
    decision_loss,
    prepare_mixed_example,
    reference_prepared_hash,
    smoke_samples,
    verified_init_adapter,
)

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
MODEL = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
INIT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


PREP_SHA = "7870a39aec8f64939dabd4035f2416b4a1c454f8dd9ec7e88f9891ffbffc28ab"
TRAINING_SHA = "d957e487367e75b186e7d7dc1cb8a2b02631d71d35d8ad4a0733fa0a4c96e19f"
AFTER_SHA = "831d78160b0a7e39413b53da5dd93c5c9bccef82972d511746e302d2ce426a00"
ORDER_VERSION = "clevrer-train-collision-order-v1"


def select_train12(samples, ledger, source_hashes):
    videos = [s for s in samples if s.meta.data_version == ORDER_VERSION]
    require(
        len(videos) == 60 and len({s.meta.family_id for s in videos}) == 30, "fixed60 program gold"
    )
    by_id = {r["item_id"]: r for r in ledger}
    require(len(by_id) == len(ledger), "unique ledger IDs")
    for sample in videos:
        row = by_id[sample.id]
        prov = row["program_provenance"]
        program = prov["program"]
        require(sample.meta.pool == row["pool"] == "train", "TRAIN only")
        require(
            row["derived_sample_sha256"] == canonical_sample_sha(sample), "canonical sample SHA"
        )
        require(row["family_id"] == sample.meta.family_id, "ledger family")
        require(row["media_sha256"] == sample.prefix.media[0].sha256, "ledger media SHA")
        require(
            prov["annotations_zip_sha256"] == source_hashes["annotations"]
            and prov["official_questions_sha256"] == source_hashes["questions"],
            "raw source SHA",
        )
        require(
            hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
            == sample.meta.source_sha256,
            "raw provenance SHA",
        )
        require(
            hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
            == prov["program_sha256"],
            "program SHA",
        )
        frames = [e["frame"] for e in program["events"]]
        require(
            program["operation"] == "less_frame"
            and len(frames) == 2
            and abs(frames[0] - frames[1]) >= 16,
            "program operation",
        )
        truth = "yes" if frames[0] < frames[1] else "no"
        require(program["result"] == sample.questions[0].target.value == truth, "program gold")
    ordered = sorted(
        videos, key=lambda s: hashlib.sha256(("42:order-train12:" + s.id).encode()).hexdigest()
    )
    selected, seen, counts = [], set(), {"yes": 0, "no": 0}
    for sample in ordered:
        target = sample.questions[0].target.value
        if sample.meta.family_id not in seen and counts[target] < 6:
            selected.append(sample)
            seen.add(sample.meta.family_id)
            counts[target] += 1
        if len(selected) == 12:
            break
    require(counts == {"yes": 6, "no": 6} and len(seen) == 12, "balanced12 distinct families")
    return selected, [by_id[s.id] for s in selected]


def prepare_subset(out):
    os.umask(0o077)
    source = ROOT / "artifacts/s4-temporal-order-contrast-pilot-001"
    require(file_sha256(source / "report.json") == PREP_SHA, "source prep SHA")
    report = json.loads((source / "report.json").read_text())
    require(
        report["status"] == "data_ready_only"
        and report["exit_code"] == 0
        and report["inventory_isolation"] == "isolated",
        "source preparation status",
    )
    for name, digest in report["output_sha256"].items():
        require(
            Path(name).name == name and file_sha256(source / name) == digest, "source output SHA"
        )
    training_path = ROOT / "reports/s4-temporal-order-contrast-training-001.json"
    require(file_sha256(training_path) == TRAINING_SHA, "training receipt SHA")
    trained = json.loads(training_path.read_text())
    require(
        trained["status"] == "completed"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 10,
        "completed10 training required",
    )
    require(
        trained["data_sha256"] == report["output_sha256"]["train.jsonl"]
        and trained["adapter_sha256"] == AFTER_SHA
        and trained["init_adapter"]["weights_sha256"] == INIT_SHA,
        "actual training data/parent/after binding",
    )
    config = LoopConfig.model_validate_json((source / "config-ce.json").read_bytes())
    expected = config.model_dump()
    expected.pop("brier_weight", None)
    expected["init_adapter"] = trained["init_adapter"]
    require(trained["config"] == expected, "training config binding")
    samples = [
        Sample.model_validate_json(line)
        for line in (source / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ledger = [
        json.loads(line)
        for line in (source / "allocation-ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(len(samples) == 120 and all(s.meta.pool == "train" for s in samples), "actual TRAIN120")
    verify_data_is_train_pool(source / "train.jsonl", [source / "allocation-ledger.jsonl"])
    selected, selected_ledger = select_train12(samples, ledger, report["source_sha256"])
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "train.jsonl").write_bytes(data_bytes(selected))
    (out / "allocation-ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_ledger)
    )
    subset_config = config.model_dump()
    subset_config["sample_count"] = 12
    (out / "config-ce.json").write_text(json.dumps(subset_config, indent=2) + "\n")
    verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    prepared = {
        "status": "ready_for_preflight",
        "exit_code": 0,
        "samples": 12,
        "init_adapter": trained["init_adapter"],
        "original_config": config.model_dump(),
        "original_data_sha256": trained["data_sha256"],
        "source_prep_sha256": PREP_SHA,
        "training_report_sha256": TRAINING_SHA,
        "output_sha256": {
            name: file_sha256(out / name)
            for name in ["train.jsonl", "allocation-ledger.jsonl", "config-ce.json"]
        },
        "selection": (
            "sha256(42:order-train12:+id), independent family, balanced6yes6no; no predictions used"
        ),
    }
    (out / "report.json").write_text(json.dumps(prepared, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": prepared["status"],
                "samples": 12,
                "data_sha256": prepared["output_sha256"]["train.jsonl"],
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "artifacts/s4-order-train12-input-001"
    )
    parser.add_argument("--phase", choices=("prepare", "before", "after"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--before-records", type=Path)
    parser.add_argument("--before-records-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_subset(args.inputs)
        return 0
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not args.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "phase": args.phase,
        "gpu_requested": args.execute,
        "budget_seconds": 90,
        "scope": "same12 order TRAIN diagnostic, not promotion",
        "script_sha256": file_sha256(Path(__file__)),
    }

    def interrupted(signum, frame):
        raise TimeoutError(f"learnability eval signal:{signum}")

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.alarm(90)
    try:
        source = json.loads((args.inputs / "report.json").read_bytes())
        require(
            source["status"] == "ready_for_preflight"
            and source["exit_code"] == 0
            and source["samples"] == 12,
            "TRAIN diagnostic preparation failed",
        )
        for name, digest in source["output_sha256"].items():
            require(file_sha256(args.inputs / name) == digest, "prepared input SHA changed")
        config = LoopConfig.model_validate_json((args.inputs / "config-ce.json").read_bytes())
        samples = smoke_samples(
            config,
            args.inputs / "train.jsonl",
            expected_data_versions=config.expected_data_versions,
        )
        require(
            len(samples) == len({s.meta.family_id for s in samples}) == 12
            and all(s.meta.pool == "train" for s in samples),
            "same12 independent TRAIN required",
        )
        verify_data_is_train_pool(
            args.inputs / "train.jsonl", [args.inputs / "allocation-ledger.jsonl"]
        )
        identity = verified_init_adapter(args.adapter, args.expected_adapter_sha256)
        before = {}
        if args.phase == "before":
            require(
                args.expected_adapter_sha256 == INIT_SHA
                and identity == source["init_adapter"]
                and args.training_report is None,
                "before must use frozen CE0",
            )
        else:
            require(
                args.training_report is not None
                and args.before_records is not None
                and bool(args.before_records_sha256),
                "after requires training receipt and SHA-bound before records",
            )
            require(file_sha256(args.training_report) == TRAINING_SHA, "training report SHA")
            require(args.expected_adapter_sha256 == AFTER_SHA, "fixed order adapter required")
            trained = json.loads(args.training_report.read_bytes())
            require(
                trained["status"] == "completed"
                and trained["exit_code"] == 0
                and trained["steps_completed"] == 10
                and trained["adapter_sha256"] == args.expected_adapter_sha256
                and trained["data_sha256"] == source["original_data_sha256"]
                and trained["init_adapter"] == source["init_adapter"],
                "after adapter is not the fixed diagnostic training result",
            )
            expected = source["original_config"]
            expected.pop("brier_weight", None)
            expected["init_adapter"] = source["init_adapter"]
            require(
                trained["config"] == expected,
                "after training config differs from fixed CE diagnostic",
            )
            require(
                file_sha256(args.before_records) == args.before_records_sha256,
                "before records SHA mismatch",
            )
            rows = [
                json.loads(line)
                for line in args.before_records.read_text().splitlines()
                if line.strip()
            ]
            require(
                all(row["adapter_sha256"] == INIT_SHA and row["pool"] == "train" for row in rows),
                "before records must come from frozen CE0 on TRAIN",
            )
            before = {row["id"]: row for row in rows}
            require(
                len(rows) == len(before) == 12 and set(before) == {s.id for s in samples},
                "before TRAIN IDs differ",
            )
        transformers = importlib.import_module("transformers")
        processor = transformers.AutoProcessor.from_pretrained(
            str(MODEL), local_files_only=True, trust_remote_code=False
        )
        configure_image_processor(processor, config.image_max_soft_tokens)
        tokenizer = OfficialTokenizerAdapter(MODEL)
        roots = [Path(root) for root in config.media_roots or []]
        examples = [
            prepare_mixed_example(
                sample,
                tokenizer,
                processor,
                roots[0],
                max_length=config.max_length,
                extra_roots=roots[1:],
                video_num_frames=16,
            )
            for sample in samples
        ]
        bindings = [
            {
                "id": sample.id,
                "sample_sha256": hashlib.sha256(data_bytes([sample])).hexdigest(),
                "input_sha256": reference_prepared_hash(
                    {
                        **example,
                        "attention_mask": [1] * len(example["input_ids"]),
                        "use_cache": False,
                        "logits_to_keep": 1,
                    }
                ),
                "answer_token_ids": example["answer_token_ids"],
                "target": sample.questions[0].target.value,
            }
            for sample, example in zip(samples, examples, strict=True)
        ]
        if before:
            for binding in bindings:
                require(
                    all(before[binding["id"]][key] == value for key, value in binding.items()),
                    "before/after complete input/gold/mapping hashes differ",
                )
        report.update(
            status="prepared",
            exit_code=0,
            adapter=identity,
            data_sha256=source["output_sha256"]["train.jsonl"],
            prepared_bindings=bindings,
            model_revision=MODEL.name,
            dtype="bfloat16",
            video_frames=16,
            code_sha256={
                name: file_sha256(Path(name))
                for name in (
                    "train/trainer.py",
                    "train/multimodal.py",
                    "train/template.py",
                    "train/readout.py",
                )
            },
        )
        if args.execute:
            busy = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip()
            require(not busy, "GPU busy; refusing concurrent diagnostic")
            torch = importlib.import_module("torch")
            peft = importlib.import_module("peft")
            report["environment"] = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
            }
            report["cost"] = None
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
                str(MODEL),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            model = (
                peft.PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
                .to("cuda")
                .eval()
            )
            model.requires_grad_(False)
            records = []
            with torch.inference_mode(), (args.output / "records.jsonl").open("x") as sidecar:
                for binding, example in zip(bindings, examples, strict=True):
                    ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
                    media = {
                        key: example[key].to("cuda") for key in MEDIA_FORWARD_KEYS if key in example
                    }
                    output = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                        logits_to_keep=1,
                        return_dict=True,
                        **media,
                    )
                    logits = output.logits[0, -1].float()
                    require(
                        bool(torch.isfinite(logits).all()), "nonfinite complete vocabulary logits"
                    )
                    raw = {
                        token: float(logits[index].item())
                        for token, index in example["answer_token_ids"].items()
                    }
                    readout = read_probabilities(raw, ["yes", "no"], temperature=1)
                    ce = float(
                        decision_loss(
                            logits, example["token_ids"], example["target"], brier_weight=0
                        ).item()
                    )
                    margin = (raw["yes"] - raw["no"]) * (1 if binding["target"] == "yes" else -1)
                    record = {
                        **binding,
                        "pool": "train",
                        "raw_logits": raw,
                        "probabilities": readout.probabilities,
                        "joint_probabilities": readout.joint_probabilities,
                        "p_U": readout.p_insufficient,
                        "prediction": readout.prediction,
                        "gold_ce": ce,
                        "gold_margin": margin,
                        "adapter_sha256": args.expected_adapter_sha256,
                    }
                    records.append(record)
                    sidecar.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    sidecar.flush()
                    del output, logits, ids, media
            require(
                verified_init_adapter(args.adapter, args.expected_adapter_sha256) == identity,
                "adapter changed during evaluation",
            )
            report.update(
                status="completed",
                exit_code=0,
                records_sha256=file_sha256(args.output / "records.jsonl"),
                correct=sum(r["prediction"] == r["target"] for r in records),
                mean_gold_ce=sum(r["gold_ce"] for r in records) / 12,
                mean_gold_margin=sum(r["gold_margin"] for r in records) / 12,
            )
            if before:
                before_ce = sum(r["gold_ce"] for r in before.values()) / 12
                before_margin = sum(r["gold_margin"] for r in before.values()) / 12
                report.update(
                    relative_ce_reduction=(before_ce - report["mean_gold_ce"]) / before_ce
                    if before_ce
                    else None,
                    gold_margin_increase=report["mean_gold_margin"] - before_margin,
                )
            report["baseline_near_saturation"] = (
                report["correct"] == 12 and report["mean_gold_ce"] <= 0.05
                if args.phase == "before"
                else None
            )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        report["duration_seconds"] = time.monotonic() - started
        with (args.output / "receipt.json").open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "status",
                    "exit_code",
                    "correct",
                    "mean_gold_ce",
                    "mean_gold_margin",
                    "baseline_near_saturation",
                    "error",
                )
            },
            sort_keys=True,
        )
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
