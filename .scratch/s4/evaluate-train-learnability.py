"""Bounded same12TRAIN before/after logits and complete prepared tensor hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from scripts.check_tokens import OfficialTokenizerAdapter
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
INIT_SHA = "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "artifacts/s4-train-learnability-001")
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--before-records", type=Path)
    parser.add_argument("--before-records-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
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
        "scope": "same12TRAIN learnability diagnostic, not promotion",
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
            config, args.inputs / "train.jsonl", expected_data_version=config.expected_data_version
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
            trained = json.loads(args.training_report.read_bytes())
            require(
                trained["status"] == "completed"
                and trained["exit_code"] == 0
                and trained["steps_completed"] == 10
                and trained["adapter_sha256"] == args.expected_adapter_sha256
                and trained["data_sha256"] == source["output_sha256"]["train.jsonl"]
                and trained["init_adapter"] == source["init_adapter"],
                "after adapter is not the fixed diagnostic training result",
            )
            expected = config.model_dump()
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
            torch = importlib.import_module("torch")
            peft = importlib.import_module("peft")
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
