"""Offline CPU E2B asset/template/PEFT/media checks and gold-bootstrap draft; no GPU."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from scripts.assets import allowed, verify_file, write_json  # noqa: E402
from scripts.check_tokens import OfficialTokenizerAdapter, probe  # noqa: E402
from scripts.train_s1 import verify_data_is_train_pool  # noqa: E402
from train.multimodal import configure_image_processor  # noqa: E402
from train.trainer import (  # noqa: E402
    LoopConfig,
    attach_lora,
    prepare_mixed_example,
    reference_prepared_hash,
    smoke_samples,
    verified_assets,
)

E2B_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
TEACHER_SHA = "e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def draft_config(source):
    config = LoopConfig.model_validate(source)
    if config.sample_count != 1200 or config.accumulation != 12:
        raise ValueError("expected existing 1200-draw S4 program-stratified TRAIN course")
    if (
        config.brier_weight
        or config.reference_kl_weight
        or config.pair_loss_weight
        or config.video_pair_loss_weight
    ):
        raise ValueError("gold bootstrap must not inherit experimental losses")
    return config.model_copy(update={"steps": 100, "checkpoint_every": 5, "midway_step": 10})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "blocked",
        "exit_code": 1,
        "gpu_used": False,
        "scope": "real CPU assets/tokenizer/processor and meta-device LoRA; not model quality",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_script_sha256": sha(__file__),
        "cost": None,
        "teacher": {
            "adapter_sha256": TEACHER_SHA,
            "soft_labels": "pending_not_generated",
            "forward_run": False,
        },
        "calibration_read": False,
        "holdout_read": False,
        "new_model_download": False,
    }
    try:
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

        if torch.cuda.is_initialized():
            raise ValueError("CPU preparation must not initialize CUDA")
        if (
            args.model_dir.name != E2B_REVISION
            or args.model_dir.parent.name != "google__gemma-4-E2B-it"
        ):
            raise ValueError("exact resident E2B snapshot required")
        report["model_dir"] = str(args.model_dir)
        report["manifest"] = {"path": str(args.manifest), "sha256": sha(args.manifest)}
        inventory = json.loads(args.manifest.read_text())
        matches = [
            m
            for m in inventory["models"]
            if m["repo_id"] == "google/gemma-4-E2B-it" and m["revision"] == E2B_REVISION
        ]
        if len(matches) != 1:
            raise ValueError("project inventory must contain exactly the pinned E2B")
        source_model = matches[0]
        converted = []
        extra_hashes = {}
        for item in source_model["files"]:
            if not item["required"]:
                continue
            entry = {
                "path": item["path"],
                "size": item["bytes"],
                "kind": item["kind"],
                "upstream_sha256": item.get("lfs_sha256"),
                "git_blob_sha1": item.get("git_blob_sha1"),
            }
            if allowed(item["path"]):
                converted.append(entry)
            elif item["path"] == ".gitattributes":
                extra_hashes[item["path"]] = verify_file(args.model_dir / item["path"], entry)
            else:
                raise ValueError("unexpected runtime-excluded inventory file")
        runtime_manifest = args.output / "e2b-runtime-manifest.json"
        write_json(
            runtime_manifest,
            {
                "schema_version": 1,
                "models": [
                    {
                        "repo_id": source_model["repo_id"],
                        "revision": source_model["revision"],
                        "license": source_model["license"],
                        "files": converted,
                    }
                ],
            },
        )
        report["runtime_manifest"] = {
            "path": str(runtime_manifest),
            "sha256": sha(runtime_manifest),
            "derivation": (
                "lossless bytes→size/lfs_sha256→upstream_sha256; .gitattributes verified separately"
            ),
        }
        report["asset_hashes"] = {
            **verified_assets(args.model_dir, runtime_manifest),
            **extra_hashes,
        }
        report["resident_bytes"] = sum(
            (args.model_dir / name).stat().st_size for name in report["asset_hashes"]
        )
        token_report = probe(args.model_dir)
        token_report.pop("rendered_prompt", None)
        if token_report["exit_code"] or token_report["answer_count"] != 33:
            raise ValueError("33 single-token official-template probe failed")
        report["token_probe"] = token_report
        print("assets and 33-token official template verified", flush=True)
        pool = args.root / "artifacts/s4-program-stratified-pool-001"
        data = pool / "train.jsonl"
        ledger = pool / "allocation-ledger.jsonl"
        source_config = pool / "config-ce.json"
        source_receipt = pool / "report.json"
        if (
            sha(source_receipt)
            != "ac2f17fc3f5417e93e04d15292763384858f5e3ba1c4de533ca6f5e57d0aaf99"
        ):
            raise ValueError("source course receipt differs from audited S4 course")
        source_record = json.loads(source_receipt.read_text())
        for path in (data, ledger, source_config, pool / "components.jsonl"):
            if sha(path) != source_record["output_sha256"][path.name]:
                raise ValueError(f"source course file hash mismatch: {path.name}")
        config = draft_config(json.loads(source_config.read_text()))
        report["ledger_check"] = verify_data_is_train_pool(data, [ledger])
        samples = smoke_samples(config, data, expected_data_versions=config.expected_data_versions)
        report["course"] = {
            "kind": "gold_supervised_bootstrap_not_distillation",
            "source": str(pool),
            "data_sha256": sha(data),
            "ledger_sha256": sha(ledger),
            "source_config_sha256": sha(source_config),
            "source_report_sha256": sha(pool / "report.json"),
            "sample_count": len(samples),
            "unique_families": len({s.meta.family_id for s in samples}),
            "data_versions": dict(collections.Counter(s.meta.data_version for s in samples)),
            "target_kinds": dict(collections.Counter(s.questions[0].target.kind for s in samples)),
            "modality_counts": dict(
                collections.Counter("+".join(s.meta.modality) for s in samples)
            ),
            "gold_priority": "original admitted targets unchanged; no teacher-derived labels added",
            "e4b_adapter_inherited": False,
        }
        # Instantiate the real architecture and PEFT wiring without allocating model weights.
        model_config = AutoConfig.from_pretrained(
            str(args.model_dir), local_files_only=True, trust_remote_code=False
        )
        with init_empty_weights(include_buffers=True):
            base = Gemma4ForConditionalGeneration(model_config)
            from safetensors import safe_open

            projections = {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            }
            verified_shapes = {}
            with safe_open(
                str(args.model_dir / "model.safetensors"), framework="pt", device="cpu"
            ) as weights:
                keys = set(weights.keys())
                for name, module in base.named_modules():
                    if (
                        name.startswith("model.language_model.layers.")
                        and name.split(".")[-1] in projections
                    ):
                        key = name + ".weight"
                        if key not in keys or weights.get_slice(key).get_shape() != list(
                            module.weight.shape
                        ):
                            raise ValueError(
                                "resident decoder projection does not match configured E2B"
                            )
                        verified_shapes[key] = list(module.weight.shape)
            report["resident_projection_shapes"] = {
                "count": len(verified_shapes),
                "sha256": hashlib.sha256(
                    json.dumps(verified_shapes, sort_keys=True).encode()
                ).hexdigest(),
                "tensor_payload_loaded": False,
            }
            adapted = attach_lora(base, config)
        names = [name for name, p in adapted.named_parameters() if p.requires_grad]
        if not names or any(
            ".lora_" not in name or ".language_model.layers." not in name for name in names
        ):
            raise ValueError("non-decoder LoRA parameters became trainable")
        report["peft_meta"] = {
            "status": "passed",
            "real_architecture": type(base).__name__,
            "decoder_layers": model_config.text_config.num_hidden_layers,
            "trainable_parameter_count": sum(
                p.numel() for p in adapted.parameters() if p.requires_grad
            ),
            "trainable_tensors": len(names),
            "all_parameters_meta": all(p.device.type == "meta" for p in adapted.parameters()),
            "trainable_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "rank": config.rank,
            "alpha": config.alpha,
            "weights_loaded": False,
        }
        del adapted, base
        gc.collect()
        processor = AutoProcessor.from_pretrained(
            str(args.model_dir), local_files_only=True, trust_remote_code=False
        )
        configure_image_processor(processor, config.image_max_soft_tokens)
        tokenizer = OfficialTokenizerAdapter(args.model_dir)
        # One per admitted source-version: covers screenshot separately from photo.
        representatives = {}
        for sample in samples:
            representatives.setdefault(sample.meta.data_version, sample)
        prepared = []
        roots = [Path(p) for p in config.media_roots or []]
        for version, sample in representatives.items():
            item = prepare_mixed_example(
                sample,
                tokenizer,
                processor,
                roots[0],
                max_length=config.max_length,
                extra_roots=roots[1:],
                video_num_frames=config.video_num_frames,
            )
            prepared.append(
                {
                    "data_version": version,
                    "input_tokens": len(item["input_ids"]),
                    "prepared_sha256": reference_prepared_hash(item),
                    "target_count": len(item["target"]),
                    "cpu_tensor_shapes": {
                        k: list(v.shape) for k, v in item.items() if hasattr(v, "shape")
                    },
                }
            )
            print("CPU processor verified", version, flush=True)
        report["processor_checks"] = prepared
        report["versions"] = {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "peft", "accelerate", "safetensors"]
        }
        config_path = args.output / "gold-bootstrap-config.json"
        write_json(config_path, config.model_dump())
        report["draft_config"] = {
            "path": str(config_path),
            "sha256": sha(config_path),
            "config": config.model_dump(),
        }
        output = args.root / "artifacts/s6-e2b-gold-bootstrap-001"
        common = [
            "-m",
            "scripts.train_s1",
            "run",
            "--config",
            str(config_path),
            "--model-dir",
            str(args.model_dir),
            "--asset-manifest",
            str(runtime_manifest),
            "--data",
            str(data),
            "--allocation-ledger",
            str(ledger),
            "--output-dir",
            str(output),
            "--invocation-steps",
            "10",
        ]
        python = os.sys.executable
        report["commands"] = {
            "working_directory": str(Path.cwd()),
            "first10": shlex.join(
                [
                    "timeout",
                    "--signal=INT",
                    "--kill-after=20s",
                    "220s",
                    python,
                    *common,
                    "--report",
                    str(args.root / "reports/s6-e2b-gold-step0010.json"),
                ]
            ),
            "next10_example_after_review": shlex.join(
                [
                    "timeout",
                    "--signal=INT",
                    "--kill-after=20s",
                    "220s",
                    python,
                    *common,
                    "--resume-from",
                    str(output / "checkpoints/step-0010"),
                    "--report",
                    str(args.root / "reports/s6-e2b-gold-step0020.json"),
                ]
            ),
            "continuation": (
                "Repeat only after measured review, using latest immutable checkpoint "
                "and new report filename; global100steps/1200draw not automatic authorization."
            ),
            "gpu_environment": (
                "Launch in a fresh shell without CUDA_VISIBLE_DEVICES empty; "
                "this CPU script never launches these commands."
            ),
        }
        report["gpu_initialized"] = torch.cuda.is_initialized()
        if report["gpu_initialized"]:
            raise ValueError("unexpected CUDA initialization")
        report.update(status="cpu_ready_gold_bootstrap_distillation_pending", exit_code=0)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["duration_seconds"] = time.monotonic() - started
    write_json(args.output / "report.json", report)
    print(
        json.dumps({k: report[k] for k in ["status", "exit_code", "duration_seconds"]}), flush=True
    )
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
