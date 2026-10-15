"""Precompute frozen TRAIN-reference distributions; never replaces sample gold.

Only explicit --execute loads CUDA. Sidecar shares the exact trainer preparation
and shuffled 300-draw schedule. No development or sealed pool input is accepted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.check_tokens import OfficialTokenizerAdapter
from scripts.train_s1 import verify_data_is_train_pool
from train.checkpoint import file_sha256
from train.multimodal import configure_image_processor
from train.trainer import (
    MEDIA_FORWARD_KEYS,
    BoundedPreparedExamples,
    LoopConfig,
    code_revision,
    prepare_mixed_example,
    reference_header,
    reference_order,
    reference_row,
    runtime_requirements,
    smoke_samples,
    verified_assets,
    verified_init_adapter,
)


def precompute(
    config: LoopConfig,
    samples: list[Sample],
    examples: Any,
    model: Any,
    header: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    model.requires_grad_(False)
    model.eval()
    rows = []
    with torch.inference_mode():
        for draw, index in enumerate(reference_order(config, len(examples))):
            example = examples[index]
            ids = torch.tensor([example["input_ids"]], device=device, dtype=torch.long)
            media = {key: example[key].to(device) for key in MEDIA_FORWARD_KEYS if key in example}
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
                **media,
            )
            logits = output.logits[0, -1].float()[example["token_ids"]]
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite frozen reference logits")
            probs = logits.softmax(-1).cpu().tolist()
            rows.append(reference_row(draw, index, samples[index], example, probs))
            del output, ids, media, logits
    return {"header": header, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "config",
        "data",
        "ledger",
        "model-dir",
        "asset-manifest",
        "reference-adapter",
        "output",
        "replay-data",
    ):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--reference-adapter-sha256", required=True)
    parser.add_argument("--replay-data-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "cost": None,
        "code_revision": code_revision(),
        "script_sha256": file_sha256(Path(__file__)),
        "reference_is_gold": False,
        "gpu_requested": args.execute,
    }
    report_path = args.output.with_name(args.output.name + ".report.json")
    # Reserve both artifacts; never truncate previous evidence.
    with report_path.open("x") as receipt:
        try:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError("reference output already exists")
            config = LoopConfig.model_validate_json(args.config.read_bytes())
            if (
                config.seed != 42
                or config.sample_count != 300
                or config.steps * config.accumulation != 300
            ):
                raise ValueError("reference pilot requires seed42 and 300 unique TRAIN draws")
            samples = smoke_samples(
                config,
                args.data,
                expected_data_version=config.expected_data_version,
                expected_data_versions=config.expected_data_versions,
            )
            verify_data_is_train_pool(args.data, [args.ledger])
            identity = verified_init_adapter(args.reference_adapter, args.reference_adapter_sha256)
            hashes = verified_assets(args.model_dir, args.asset_manifest)
            config = config.model_copy(
                update={
                    "reference_ledger_sha256": file_sha256(args.ledger),
                    "reference_replay_data": str(args.replay_data),
                    "reference_replay_sha256": args.replay_data_sha256,
                }
            )
            header = reference_header(config, samples, identity, hashes)
            report.update(status="preflight_verified", exit_code=0, header=header)
            if args.execute:
                report["versions"] = runtime_requirements()
                torch = importlib.import_module("torch")
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
                transformers = importlib.import_module("transformers")
                processor = transformers.AutoProcessor.from_pretrained(
                    str(args.model_dir), local_files_only=True, trust_remote_code=False
                )
                configure_image_processor(processor, config.image_max_soft_tokens)
                tokenizer = OfficialTokenizerAdapter(args.model_dir)
                roots = [Path(p) for p in (config.media_roots or [config.media_root or ""])]
                examples = BoundedPreparedExamples(
                    samples,
                    lambda sample: prepare_mixed_example(
                        sample,
                        tokenizer,
                        processor,
                        roots[0],
                        max_length=config.max_length,
                        extra_roots=roots[1:],
                        video_num_frames=config.video_num_frames,
                    ),
                    config.preparation_cache_bytes,
                    {},
                )
                base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
                    str(args.model_dir),
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                peft = importlib.import_module("peft")
                model = peft.PeftModel.from_pretrained(
                    base, str(args.reference_adapter), is_trainable=False
                ).to("cuda")
                payload = precompute(config, samples, examples, model, header, "cuda")
                if (
                    verified_init_adapter(args.reference_adapter, args.reference_adapter_sha256)
                    != identity
                ):
                    raise ValueError("reference adapter changed during precomputation")
                with args.output.open("x") as sink:
                    json.dump(payload, sink, sort_keys=True, allow_nan=False)
                    sink.write("\n")
                report.update(
                    status="completed",
                    sidecar_sha256=file_sha256(args.output),
                    draws=len(payload["rows"]),
                )
        except Exception as exc:
            report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        finally:
            report["duration_seconds"] = time.monotonic() - start
            json.dump(report, receipt, indent=2)
            receipt.write("\n")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
