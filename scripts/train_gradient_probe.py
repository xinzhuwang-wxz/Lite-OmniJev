"""Bounded TRAIN-only gold-CE LoRA diagnostic; --execute explicitly enables CUDA."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from scripts.check_tokens import OfficialTokenizerAdapter
from scripts.train_s1 import verify_data_is_train_pool
from train.checkpoint import file_sha256
from train.multimodal import configure_image_processor
from train.trainer import (
    MEDIA_FORWARD_KEYS,
    BoundedPreparedExamples,
    LoopConfig,
    attach_lora,
    code_revision,
    decision_loss,
    prepare_mixed_example,
    reference_order,
    reference_prepared_hash,
    runtime_requirements,
    smoke_samples,
    verified_assets,
    verified_init_adapter,
)

GROUPS = ("video", "crossmodal", "other")


def group(example: dict[str, Any]) -> str:
    video = "pixel_values_videos" in example
    image = "pixel_values" in example
    audio = "input_features" in example
    if video and (image or audio):
        raise ValueError("mixed video inputs are forbidden")
    if image and audio:
        return "crossmodal"
    return "video" if video else "other"


def metrics(
    vectors: dict[str, Any], counts: dict[str, int], losses: dict[str, float]
) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    total = sum(vectors.values())
    all_vectors = {**vectors, "total": total}
    norms = {k: float(torch.linalg.vector_norm(v.double())) for k, v in all_vectors.items()}
    if not all(math.isfinite(n) for n in norms.values()):
        raise ValueError("nonfinite gradient norm")
    pairs = {}
    for a, b in (
        ("video", "crossmodal"),
        ("video", "other"),
        ("crossmodal", "other"),
        ("total", "crossmodal"),
    ):
        dot = float(torch.dot(all_vectors[a].double(), all_vectors[b].double()))
        cosine = dot / (norms[a] * norms[b]) if norms[a] and norms[b] else None
        cosine = max(-1.0, min(1.0, cosine)) if cosine is not None else None
        pairs[f"{a}:{b}"] = {
            "dot": dot,
            "cosine": cosine,
            "angle_degrees": math.degrees(math.acos(cosine)) if cosine is not None else None,
        }
    return {
        "counts": counts,
        "mean_gold_ce": {k: losses[k] / counts[k] if counts[k] else None for k in GROUPS},
        "gradient_norm": norms,
        "pairs": pairs,
        "finite": True,
    }


def probe(config: LoopConfig, examples: Any, model: Any, device: str) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    if config.accumulation != 12 or config.steps < 3:
        raise ValueError("probe requires accumulation12 and at least three configured steps")
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any("lora_" not in n for n, _ in parameters):
        raise ValueError("only LoRA parameters may require gradients")
    model.eval()
    size = sum(p.numel() for _, p in parameters)
    aggregate = {k: torch.zeros(size, dtype=torch.float32) for k in GROUPS}
    aggregate_counts = dict.fromkeys(GROUPS, 0)
    aggregate_losses = dict.fromkeys(GROUPS, 0.0)
    steps = []
    draw_bindings: list[dict[str, Any]] = []
    order = reference_order(config, len(examples))[:36]
    for step in range(3):
        vectors = {k: torch.zeros(size, dtype=torch.float32) for k in GROUPS}
        counts = dict.fromkeys(GROUPS, 0)
        losses = dict.fromkeys(GROUPS, 0.0)
        for index in order[step * 12 : (step + 1) * 12]:
            example = examples[index]
            kind = group(example)
            draw_bindings.append(
                {
                    "draw": len(draw_bindings),
                    "sample_index": index,
                    "sample_id": example.get("id"),
                    "group": kind,
                    "prepared_sha256": reference_prepared_hash(example),
                }
            )
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([example["input_ids"]], device=device, dtype=torch.long)
            media = {k: example[k].to(device) for k in MEDIA_FORWARD_KEYS if k in example}
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
                **media,
            )
            loss = decision_loss(
                output.logits[0, -1], example["token_ids"], example["target"], brier_weight=0.0
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite gold CE")
            (loss / 12).backward()
            offset = 0
            for _, parameter in parameters:
                count = parameter.numel()
                if parameter.grad is not None:
                    gradient = parameter.grad.detach().float().reshape(-1).cpu()
                    if not bool(torch.isfinite(gradient).all()):
                        raise ValueError("nonfinite LoRA gradient")
                    vectors[kind][offset : offset + count].add_(gradient)
                offset += count
            counts[kind] += 1
            losses[kind] += float(loss.detach().cpu())
            del output, loss, ids, media
        steps.append(metrics(vectors, counts, losses))
        for kind in GROUPS:
            aggregate[kind].add_(vectors[kind])
            aggregate_counts[kind] += counts[kind]
            aggregate_losses[kind] += losses[kind]
    model.zero_grad(set_to_none=True)
    return {
        "steps": steps,
        "aggregate": metrics(aggregate, aggregate_counts, aggregate_losses),
        "draws": len(order),
        "draw_bindings": draw_bindings,
        "parameter_names": [n for n, _ in parameters],
        "parameter_count": size,
        "denominator": 12,
        "model_mode": "eval; all dropout disabled; differs from train mode if dropout configured",
        "optimizer_updates": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "data",
        "ledger",
        "model-dir",
        "asset-manifest",
        "init-adapter",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config", "data", "ledger", "asset-manifest", "adapter", "adapter-config"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    start = time.monotonic()
    report: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "gpu_requested": args.execute,
        "code_revision": code_revision(),
        "code_revision_source": (
            "observed local git; unavailable in archives; external receipt must bind archive"
        ),
        "script_sha256": file_sha256(Path(__file__)),
        "cost": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as receipt:
        try:
            if args.expected_code_revision != code_revision():
                raise ValueError("code revision mismatch")
            for name in ("config", "data", "ledger", "asset_manifest"):
                if file_sha256(getattr(args, name)) != getattr(
                    args, "expected_" + name + "_sha256"
                ):
                    raise ValueError(name + " hash mismatch")
            config = LoopConfig.model_validate_json(args.config.read_bytes())
            if config.seed != 42 or config.accumulation != 12 or config.steps < 3:
                raise ValueError("requires seed42, accumulation12, at least three steps")
            samples = smoke_samples(
                config,
                args.data,
                expected_data_version=config.expected_data_version,
                expected_data_versions=config.expected_data_versions,
            )
            report["ledger_audit"] = verify_data_is_train_pool(args.data, [args.ledger])
            identity = verified_init_adapter(args.init_adapter, args.expected_adapter_sha256)
            if identity["config_sha256"] != args.expected_adapter_config_sha256:
                raise ValueError("adapter config hash mismatch")
            report.update(
                adapter=identity,
                asset_hashes=verified_assets(args.model_dir, args.asset_manifest),
                config=config.model_dump(),
                bindings={
                    n: getattr(args, "expected_" + n + "_sha256")
                    for n in ("config", "data", "ledger", "asset_manifest")
                },
            )
            report.update(status="preflight_verified", exit_code=0)
            if args.execute:
                os.environ.update(
                    CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
                )
                report["versions"] = runtime_requirements()
                torch = importlib.import_module("torch")
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.cuda.reset_peak_memory_stats()
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
                model = attach_lora(base, config, identity).to("cuda")
                report["diagnostic"] = probe(config, examples, model, "cuda")
                torch.cuda.synchronize()
                report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
                if (
                    verified_init_adapter(args.init_adapter, args.expected_adapter_sha256)
                    != identity
                ):
                    raise ValueError("adapter changed during probe")
                report["status"] = "completed"
        except Exception as exc:
            report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        finally:
            report["duration_seconds"] = time.monotonic() - start
            json.dump(report, receipt, indent=2, allow_nan=False)
            receipt.write("\n")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
