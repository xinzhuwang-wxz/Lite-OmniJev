"""Bounded text-only decision LoRA smoke; no quality or GPU-parity claims."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from data.schema import (
    Contract,
    HardTarget,
    InsufficientTarget,
    Sample,
    SoftTarget,
    answer_tokens,
    validate_pool_isolation,
)
from data.synth import generate_samples
from scripts.assets import selected_files, verify_file, write_json
from scripts.check_tokens import OfficialTokenizerAdapter
from train.checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from train.monitoring import SwanLabConfig, Telemetry, start_swanlab
from train.template import check_answer_tokens, render_prompt


class SmokeConfig(Contract):
    seed: int = Field(default=42, ge=0)
    sample_count: Literal[100] = 100
    steps: Literal[10] = 10
    accumulation: Literal[10] = 10
    max_length: int = Field(default=2048, ge=32, le=4096)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.001)
    rank: Literal[16] = 16
    alpha: Literal[32] = 32
    dtype: Literal["bfloat16"] = "bfloat16"
    modality: Literal["text"] = "text"

    @model_validator(mode="after")
    def finite_values(self) -> SmokeConfig:
        if not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite")
        return self


def smoke_samples(config: SmokeConfig, data_path: Path | None = None) -> list[Sample]:
    if data_path is None:
        samples = generate_samples(config.sample_count, config.seed)
        # Keep exactly one question per source sample, balanced across four target tasks.
        for index, sample in enumerate(samples):
            sample.questions = [sample.questions[index % len(sample.questions)]]
    else:
        samples = [
            Sample.model_validate_json(line)
            for line in data_path.read_text().splitlines()
            if line.strip()
        ]
    if len(samples) != config.sample_count or any(len(s.questions) != 1 for s in samples):
        raise ValueError("smoke requires exactly 100 samples with one question each")
    validate_pool_isolation(samples)
    for sample in samples:
        if sample.meta.pool != "train" or sample.meta.modality != ["text"] or sample.prefix.media:
            raise ValueError("smoke accepts text-only training-pool samples")
    return samples


def data_bytes(samples: list[Sample]) -> bytes:
    return ("\n".join(s.model_dump_json() for s in samples) + "\n").encode()


def prepare_examples(
    samples: list[Sample], tokenizer: Any, max_length: int
) -> list[dict[str, Any]]:
    examples = []
    for sample in samples:
        question = sample.questions[0]
        tokens = (*answer_tokens(question), "U")
        mapping = check_answer_tokens(tokenizer, question)
        prompt = render_prompt(tokenizer, sample.prefix, question)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids or len(input_ids) > max_length:
            raise ValueError("empty or over-length prompt; truncation is forbidden")
        for token in tokens:
            if tokenizer.encode(prompt + token, add_special_tokens=False) != input_ids + [
                mapping[token]
            ]:
                raise ValueError(f"answer token changes at generation boundary: {token}")
        target = question.target
        if isinstance(target, HardTarget):
            expected = [float(t == str(target.value)) for t in tokens]
        elif isinstance(target, SoftTarget):
            expected = [*target.probs, 0.0]
        elif isinstance(target, InsufficientTarget):
            expected = [0.0] * (len(tokens) - 1) + [1.0]
        else:
            raise ValueError("sample lacks supervision")
        examples.append(
            {
                "id": sample.id,
                "input_ids": input_ids,
                "token_ids": [mapping[t] for t in tokens],
                "target": expected,
            }
        )
    return examples


def decision_loss(logits: Any, token_ids: list[int], target: list[float]) -> Any:
    """CE on all legal original logits including U, with differentiable FP32 reduction."""
    torch = importlib.import_module("torch")
    if logits.ndim != 1 or not 3 <= len(token_ids) <= 21 or len(token_ids) != len(target):
        raise ValueError("loss expects one vocabulary vector and 2–20 answers plus U")
    if len(set(token_ids)) != len(token_ids) or any(
        type(i) is not int or i < 0 or i >= logits.shape[0] for i in token_ids
    ):
        raise ValueError("option token IDs must be distinct vocabulary indices")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in target) or not math.isclose(
        math.fsum(target), 1.0, abs_tol=1e-9
    ):
        raise ValueError("target must be a finite probability distribution including U")
    selected = logits[token_ids].float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("nonfinite legal-token logits")
    labels = torch.tensor(target, device=selected.device, dtype=torch.float32)
    return -(labels * selected.log_softmax(dim=-1)).sum()


def verified_assets(model_dir: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    models = manifest.get("models", [])
    matching = [m for m in models if m.get("revision") == model_dir.name]
    if len(matching) != 1 or matching[0].get("repo_id") != "google/gemma-4-E4B-it":
        raise ValueError("model directory must identify the manifest's pinned E4B revision")
    model = matching[0]
    if not re.fullmatch(r"[0-9a-f]{40}", model_dir.name):
        raise ValueError("model revision must be immutable")
    scoped = {**manifest, "models": [model]}
    files = selected_files(scoped, metadata_only=False)
    if not any(file["kind"] == "weights" for _, file in files):
        raise ValueError("manifest contains no model weights")
    hashes = {}
    for _, file in files:
        path = model_dir / file["path"]
        if not path.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError("model file escapes pinned directory")
        hashes[file["path"]] = verify_file(path, file)
    expected_weights = {f["path"] for _, f in files if f["kind"] == "weights"}
    if {p.name for p in model_dir.glob("model*.safetensors")} != expected_weights:
        raise ValueError("unverified or missing safetensors files in model directory")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards or shards != expected_weights:
            raise ValueError("safetensors index references unverified shards")
    model_config = json.loads((model_dir / "config.json").read_text())
    if model_config.get("model_type") != "gemma4" or model_config.get("architectures") != [
        "Gemma4ForConditionalGeneration"
    ]:
        raise ValueError("unsupported model architecture")
    return hashes


def runtime_requirements() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        versions[package] = importlib.metadata.version(package)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
        raise ValueError("installed Transformers lacks Gemma4ForConditionalGeneration")
    if not hasattr(peft, "get_peft_model") or not hasattr(peft, "LoraConfig"):
        raise ValueError("installed PEFT lacks required LoRA APIs")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for the real Gemma smoke; CPU/MPS is not P0")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device must support bfloat16")
    return versions


def attach_lora(model: Any, config: SmokeConfig) -> Any:
    """Freeze all original weights and target only decoder attention/MLP projections."""
    peft = importlib.import_module("peft")
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [
        name
        for name, _ in model.named_modules()
        if name.startswith("model.language_model.layers.") and name.split(".")[-1] in projections
    ]
    if {name.split(".")[-1] for name in targets} != projections:
        raise ValueError("cannot locate all intended Gemma decoder projections")
    model.requires_grad_(False)
    adapted = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            inference_mode=False,
        ),
    )
    trainable = [name for name, p in adapted.named_parameters() if p.requires_grad]
    if not trainable or any(
        ".lora_" not in name or ".language_model.layers." not in name for name in trainable
    ):
        raise ValueError("trainable parameters extend beyond decoder LoRA")
    return adapted


def log_training_metrics(
    telemetry: Telemetry,
    config: SmokeConfig,
    report: dict[str, Any],
    *,
    mean_loss: float,
    grad_norm: float,
) -> None:
    telemetry.log(
        {
            "train/loss": mean_loss,
            "train/learning_rate": config.learning_rate,
            "train/grad_norm": grad_norm,
            "train/steps_completed": float(report["steps_completed"]),
        },
        step=int(report["steps_completed"]),
    )


def train_smoke(
    config: SmokeConfig,
    model_dir: Path,
    examples: list[dict[str, Any]],
    output_dir: Path,
    report: dict[str, Any],
    telemetry: Telemetry,
    resume_from: Path | None = None,
) -> None:
    """Execute 10 optimizer steps, each accumulating 10 single-sequence examples."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = attach_lora(base, config).to("cuda")
    model.train()
    # Encoders stay in eval mode as well as frozen; this smoke never supplies media.
    for name, module in model.named_modules():
        if name.endswith(("vision_tower", "audio_tower")):
            module.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    report["trainable_parameters"] = sum(p.numel() for p in parameters)
    report["device"] = torch.cuda.get_device_name(0)
    report["cuda_version"] = torch.version.cuda
    order = list(range(len(examples)))
    random.Random(config.seed).shuffle(order)
    checkpoint = None
    start_step = 0
    report["steps_completed"] = 0
    report["losses"] = []
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        start_step = verify_checkpoint(
            checkpoint,
            config=config.model_dump(),
            model_revision=report["model_revision"],
            data_sha256=report["data_sha256"],
            asset_hashes=report["asset_hashes"],
            order=order,
        )
        if start_step > config.steps:
            raise ValueError("checkpoint step exceeds configured smoke length")
        restore_checkpoint(model, optimizer, checkpoint)
        report["steps_completed"] = start_step
        report["losses"] = list(checkpoint["losses"])
        if len(report["losses"]) != start_step:
            raise ValueError("checkpoint losses do not match completed steps")
        report["resumed_from"] = str(resume_from)
    telemetry.log(
        {
            "validation/examples": float(report["examples"]),
            "validation/max_prompt_tokens": float(report["max_prompt_tokens"]),
            "train/learning_rate": config.learning_rate,
        },
        step=start_step,
    )
    for step in range(start_step, config.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for offset in range(config.accumulation):
            example = examples[order[step * config.accumulation + offset]]
            ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
            loss = decision_loss(output.logits[0, -1], example["token_ids"], example["target"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            (loss / config.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError("no LoRA gradient reached the decision loss")
        optimizer.step()
        report["steps_completed"] = step + 1
        mean_loss = math.fsum(losses) / len(losses)
        report["losses"].append(mean_loss)
        log_training_metrics(telemetry, config, report, mean_loss=mean_loss, grad_norm=float(norm))
        checkpoint_path = save_checkpoint(
            output_dir,
            step=report["steps_completed"],
            config=config.model_dump(),
            model_revision=report["model_revision"],
            data_sha256=report["data_sha256"],
            asset_hashes=report["asset_hashes"],
            order=order,
            losses=report["losses"],
            model=model,
            optimizer=optimizer,
        )
        report["checkpoint"] = str(checkpoint_path)
        write_json(output_dir / "run.json", report)
    model.save_pretrained(
        str(output_dir / "adapter"), safe_serialization=True, save_embedding_layers=False
    )
    adapter = output_dir / "adapter" / "adapter_model.safetensors"
    if not adapter.is_file():
        raise ValueError("PEFT did not save a safetensors adapter")
    report["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()


def code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run(
    config: SmokeConfig,
    model_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    execute: bool,
    data_path: Path | None = None,
    monitoring: SwanLabConfig | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    report: dict[str, Any] = {
        "status": "blocked",
        "exit_code": 1,
        "code_revision": code_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config.model_dump(),
        "model_revision": model_dir.name,
        "scope": "text-only real-model plumbing smoke; not P0 completion or quality evidence",
        "resume_supported": True,
        "cost": None,
        "steps_completed": 0,
    }
    monitoring_config = monitoring or SwanLabConfig.from_env()
    report["monitoring"] = monitoring_config.describe()
    created = False
    telemetry: Telemetry | None = None
    try:
        if execute:
            if resume_from is None:
                output_dir.mkdir(parents=True, exist_ok=False)
                created = True
            else:
                checkpoint_root = (output_dir / "checkpoints").resolve()
                if not resume_from.resolve().is_relative_to(checkpoint_root):
                    raise ValueError("resume checkpoint must be inside output-dir/checkpoints")
                if not output_dir.is_dir():
                    raise ValueError("resume requires an existing output directory")
        samples = smoke_samples(config, data_path)
        report["data_sha256"] = hashlib.sha256(data_bytes(samples)).hexdigest()
        blockers = []
        operations: list[tuple[str, Callable[[], dict[str, str]]]] = [
            ("asset_hashes", lambda: verified_assets(model_dir, manifest_path)),
            ("versions", runtime_requirements),
        ]
        for field, operation in operations:
            try:
                report[field] = operation()
            except Exception as exc:
                blockers.append(f"{field}: {type(exc).__name__}: {exc}")
        examples = []
        try:
            adapter = OfficialTokenizerAdapter(model_dir)
            examples = prepare_examples(samples, adapter, config.max_length)
            report["examples"] = len(examples)
            report["max_prompt_tokens"] = max(len(e["input_ids"]) for e in examples)
        except Exception as exc:
            blockers.append(f"examples: {type(exc).__name__}: {exc}")
        blockers.extend(monitoring_config.blockers())
        report["blockers"] = blockers
        if blockers:
            raise ValueError("training preflight has blockers")
        if execute:
            payload = data_bytes(samples)
            data_output = output_dir / "data.jsonl"
            if resume_from is None:
                data_output.write_bytes(payload)
            elif (
                not data_output.is_file()
                or hashlib.sha256(data_output.read_bytes()).hexdigest() != report["data_sha256"]
            ):
                raise ValueError("resume data.jsonl is missing or does not match current data")
            report["status"] = "running"
            write_json(output_dir / "run.json", report)
            telemetry = start_swanlab(monitoring_config, report)
            try:
                train_smoke(
                    config,
                    model_dir,
                    examples,
                    output_dir,
                    report,
                    telemetry,
                    resume_from=resume_from,
                )
            finally:
                telemetry.finish(report)
        report.update(status="completed" if execute else "ready", exit_code=0)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="failed" if report["status"] == "running" else "blocked",
            error=f"{type(exc).__name__}: {exc}",
        )
    report["duration_seconds"] = time.monotonic() - start
    if created or (execute and resume_from is not None and output_dir.is_dir()):
        write_json(output_dir / "run.json", report)
    return report
