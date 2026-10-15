"""Bounded text/image decision LoRA smoke; no quality or GPU-parity claims."""

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
    Prefix,
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
    verify_resume_request,
)
from train.monitoring import SwanLabConfig, Telemetry, start_swanlab
from train.multimodal import load_media, resolve_media
from train.template import check_answer_tokens, render_payload, render_prompt


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


class LoopConfig(Contract):
    """S1-scale training over a frozen pool; P0 smoke keeps its fixed Literals."""

    seed: int = Field(default=42, ge=0)
    sample_count: int = Field(ge=2, le=100_000)
    steps: int = Field(ge=1, le=100_000)
    accumulation: int = Field(ge=1, le=64)
    max_length: int = Field(default=2048, ge=32, le=4096)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.001)
    rank: Literal[16] = 16
    alpha: Literal[32] = 32
    dtype: Literal["bfloat16"] = "bfloat16"
    modality: Literal["text", "image", "screenshot"] = "text"
    media_root: str | None = None
    midway_step: int | None = Field(default=None, ge=1)
    expected_data_version: str | None = None
    checkpoint_every: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_loop(self) -> LoopConfig:
        if not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite")
        if self.midway_step is not None and self.midway_step > self.steps:
            raise ValueError("midway_step must not exceed steps")
        if self.modality in {"image", "screenshot"} and not (self.media_root or "").strip():
            raise ValueError(f"{self.modality} training requires a nonempty media_root")
        return self


TrainConfig = SmokeConfig | LoopConfig


def smoke_samples(
    config: TrainConfig,
    data_path: Path | None = None,
    *,
    expected_data_version: str | None = None,
) -> list[Sample]:
    if data_path is None and config.modality != "text":
        raise ValueError("image/screenshot cohorts require a frozen data file")
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
        raise ValueError(
            f"training requires exactly {config.sample_count} samples, one question each"
        )
    validate_pool_isolation(samples)
    for sample in samples:
        if config.modality != "text":
            if sample.meta.pool != "train":
                raise ValueError("smoke accepts training-pool samples")
            if (
                len(sample.prefix.media) != 1
                or sample.prefix.media[0].type != "image"
                or "image" not in sample.meta.modality
            ):
                raise ValueError(
                    f"{config.modality} training requires exactly one image per sample"
                )
        elif sample.meta.pool != "train" or sample.meta.modality != ["text"] or sample.prefix.media:
            raise ValueError("smoke accepts text-only training-pool samples")
        if expected_data_version is not None and sample.meta.data_version != expected_data_version:
            raise ValueError("training data_version does not match the frozen allocation")
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


def verified_media_file(media_root: Path, uri: str, sha256: str) -> Path:
    """Streaming-sha-verified local media path under media_root, per the S0 asset rules."""
    return resolve_media(media_root, uri, sha256, max_bytes=64 * 1024 * 1024)


def prepare_media_example(
    sample: Sample,
    processor: Any,
    media_root: Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Official-processor rendering for one image sample, isomorphic to s0_baseline_cpu."""
    question = sample.questions[0]
    if len(sample.prefix.media) != 1 or sample.prefix.media[0].type != "image":
        raise ValueError(f"sample {sample.id} must carry exactly one image")
    item = sample.prefix.media[0]
    image = load_media(
        verified_media_file(media_root, item.uri, item.sha256),
        "image",
        num_frames=16,
        max_seconds=60,
    )
    payload = render_payload(Prefix(state=sample.prefix.state), question)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": payload}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("official image template returned empty prompt")
    tokenizer = processor.tokenizer
    tokens = (*answer_tokens(question), "U")
    mapping = check_answer_tokens(tokenizer, question)
    inputs = processor(text=[prompt], images=[[image]], return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0].tolist()
    if "pixel_values" not in inputs:
        raise ValueError("processor omitted pixel_values")
    if not input_ids or len(input_ids) > max_length:
        raise ValueError("empty or over-length multimodal prompt; truncation is forbidden")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for token in tokens:
        if tokenizer.encode(prompt + token, add_special_tokens=False) != prompt_ids + [
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
    return {
        "id": sample.id,
        "input_ids": input_ids,
        "pixel_values": inputs["pixel_values"],
        "token_ids": [mapping[t] for t in tokens],
        "answer_token_ids": mapping,
        "target": expected,
    }


def prepare_multimodal_examples(
    samples: list[Sample],
    model_dir: Path,
    media_root: Path,
    max_length: int,
) -> list[dict[str, Any]]:
    """Cohort wrapper: load the official processor once, render every sample."""
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    return [
        prepare_media_example(sample, processor, media_root, max_length=max_length)
        for sample in samples
    ]


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


def attach_lora(model: Any, config: TrainConfig) -> Any:
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
    config: TrainConfig,
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
    config: TrainConfig,
    model_dir: Path,
    examples: list[dict[str, Any]],
    output_dir: Path,
    report: dict[str, Any],
    telemetry: Telemetry,
    resume_from: Path | None = None,
    resume_step: int = 0,
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
    for name, parameter in model.named_parameters():
        if ".vision_tower." in name or ".audio_tower." in name:
            if parameter.requires_grad:
                raise ValueError(f"encoder tower must stay frozen under decision LoRA: {name}")
    model.train()
    # Encoders stay in eval mode as well as frozen; media cohorts only pass pixels through.
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
    start_step = resume_step
    report["steps_completed"] = 0
    report["losses"] = []
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        restore_checkpoint(model, optimizer, checkpoint)
        report["steps_completed"] = start_step
        report["losses"] = list(checkpoint["losses"])
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
            draw = step * config.accumulation + offset
            example = examples[order[draw % len(examples)]]
            ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
            if "pixel_values" in example:
                output = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    pixel_values=example["pixel_values"].to(ids.device),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            else:
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
        midway = getattr(config, "midway_step", None)
        if midway is not None and report["steps_completed"] == midway:
            midway_dir = output_dir / f"adapter-at-step-{midway:04d}"
            model.save_pretrained(
                str(midway_dir), safe_serialization=True, save_embedding_layers=False
            )
            midway_adapter = midway_dir / "adapter_model.safetensors"
            if not midway_adapter.is_file():
                raise ValueError("PEFT did not save the midway safetensors adapter")
            report.setdefault("midway_adapters", []).append(
                {
                    "step": midway,
                    "path": str(midway_dir),
                    "sha256": hashlib.sha256(midway_adapter.read_bytes()).hexdigest(),
                }
            )
            write_json(output_dir / "run.json", report)
        every = getattr(config, "checkpoint_every", None)
        if (
            every is None
            or report["steps_completed"] % every == 0
            or report["steps_completed"] == config.steps
        ):
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
    config: TrainConfig,
    model_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    execute: bool,
    data_path: Path | None = None,
    monitoring: SwanLabConfig | None = None,
    resume_from: Path | None = None,
    scope: str | None = None,
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
        "scope": scope
        or "text-only real-model plumbing smoke; not P0 completion or quality evidence",
        "resume_supported": True,
        "cost": None,
        "steps_completed": 0,
    }
    monitoring_config = monitoring or SwanLabConfig.from_env()
    report["monitoring"] = monitoring_config.describe()
    created = False
    accepted_resume = False
    telemetry: Telemetry | None = None
    try:
        if execute:
            if resume_from is None:
                output_dir.mkdir(parents=True, exist_ok=False)
                created = True
            else:
                if not output_dir.is_dir():
                    raise ValueError("resume requires an existing output directory")
        samples = smoke_samples(
            config,
            data_path,
            expected_data_version=getattr(config, "expected_data_version", None),
        )
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
            if config.modality == "text":
                adapter = OfficialTokenizerAdapter(model_dir)
                examples = prepare_examples(samples, adapter, config.max_length)
            else:
                if not config.media_root:
                    raise ValueError("image/screenshot training requires media_root")
                examples = prepare_multimodal_examples(
                    samples, model_dir, Path(config.media_root), config.max_length
                )
            report["examples"] = len(examples)
            report["max_prompt_tokens"] = max(len(e["input_ids"]) for e in examples)
        except Exception as exc:
            blockers.append(f"examples: {type(exc).__name__}: {exc}")
        blockers.extend(monitoring_config.blockers())
        report["blockers"] = blockers
        if blockers:
            raise ValueError("training preflight has blockers")
        order = list(range(len(examples)))
        random.Random(config.seed).shuffle(order)
        resume_step = 0
        if execute and resume_from is not None:
            payload = data_bytes(samples)
            data_output = output_dir / "data.jsonl"
            if (
                not data_output.is_file()
                or hashlib.sha256(data_output.read_bytes()).hexdigest() != report["data_sha256"]
            ):
                raise ValueError("resume data.jsonl is missing or does not match current data")
            resume_metadata = verify_resume_request(
                output_dir,
                resume_from,
                config=config.model_dump(),
                model_revision=report["model_revision"],
                data_sha256=report["data_sha256"],
                asset_hashes=report["asset_hashes"],
                order=order,
            )
            resume_step = int(resume_metadata["step"])
            accepted_resume = True
        if execute:
            payload = data_bytes(samples)
            data_output = output_dir / "data.jsonl"
            if resume_from is None:
                data_output.write_bytes(payload)
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
                    resume_step=resume_step,
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
    if created or accepted_resume:
        write_json(output_dir / "run.json", report)
    return report
