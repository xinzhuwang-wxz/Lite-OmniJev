"""Run pinned JevBench231 zero-shot public baselines for official Gemma 4 backbones."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from data.jevbench import (
    EXPECTED_PUBLIC_DECISIONS,
    REQUIRED_SOURCE_FILE_COUNT,
    JevBenchTask,
    build_input_question,
    deterministic_state_text,
    load_public_tasks,
    project_distribution_to_source,
    public_protocol_summary,
)
from data.schema import Prefix, Question, answer_tokens
from eval.jevbench import score_predictions
from scripts.assets import verify_file, write_json
from serve.gateway.app import core_question
from train.readout import read_probabilities
from train.template import check_answer_tokens, render_payload

BASELINE_VERSION = "s0-public-baselines-v1"
SEED = 42
TF32_ALLOWED = False
DEFAULT_DTYPE = "bfloat16"
ALLOWED_DTYPES = {"bfloat16", "float16", "float32"}
MODEL_SPECS: dict[str, dict[str, str]] = {
    "gemma4-e4b-it": {
        "model_type": "gemma4",
        "architecture": "Gemma4ForConditionalGeneration",
        "transformers_class": "Gemma4ForConditionalGeneration",
    },
    "gemma4-e2b-it": {
        "model_type": "gemma4",
        "architecture": "Gemma4ForConditionalGeneration",
        "transformers_class": "Gemma4ForConditionalGeneration",
    },
    "onejev-4b": {
        "model_type": "qwen3_5",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "transformers_class": "Qwen3_5ForConditionalGeneration",
    },
    "onejev-27b": {
        "model_type": "qwen3_5",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "transformers_class": "Qwen3_5ForConditionalGeneration",
    },
}
SUPPORTED_MODEL_IDS = set(MODEL_SPECS)
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CODE_FILES = (
    "scripts/s0_public_baselines.py",
    "data/jevbench.py",
    "eval/jevbench.py",
    "serve/gateway/app.py",
    "train/readout.py",
    "train/template.py",
    "scripts/assets.py",
    "scripts/s0_parity.py",
)
NORMAL_METADATA_NAMES = {
    ".gitattributes",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


class PublicBaselineError(RuntimeError):
    """Raised when baseline evidence is not admissible."""


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]: ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def __call__(
        self, text: str, *, return_tensors: str, add_special_tokens: bool = True
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PreparedQuestion:
    task: JevBenchTask
    question: Question
    payload: str
    prompt: str
    input_ids: list[int]
    answer_token_ids: dict[str, int]
    input_sha256: str
    prompt_sha256: str
    payload_sha256: str


ForwardFn = Callable[[PreparedQuestion], Mapping[int, float]]
TokenizerFactory = Callable[[Path], ChatTokenizer]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise PublicBaselineError(f"JSON root must be an object: {path}")
    return data


def git_revision() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": run("status", "--short"),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "safetensors", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cuda_summary() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return {"torch_importable": False, "available": False}
    summary: dict[str, Any] = {
        "torch_importable": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "available": bool(torch.cuda.is_available()),
    }
    if summary["available"]:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "current_device": int(index),
                "device_name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return summary


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "packages": package_versions(),
        "cuda": cuda_summary(),
    }


def code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REQUIRED_CODE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise PublicBaselineError(f"required code file missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def source_identity() -> dict[str, Any]:
    return {"repo_root": str(REPO_ROOT), "git": git_revision(), "code_hashes": code_hashes()}


def load_asset_manifest(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise PublicBaselineError("unsupported project model asset manifest")
    return data


def model_by_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    models = manifest.get("models")
    if not isinstance(models, list):
        raise PublicBaselineError("asset manifest models must be a list")
    result: dict[str, dict[str, Any]] = {}
    for raw in models:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise PublicBaselineError("asset manifest has invalid model entry")
        if raw["id"] in result:
            raise PublicBaselineError(f"duplicate model id: {raw['id']}")
        result[raw["id"]] = raw
    return result


def select_supported_model(manifest: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    if model_id not in SUPPORTED_MODEL_IDS:
        raise NotImplementedError(
            f"public JevBench baseline currently supports only {sorted(SUPPORTED_MODEL_IDS)}; "
            f"requested {model_id!r}"
        )
    models = model_by_id(manifest)
    if model_id not in models:
        raise PublicBaselineError(f"asset manifest missing requested model: {model_id}")
    return models[model_id]


def _normal_metadata_path(path: str) -> bool:
    name = Path(path).name
    if path.endswith((".safetensors", ".pt", ".bin", ".pth", ".ckpt")):
        return False
    if path.startswith(("assets/", "card-assets/")):
        return False
    return name in NORMAL_METADATA_NAMES


def _file_contract(
    file: Mapping[str, Any], *, kind: Literal["metadata", "weights"]
) -> dict[str, Any]:
    return {
        "path": file["path"],
        "size": file.get("bytes"),
        "kind": kind,
        "upstream_sha256": file.get("lfs_sha256"),
        "git_blob_sha1": file.get("git_blob_sha1"),
        "actual_sha256": file.get("actual_sha256"),
    }


def metadata_files(model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise PublicBaselineError(f"model lacks file inventory: {model.get('id')}")
    selected: list[Mapping[str, Any]] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            raise PublicBaselineError("file inventory entry must be an object")
        path = str(raw.get("path", ""))
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise PublicBaselineError(f"unsafe asset path for {model.get('id')}: {path}")
        if raw.get("required") is True and _normal_metadata_path(path):
            selected.append(raw)
    if not selected:
        raise PublicBaselineError(f"model lacks required normal metadata: {model.get('id')}")
    return selected


def weight_files(model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    files = model.get("files")
    if not isinstance(files, list):
        raise PublicBaselineError("model file inventory must be a list")
    selected = [
        cast(Mapping[str, Any], file)
        for file in files
        if isinstance(file, Mapping)
        and file.get("required") is True
        and str(file.get("path", "")).endswith(".safetensors")
    ]
    if not selected:
        raise PublicBaselineError(f"model lacks required safetensors inventory: {model.get('id')}")
    return selected


def manifest_metadata_digest(model: Mapping[str, Any]) -> str:
    keep = [
        {
            "path": file.get("path"),
            "bytes": file.get("bytes"),
            "lfs_sha256": file.get("lfs_sha256"),
            "git_blob_sha1": file.get("git_blob_sha1"),
        }
        for file in metadata_files(model)
    ]
    return sha256_json(sorted(keep, key=lambda item: str(item["path"])))


def model_dir(root: Path, model: Mapping[str, Any]) -> Path:
    repo_id = str(model["repo_id"])
    revision = str(model["revision"])
    if "/" not in repo_id or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise PublicBaselineError("unsafe repo_id in manifest")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PublicBaselineError("model revision must be an immutable sha")
    return root / repo_id.replace("/", "__") / revision


def verify_required_metadata(directory: Path, model: Mapping[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for file in metadata_files(model):
        rel = Path(str(file["path"]))
        path = directory / rel
        verified[str(rel)] = verify_file(path, _file_contract(file, kind="metadata"))
    return dict(sorted(verified.items()))


def verify_required_weights(directory: Path, model: Mapping[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for file in weight_files(model):
        rel = Path(str(file["path"]))
        path = directory / rel
        verified[str(rel)] = verify_file(path, _file_contract(file, kind="weights"))
    present = {p.name for p in directory.glob("model*.safetensors")}
    expected = {Path(str(file["path"])).name for file in weight_files(model)}
    if present != expected:
        raise PublicBaselineError("unverified or missing safetensors files in model directory")
    return dict(sorted(verified.items()))


def verify_model_architecture(directory: Path, model_id: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_id]
    config_path = directory / "config.json"
    config = load_json(config_path)
    expected_architectures = [spec["architecture"]]
    if (
        config.get("model_type") != spec["model_type"]
        or config.get("architectures") != expected_architectures
    ):
        raise PublicBaselineError("unsupported model architecture")
    auto_map = config.get("auto_map")
    if auto_map not in (None, {}):
        raise PublicBaselineError("custom-code auto_map is not allowed for public baselines")
    return {
        "config_sha256": sha256_file(config_path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "auto_map_absent": True,
        "transformers_class": spec["transformers_class"],
    }


def default_tokenizer_factory(directory: Path) -> ChatTokenizer:
    transformers = importlib.import_module("transformers")
    return cast(
        ChatTokenizer,
        transformers.AutoTokenizer.from_pretrained(
            str(directory),
            local_files_only=True,
            trust_remote_code=False,
        ),
    )


def load_public_source(source_root: Path, source_config: Path) -> tuple[JevBenchTask, ...]:
    config = load_json(source_config)
    tasks = load_public_tasks(source_root, config)
    if len(tasks) != EXPECTED_PUBLIC_DECISIONS:
        raise PublicBaselineError("public denominator changed")
    files = config.get("files")
    if not isinstance(files, list) or len(files) != REQUIRED_SOURCE_FILE_COUNT:
        raise PublicBaselineError("JevBench public source must have exactly 8 fixed files")
    return tasks


def make_question(task: JevBenchTask) -> Question:
    item = build_input_question(task)
    question, labels = core_question(task.id, item)
    if tuple(labels) != tuple(task.project_tokens):
        raise PublicBaselineError(f"gateway/project token mismatch for {task.id}")
    return question


def build_prepared_question(task: JevBenchTask, tokenizer: ChatTokenizer) -> PreparedQuestion:
    question = make_question(task)
    prefix = Prefix(state=deterministic_state_text(task.state), media=[])
    payload = render_payload(prefix, question)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": payload}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise PublicBaselineError("official chat template must return a nonempty string")
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not prompt_ids:
        raise PublicBaselineError(f"empty prompt ids for {task.id}")
    mapping = check_answer_tokens(tokenizer, question)
    for token, token_id in mapping.items():
        if tokenizer.encode(rendered + token, add_special_tokens=False) != [*prompt_ids, token_id]:
            raise PublicBaselineError(f"answer token changes at generation boundary: {token}")
    expected_tokens = (*answer_tokens(question), "U")
    if tuple(mapping) != expected_tokens:
        raise PublicBaselineError("answer token order changed")
    prompt_sha256 = sha256_bytes(rendered.encode())
    input_identity = {
        "task_id": task.id,
        "source_sha256": task.source_sha256,
        "prompt_sha256": prompt_sha256,
        "answer_token_ids": mapping,
    }
    return PreparedQuestion(
        task=task,
        question=question,
        payload=payload,
        prompt=rendered,
        input_ids=prompt_ids,
        answer_token_ids=mapping,
        input_sha256=sha256_json(input_identity),
        prompt_sha256=prompt_sha256,
        payload_sha256=sha256_bytes(payload.encode()),
    )


def prepare_questions(
    tasks: Sequence[JevBenchTask], tokenizer: ChatTokenizer
) -> list[PreparedQuestion]:
    return [build_prepared_question(task, tokenizer) for task in tasks]


def payload_has_gold_or_provenance_fields(payload: str) -> bool:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PublicBaselineError("rendered payload must stay valid JSON") from exc
    if not isinstance(decoded, dict):
        raise PublicBaselineError("rendered payload must be a JSON object")
    forbidden = {"expected", "gold", "gold_probs", "provenance", "target"}
    return bool(forbidden & set(decoded))


def render_preflight_rows(prepared: Sequence[PreparedQuestion]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.task.id,
            "primitive": item.task.primitive,
            "source_labels": list(item.task.labels),
            "project_tokens": list(item.task.project_tokens),
            "source_to_project": item.task.source_to_project,
            "project_to_source": item.task.project_to_source,
            "prompt_tokens": len(item.input_ids),
            "prompt_sha256": item.prompt_sha256,
            "payload_sha256": item.payload_sha256,
            "input_sha256": item.input_sha256,
            "answer_token_ids": item.answer_token_ids,
            "gold_or_provenance_fields_in_payload": payload_has_gold_or_provenance_fields(
                item.payload
            ),
        }
        for item in prepared
    ]


def readout_from_logprobs(
    item: PreparedQuestion, logprobs_by_id: Mapping[int, float]
) -> dict[str, Any]:
    tokens = (*answer_tokens(item.question), "U")
    by_token: dict[str, float] = {}
    for token in tokens:
        token_id = item.answer_token_ids[token]
        if token_id not in logprobs_by_id:
            raise PublicBaselineError(f"missing logprob for token id {token_id} on {item.task.id}")
        value = float(logprobs_by_id[token_id])
        if not math.isfinite(value) or value > 0.0:
            raise PublicBaselineError(f"inadmissible logprob for {item.task.id}:{token}")
        by_token[token] = value
    readout = read_probabilities(by_token, answer_tokens(item.question))
    source_probs = project_distribution_to_source(item.task, readout.probabilities)
    prediction_project = readout.prediction
    prediction_source = item.task.project_to_source[prediction_project]
    return {
        "raw_logprobs": by_token,
        "joint_k_plus_u": readout.joint_probabilities,
        "conditional_without_u_project": readout.probabilities,
        "p_U": readout.p_insufficient,
        "source_probabilities": source_probs,
        "prediction_project": prediction_project,
        "prediction_source": prediction_source,
    }


def make_record(item: PreparedQuestion, pass_index: int, forward: ForwardFn) -> dict[str, Any]:
    started = time.perf_counter()
    raw = forward(item)
    forward_wall = time.perf_counter() - started
    readout = readout_from_logprobs(item, raw)
    return {
        "pass_index": pass_index,
        "id": item.task.id,
        "primitive": item.task.primitive,
        "source_labels": list(item.task.labels),
        "project_tokens": list(item.task.project_tokens),
        "expected_present_for_scoring_only": str(item.task.expected),
        "prompt_tokens": len(item.input_ids),
        "prompt_sha256": item.prompt_sha256,
        "payload_sha256": item.payload_sha256,
        "input_sha256": item.input_sha256,
        "answer_token_ids": item.answer_token_ids,
        "forward_wall_seconds": round(forward_wall, 6),
        **readout,
    }


def append_record_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def run_prepared_pass(
    prepared: Sequence[PreparedQuestion],
    *,
    pass_index: int,
    forward: ForwardFn,
    sidecar: Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in prepared:
        record = make_record(item, pass_index, forward)
        if sidecar is not None:
            append_record_jsonl(sidecar, record)
        records.append(record)
    return records


def sidecar_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False, "records": 0, "sha256": None}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "path": str(path),
        "exists": True,
        "records": len(lines),
        "sha256": sha256_file(path),
    }


def prediction_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": str(record["id"]), "probabilities": record["source_probabilities"]}
        for record in records
    ]


def compare_repeatability(
    a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    if len(a) != len(b):
        mismatches.append({"kind": "length", "a": len(a), "b": len(b)})
    for left, right in zip(a, b, strict=False):
        if left.get("id") != right.get("id"):
            mismatches.append({"kind": "id", "a": left.get("id"), "b": right.get("id")})
            continue
        if left.get("source_probabilities") != right.get("source_probabilities"):
            mismatches.append({"kind": "probabilities", "id": left.get("id")})
        if left.get("prediction_source") != right.get("prediction_source"):
            mismatches.append({"kind": "prediction", "id": left.get("id")})
        if left.get("raw_logprobs") != right.get("raw_logprobs"):
            mismatches.append({"kind": "raw_logprobs", "id": left.get("id")})
    return {
        "passed": not mismatches,
        "records_compared": min(len(a), len(b)),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def score_records(
    tasks: Sequence[JevBenchTask], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return score_predictions(tasks, prediction_rows(records))


def ensure_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise PublicBaselineError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def report_base(action: str, started_at: str, started: float) -> dict[str, Any]:
    return {
        "schema": "lite-omnijev.s0_public_baselines.report.v1",
        "version": BASELINE_VERSION,
        "action": action,
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "environment": environment_summary(),
        "source": source_identity(),
        "cost": {"new_paid_resources": None, "gpu_started_by_this_cli": False},
    }


def model_evidence(
    manifest_path: Path,
    model_root: Path,
    model_id: str,
    *,
    require_weights: bool,
) -> tuple[dict[str, Any], Path]:
    manifest = load_asset_manifest(manifest_path)
    model = select_supported_model(manifest, model_id)
    directory = model_dir(model_root, model)
    metadata_hashes = verify_required_metadata(directory, model)
    architecture = verify_model_architecture(directory, model_id)
    weights = verify_required_weights(directory, model) if require_weights else None
    return (
        {
            "model_id": model_id,
            "repo_id": model.get("repo_id"),
            "revision": model.get("revision"),
            "license": model.get("license"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_metadata_sha256": manifest_metadata_digest(model),
            "model_dir": str(directory),
            "metadata_hashes": metadata_hashes,
            "weights_hashes": weights,
            "architecture": architecture,
            "trust_remote_code": False,
            "local_files_only": True,
        },
        directory,
    )


def adapter_evidence(
    adapter_dir: Path | None,
    expected_sha256: str | None,
    model: Mapping[str, Any],
    directory: Path,
) -> dict[str, str] | None:
    """Bind a local, standard decoder-only E4B decision LoRA before any forward."""
    if adapter_dir is None:
        if expected_sha256 is not None:
            raise PublicBaselineError("adapter SHA requires --adapter-dir")
        return None
    if model["model_id"] != "gemma4-e4b-it":
        raise PublicBaselineError("adapter supports only gemma4-e4b-it")
    if (
        expected_sha256 is None
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise PublicBaselineError("adapter requires expected lowercase weights SHA256")
    directory_adapter = adapter_dir.resolve(strict=True)
    weights = directory_adapter / "adapter_model.safetensors"
    config_path = directory_adapter / "adapter_config.json"
    if weights.is_symlink() or config_path.is_symlink():
        raise PublicBaselineError("adapter files must not be symlinks")
    if sha256_file(weights) != expected_sha256:
        raise PublicBaselineError("adapter weights hash mismatch")
    saved = load_json(config_path)
    targets = saved.get("target_modules")
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if (
        not isinstance(targets, list)
        or not targets
        or any(
            not isinstance(name, str)
            or len(name.split(".")) != 6
            or not name.startswith("model.language_model.layers.")
            or not name.split(".")[3].isdigit()
            or name.split(".")[4] not in {"self_attn", "mlp"}
            for name in targets
        )
    ):
        raise PublicBaselineError("adapter targets must be decoder projections")
    if {name.split(".")[-1] for name in targets} != projections:
        raise PublicBaselineError("adapter decoder projections incomplete")
    peft = importlib.import_module("peft")
    expected = peft.LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
        inference_mode=True,
    ).to_dict()
    metadata = {
        "auto_mapping",
        "peft_version",
        "base_model_name_or_path",
        "revision",
        "inference_mode",
    }
    if not {"peft_type", "r", "lora_alpha", "target_modules"} <= saved.keys():
        raise PublicBaselineError("adapter config incomplete")
    for key in saved.keys() | expected.keys():
        if key in metadata:
            continue
        actual = saved.get(key, expected.get(key))
        wanted = expected.get(key)
        if key == "target_modules":
            actual, wanted = set(actual), set(wanted)
        if actual != wanted:
            raise PublicBaselineError(f"adapter config mismatch: {key}")
    base = saved.get("base_model_name_or_path")
    if base not in {str(directory), str(directory.resolve()), model["repo_id"]}:
        if not isinstance(base, str) or (
            Path(base).name != model["revision"]
            or Path(base).parent.name != str(model["repo_id"]).replace("/", "__")
        ):
            raise PublicBaselineError("adapter base model mismatch")
    if saved.get("revision") not in {None, model["revision"]}:
        raise PublicBaselineError("adapter base revision mismatch")
    return {
        "path": str(directory_adapter),
        "weights_sha256": expected_sha256,
        "config_sha256": sha256_file(config_path),
        "base_model_name_or_path": str(base),
    }


def attach_public_adapter(model: Any, evidence: Mapping[str, str]) -> Any:
    path = Path(evidence["path"])
    if (
        sha256_file(path / "adapter_model.safetensors") != evidence["weights_sha256"]
        or sha256_file(path / "adapter_config.json") != evidence["config_sha256"]
    ):
        raise PublicBaselineError("adapter binding changed before load")
    peft = importlib.import_module("peft")
    adapted = peft.PeftModel.from_pretrained(
        model,
        str(path),
        local_files_only=True,
        is_trainable=False,
    )
    if any(parameter.requires_grad for _, parameter in adapted.named_parameters()):
        raise PublicBaselineError("evaluation adapter exposes trainable parameters")
    return adapted.eval()


def preflight_report(
    *,
    source_root: Path,
    source_config: Path,
    asset_manifest: Path,
    model_root: Path,
    model_id: str,
    tokenizer_factory: TokenizerFactory = default_tokenizer_factory,
    adapter_dir: Path | None = None,
    adapter_weights_sha256: str | None = None,
) -> dict[str, Any]:
    tasks = load_public_source(source_root, source_config)
    model, directory = model_evidence(asset_manifest, model_root, model_id, require_weights=False)
    adapter = adapter_evidence(adapter_dir, adapter_weights_sha256, model, directory)
    tokenizer = tokenizer_factory(directory)
    prepared = prepare_questions(tasks, tokenizer)
    rows = render_preflight_rows(prepared)
    if any(row["gold_or_provenance_fields_in_payload"] for row in rows):
        raise PublicBaselineError("gold/provenance fields leaked into at least one prompt")
    return {
        "status": "passed",
        "exit_code": 0,
        "acceptance": {
            "cpu_safe_preflight_only": True,
            "jevbench_public_231_denominator": len(tasks) == EXPECTED_PUBLIC_DECISIONS,
            "fixed_source_file_count": REQUIRED_SOURCE_FILE_COUNT,
            "model_metadata_verified": True,
            "weights_loaded": False,
            "answer_tokens_single_token_and_boundary_verified": True,
            "official_chat_template_enable_thinking_false": True,
            "gold_or_provenance_fields_in_prompt": False,
            "release_or_stage_acceptance": False,
        },
        "model": model,
        **({"adapter": adapter} if adapter else {}),
        "jevbench": public_protocol_summary(tasks),
        "source_config": {"path": str(source_config), "sha256": sha256_file(source_config)},
        "source_root": str(source_root),
        "resume": {
            "supported": False,
            "reason": "preflight is deterministic and writes one report",
        },
        "rows": rows,
    }


def ensure_cuda_available() -> Any:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise PublicBaselineError(
            "cuda_required_before_model_load: refusing to load backbone on CPU"
        )
    return torch


def torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise PublicBaselineError(f"unsupported dtype: {dtype_name}")


def configure_torch_runtime(torch: Any) -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = TF32_ALLOWED
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = TF32_ALLOWED


def prepared_model_inputs(item: PreparedQuestion, torch: Any) -> dict[str, Any]:
    input_ids = torch.tensor([item.input_ids], device="cuda", dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def real_forward_factory(
    *,
    model_dir_path: Path,
    model_id: str,
    tokenizer: ChatTokenizer,
    dtype_name: str,
    adapter: Mapping[str, str] | None = None,
) -> tuple[ForwardFn, Callable[[], None]]:
    torch = ensure_cuda_available()
    configure_torch_runtime(torch)
    dtype = torch_dtype(torch, dtype_name)
    transformers = importlib.import_module("transformers")
    class_name = MODEL_SPECS[model_id]["transformers_class"]
    if not hasattr(transformers, class_name):
        raise PublicBaselineError(f"installed Transformers lacks {class_name}")
    model_cls = getattr(transformers, class_name)
    model = model_cls.from_pretrained(
        str(model_dir_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    if adapter is not None:
        model = attach_public_adapter(model, adapter)
    model.eval()
    model.to("cuda")

    def forward(item: PreparedQuestion) -> Mapping[int, float]:
        inputs = prepared_model_inputs(item, torch)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = output.logits[0, -1].float()
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            return {
                int(token_id): float(logprobs[int(token_id)].item())
                for token_id in item.answer_token_ids.values()
            }

    def cleanup() -> None:
        nonlocal model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return forward, cleanup


def run_baseline_report(
    *,
    source_root: Path,
    source_config: Path,
    asset_manifest: Path,
    model_root: Path,
    model_id: str,
    dtype: str = DEFAULT_DTYPE,
    tokenizer_factory: TokenizerFactory = default_tokenizer_factory,
    forward_factory: Callable[[Path, ChatTokenizer, str], tuple[ForwardFn, Callable[[], None]]]
    | None = None,
    fake_forward: bool = False,
    records_sidecar: Path | None = None,
    adapter_dir: Path | None = None,
    adapter_weights_sha256: str | None = None,
) -> dict[str, Any]:
    if dtype not in ALLOWED_DTYPES:
        raise PublicBaselineError(f"unsupported dtype: {dtype}")
    tasks = load_public_source(source_root, source_config)
    model, directory = model_evidence(asset_manifest, model_root, model_id, require_weights=True)
    adapter = adapter_evidence(adapter_dir, adapter_weights_sha256, model, directory)
    tokenizer = tokenizer_factory(directory)
    prepared = prepare_questions(tasks, tokenizer)
    factory = forward_factory or (
        lambda model_path, tok, dtype_name: real_forward_factory(
            model_dir_path=model_path,
            model_id=model_id,
            tokenizer=tok,
            dtype_name=dtype_name,
            adapter=adapter,
        )
    )
    forward, cleanup = factory(directory, tokenizer, dtype)
    try:
        records_a = run_prepared_pass(
            prepared, pass_index=1, forward=forward, sidecar=records_sidecar
        )
        records_b = run_prepared_pass(
            prepared, pass_index=2, forward=forward, sidecar=records_sidecar
        )
    finally:
        cleanup()
    repeat = compare_repeatability(records_a, records_b)
    score_a = score_records(tasks, records_a)
    score_b = score_records(tasks, records_b)
    passed = bool(repeat["passed"] and len(records_a) == EXPECTED_PUBLIC_DECISIONS)
    p_u_values = [float(record["p_U"]) for record in records_a]
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 2,
        "acceptance": {
            "real_cuda_forward": not fake_forward,
            "fake_forward_control_only": fake_forward,
            "release_or_stage_acceptance": bool(passed and not fake_forward),
            "full_denominator": len(records_a) == EXPECTED_PUBLIC_DECISIONS,
            "two_pass_exact_probabilities_and_predictions": repeat["passed"],
            "weights_verified_before_load": True,
            "trust_remote_code": False,
            "local_files_only": True,
            "use_safetensors": True,
            "official_chat_template_enable_thinking_false": True,
            "no_holdout": True,
        },
        "model": model,
        **({"adapter": adapter} if adapter else {}),
        "precision": {
            "dtype": dtype,
            "seed": SEED,
            "tf32_allowed": TF32_ALLOWED,
            "determinism": (
                "two complete passes require exact raw logprob/probability/prediction match"
            ),
        },
        "jevbench": public_protocol_summary(tasks),
        "source_config": {"path": str(source_config), "sha256": sha256_file(source_config)},
        "source_root": str(source_root),
        "resume": {
            "supported": False,
            "reason": (
                "minimal baseline runner saves completed pass records "
                "in the report; rerun starts fresh"
            ),
            "records_saved": len(records_a),
            "sidecar": sidecar_evidence(records_sidecar),
        },
        "repeatability": repeat,
        "score_pass_1": score_a,
        "score_pass_2": score_b,
        "p_U_summary": {
            "min": min(p_u_values) if p_u_values else None,
            "max": max(p_u_values) if p_u_values else None,
            "mean": math.fsum(p_u_values) / len(p_u_values) if p_u_values else None,
        },
        "records": records_a,
        "records_pass_2_sha256": sha256_json(records_b),
        "records_sidecar": sidecar_evidence(records_sidecar),
    }


def command_report(action: str, args: argparse.Namespace) -> int:
    ensure_output(args.report)
    started = time.perf_counter()
    started_at = now_iso()
    records_sidecar = getattr(args, "records_sidecar", None)
    if action == "run" and records_sidecar is None:
        records_sidecar = args.report.with_name(args.report.name + ".records.jsonl")
    if records_sidecar is not None:
        ensure_output(records_sidecar)
    try:
        if action == "preflight":
            body = preflight_report(
                source_root=args.source_root,
                source_config=args.source_config,
                asset_manifest=args.asset_manifest,
                model_root=args.model_root,
                model_id=args.model_id,
                adapter_dir=args.adapter_dir,
                adapter_weights_sha256=args.adapter_weights_sha256,
            )
        elif action == "run":
            body = run_baseline_report(
                source_root=args.source_root,
                source_config=args.source_config,
                asset_manifest=args.asset_manifest,
                model_root=args.model_root,
                model_id=args.model_id,
                adapter_dir=args.adapter_dir,
                adapter_weights_sha256=args.adapter_weights_sha256,
                dtype=args.dtype,
                records_sidecar=records_sidecar,
            )
        else:
            raise PublicBaselineError(f"unknown action: {action}")
    except NotImplementedError as exc:
        base = report_base(action, started_at, started)
        base.update(
            {
                "status": "notimplemented",
                "exit_code": 2,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "acceptance": {"release_or_stage_acceptance": False},
                "records_sidecar": sidecar_evidence(records_sidecar),
            }
        )
        write_json(args.report, base)
        return 2
    except Exception as exc:
        base = report_base(action, started_at, started)
        base.update(
            {
                "status": "blocked",
                "exit_code": 2,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "acceptance": {"release_or_stage_acceptance": False},
                "records_sidecar": sidecar_evidence(records_sidecar),
            }
        )
        write_json(args.report, base)
        return 2
    report = report_base(action, started_at, started)
    report.update(body)
    report["command"] = {"argv": sys.argv, "cwd": str(Path.cwd())}
    write_json(args.report, report)
    return int(report["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(child: argparse.ArgumentParser) -> None:
        child.add_argument("--source-root", type=Path, required=True)
        child.add_argument(
            "--source-config", type=Path, default=Path("configs/jevbench-public-assets.json")
        )
        child.add_argument(
            "--asset-manifest", type=Path, default=Path("configs/project-model-assets.json")
        )
        child.add_argument("--model-root", type=Path, required=True)
        child.add_argument("--model-id", required=True)
        child.add_argument("--report", type=Path, required=True)
        child.add_argument("--adapter-dir", type=Path)
        child.add_argument("--adapter-weights-sha256")

    preflight = sub.add_parser("preflight")
    add_common(preflight)
    run = sub.add_parser("run")
    add_common(run)
    run.add_argument("--dtype", choices=sorted(ALLOWED_DTYPES), default=DEFAULT_DTYPE)
    run.add_argument("--records-sidecar", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return command_report(args.action, args)


if __name__ == "__main__":
    raise SystemExit(main())
