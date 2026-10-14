"""S0 multimodal HF/vLLM probability parity runner.

This script deliberately separates CPU-safe preparation from CUDA-only model
forward. It never starts a server and treats vLLM identity as operator-attested
input, not remote proof.
"""

from __future__ import annotations

import argparse
import asyncio
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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from eval.multimodal_parity import RawReadout, compare_rows, make_row
from scripts.assets import write_json
from scripts.s0_prepare import read_manifest
from serve.vllm_chat import VllmChatClient
from train.multimodal import PreparationCase, PreparedDecision, prepare_case
from train.trainer import runtime_requirements, verified_assets

S0_PARITY_VERSION = "s0-parity-v1"
SEED = 42
TF32_ALLOWED = False
THRESHOLD = 0.001
REQUIRED_COHORT_SIZE = 100
ALLOWED_DTYPES = {"bfloat16", "float16", "float32"}
CANONICAL_HTTP_PROFILE: Literal["canonical_http"] = "canonical_http"
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = (
    "scripts/s0_parity.py",
    "scripts/s0_prepare.py",
    "scripts/assets.py",
    "train/multimodal.py",
    "train/template.py",
    "train/readout.py",
    "train/trainer.py",
    "serve/vllm_chat.py",
    "eval/multimodal_parity.py",
)
ROW_JSON_KEYS = {
    "id_sha256",
    "input_sha256",
    "media_sha256",
    "token_ids",
    "raw_logprobs",
    "raw_probabilities",
    "joint_k_plus_u",
    "conditional_without_u",
    "p_U",
}


class S0ParityError(RuntimeError):
    """Raised when parity inputs or evidence are not admissible."""


@dataclass(frozen=True)
class AdapterEvidence:
    path: str
    config_sha256: str
    safetensors_sha256: str
    safetensors_size: int


@dataclass(frozen=True)
class ParityPaths:
    manifest: Path
    asset_root: Path
    model_dir: Path
    asset_manifest: Path
    adapter_dir: Path | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_output_error(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise S0ParityError(f"refusing to overwrite report: {path}")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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
    packages = ("torch", "transformers", "peft", "httpx", "pydantic")
    versions: dict[str, str | None] = {}
    for package in packages:
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
    for relative in REQUIRED_SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise S0ParityError(f"required source file is missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def source_identity() -> dict[str, Any]:
    return {
        "repo_root": str(REPO_ROOT),
        "git": git_revision(),
        "code_hashes": code_hashes(),
    }


def adapter_evidence(adapter_dir: Path | None) -> AdapterEvidence | None:
    if adapter_dir is None:
        return None
    config = adapter_dir / "adapter_config.json"
    safetensors = adapter_dir / "adapter_model.safetensors"
    if not config.is_file() or not safetensors.is_file():
        raise S0ParityError(
            "adapter must contain adapter_config.json and adapter_model.safetensors"
        )
    if safetensors.is_symlink() or config.is_symlink():
        raise S0ParityError("adapter evidence must not use symlinks")
    return AdapterEvidence(
        path=str(adapter_dir),
        config_sha256=sha256_file(config),
        safetensors_sha256=sha256_file(safetensors),
        safetensors_size=safetensors.stat().st_size,
    )


def load_cases(manifest: Path) -> list[PreparationCase]:
    cases = read_manifest(manifest, allow_partial=False)
    if len(cases) != REQUIRED_COHORT_SIZE:
        raise S0ParityError(f"S0 parity requires exactly {REQUIRED_COHORT_SIZE} cases")
    return cases


def build_provenance(paths: ParityPaths, *, dtype: str) -> dict[str, Any]:
    if dtype not in ALLOWED_DTYPES:
        raise S0ParityError(f"unsupported dtype: {dtype}")
    model_files = verified_assets(paths.model_dir, paths.asset_manifest)
    adapter = adapter_evidence(paths.adapter_dir)
    return {
        "version": S0_PARITY_VERSION,
        "profile": CANONICAL_HTTP_PROFILE,
        "hf_reference_requires_cuda": True,
        "vllm_compare_starts_server": False,
        "thinking": False,
        "truncation": False,
        "cohort_size": REQUIRED_COHORT_SIZE,
        "threshold_exclusive": THRESHOLD,
        "dtype": dtype,
        "execution_config": {"seed": SEED, "tf32_allowed": TF32_ALLOWED},
        "manifest": {
            "path": str(paths.manifest),
            "sha256": sha256_file(paths.manifest),
        },
        "asset_root": str(paths.asset_root),
        "model_dir": str(paths.model_dir),
        "asset_manifest": {
            "path": str(paths.asset_manifest),
            "sha256": sha256_file(paths.asset_manifest),
        },
        "model_files": model_files,
        "adapter": asdict(adapter) if adapter else None,
        "source": source_identity(),
    }


def load_official_processor(model_dir: Path) -> Any:
    transformers = importlib.import_module("transformers")
    return transformers.AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )


def row_id_hash(case_id: str) -> str:
    return sha256_json({"id": case_id})


def prepared_input_hash(prepared: PreparedDecision) -> str:
    return sha256_json(
        {
            "input_profile": prepared.summary.get("input_profile"),
            "prompt_sha256": prepared.summary.get("prompt_sha256"),
            "http_messages_sha256": prepared.summary.get("http_messages_sha256"),
            "http_media_io_kwargs_sha256": prepared.summary.get("http_media_io_kwargs_sha256"),
            "tensors": prepared.summary.get("tensors"),
            "answer_token_ids": prepared.answer_token_ids,
            "answer_token_mapping_sha256": prepared.summary.get("answer_token_mapping_sha256"),
        }
    )


def prepared_media_hash(prepared: PreparedDecision) -> list[str]:
    return [
        sha256_json(
            {
                "media_sha256": prepared.summary.get("media_sha256"),
                "canonical_media": prepared.summary.get("canonical_media"),
            }
        )
    ]


def _logprobs_by_token(
    prepared: PreparedDecision, raw_logprobs: Mapping[Any, float] | Sequence[float]
) -> list[float]:
    token_ids = list(prepared.answer_token_ids.values())
    if not isinstance(raw_logprobs, Mapping):
        values = [float(value) for value in raw_logprobs]
        if len(values) != len(token_ids):
            raise S0ParityError("raw logprob count does not match answer token count")
        for token, value in zip(token_ids, values, strict=True):
            if not math.isfinite(value) or value <= -9999.0 or value > 0.0:
                raise S0ParityError(f"inadmissible raw logprob for token {token}: {value}")
        return values
    by_token: dict[int, float] = {}
    for answer, token in prepared.answer_token_ids.items():
        if token in raw_logprobs:
            by_token[token] = float(raw_logprobs[token])
        elif str(token) in raw_logprobs:
            by_token[token] = float(raw_logprobs[str(token)])
        elif answer in raw_logprobs:
            by_token[token] = float(raw_logprobs[answer])
    missing = [token for token in token_ids if token not in by_token]
    if missing:
        raise S0ParityError(f"missing raw logprobs for answer token ids: {missing}")
    mapped_values: list[float] = []
    for token in token_ids:
        value = by_token[token]
        if not math.isfinite(value) or value <= -9999.0 or value > 0.0:
            raise S0ParityError(f"inadmissible raw logprob for token {token}: {value}")
        mapped_values.append(value)
    return mapped_values


def row_from_prepared(
    case: PreparationCase,
    prepared: PreparedDecision,
    raw_logprobs: Mapping[Any, float] | Sequence[float],
) -> RawReadout:
    token_ids = list(prepared.answer_token_ids.values())
    return make_row(
        id_sha256=row_id_hash(case.id),
        input_sha256=prepared_input_hash(prepared),
        media_sha256=prepared_media_hash(prepared),
        token_ids=token_ids,
        raw_logprobs=_logprobs_by_token(prepared, raw_logprobs),
    )


def row_to_json(row: RawReadout) -> dict[str, Any]:
    return {
        "id_sha256": row.id_sha256,
        "input_sha256": row.input_sha256,
        "media_sha256": row.media_sha256,
        "token_ids": row.token_ids,
        "raw_logprobs": row.raw_logprobs,
        "raw_probabilities": row.raw_probabilities,
        "joint_k_plus_u": row.joint,
        "conditional_without_u": row.conditional,
        "p_U": row.p_U,
    }


def _require_float_list(name: str, values: Any) -> list[float]:
    if not isinstance(values, list) or any(type(value) is not float for value in values):
        raise S0ParityError(f"saved row {name} must be a strict list of floats")
    return values


def _require_int_list(name: str, values: Any) -> list[int]:
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise S0ParityError(f"saved row {name} must be a strict list of ints")
    return values


def _require_sha_list(name: str, values: Any) -> list[str]:
    if not isinstance(values, list) or any(type(value) is not str for value in values):
        raise S0ParityError(f"saved row {name} must be a strict list of sha strings")
    return values


def row_from_json(value: Mapping[str, Any]) -> RawReadout:
    if set(value) != ROW_JSON_KEYS:
        raise S0ParityError("saved row schema keys do not match the strict parity schema")
    try:
        if type(value["id_sha256"]) is not str or type(value["input_sha256"]) is not str:
            raise S0ParityError("saved row identity hashes must be strict strings")
        if type(value["p_U"]) is not float:
            raise S0ParityError("saved row p_U must be a strict float")
        return RawReadout.model_validate(
            {
                "id_sha256": value["id_sha256"],
                "input_sha256": value["input_sha256"],
                "media_sha256": _require_sha_list("media_sha256", value["media_sha256"]),
                "token_ids": _require_int_list("token_ids", value["token_ids"]),
                "raw_logprobs": _require_float_list("raw_logprobs", value["raw_logprobs"]),
                "raw_probabilities": _require_float_list(
                    "raw_probabilities", value["raw_probabilities"]
                ),
                "joint": _require_float_list("joint_k_plus_u", value["joint_k_plus_u"]),
                "conditional": _require_float_list(
                    "conditional_without_u", value["conditional_without_u"]
                ),
                "p_U": value["p_U"],
            }
        )
    except ValidationError as exc:
        raise S0ParityError("saved row fails strict derived probability validation") from exc


def rows_from_json(values: Sequence[Mapping[str, Any]]) -> list[RawReadout]:
    return [row_from_json(value) for value in values]


def rows_to_json(rows: Sequence[RawReadout]) -> list[dict[str, Any]]:
    return [row_to_json(row) for row in rows]


def _with_passed(label: str, report: dict[str, Any]) -> dict[str, Any]:
    return {"label": label, "passed": report.get("status") == "passed", **report}


def compare_exact(
    label: str, reference: Sequence[RawReadout], candidate: Sequence[RawReadout]
) -> dict[str, Any]:
    report = compare_rows(reference, candidate, exact=True)
    return _with_passed(label, report)


def compare_threshold(
    label: str, reference: Sequence[RawReadout], candidate: Sequence[RawReadout]
) -> dict[str, Any]:
    report = compare_rows(reference, candidate, exact=False)
    return _with_passed(label, report)


def validate_prepared_rows(
    rows: Sequence[RawReadout],
    *,
    cases: Sequence[PreparationCase],
    processor: Any,
    asset_root: Path,
    prepare: Callable[[PreparationCase, Any, Path], PreparedDecision] | None = None,
) -> dict[str, Any]:
    if len(rows) != REQUIRED_COHORT_SIZE:
        raise S0ParityError(f"reference contains {len(rows)} rows; expected {REQUIRED_COHORT_SIZE}")
    prepare_fn = prepare or (
        lambda case, proc, root: prepare_case(case, proc, root, profile=CANONICAL_HTTP_PROFILE)
    )
    checked: list[RawReadout] = []
    for case, row in zip(cases, rows, strict=True):
        prepared = prepare_fn(case, processor, asset_root)
        checked.append(row_from_prepared(case, prepared, row.raw_logprobs))
    try:
        report = compare_exact("reference_rows_match_current_canonical_http_inputs", checked, rows)
    except ValueError as exc:
        raise S0ParityError(
            "reference rows do not match current canonical_http prepared inputs"
        ) from exc
    if not report["passed"]:
        raise S0ParityError("reference rows do not match current canonical_http prepared inputs")
    return report


def validate_reference_report(
    reference_report: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    cases: Sequence[PreparationCase],
    processor: Any,
    asset_root: Path,
) -> tuple[list[RawReadout], dict[str, Any]]:
    if reference_report.get("action") != "hf-reference":
        raise S0ParityError("reference report action is not hf-reference")
    if reference_report.get("status") != "reference-ready":
        raise S0ParityError("reference report is not reference-ready")
    if reference_report.get("provenance") != provenance:
        raise S0ParityError("reference provenance does not match current run")
    rows = rows_from_json(reference_report.get("rows", []))
    validation = validate_prepared_rows(
        rows, cases=cases, processor=processor, asset_root=asset_root
    )
    repeat = reference_report.get("hf_repeatability", {})
    if repeat.get("passed") is not True:
        raise S0ParityError("reference repeatability is absent or failed")
    return rows, validation


def validate_server_record(
    record: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if record.get("operator_attested") is not True:
        raise S0ParityError("server record must be operator_attested=true")
    if record.get("provenance") != provenance:
        raise S0ParityError("server record provenance does not match current run")
    if record.get("raw_logprobs_confirmed") is not True:
        raise S0ParityError("server record must confirm raw_logprobs_confirmed=true")
    if record.get("dtype") != provenance.get("dtype"):
        raise S0ParityError("server record dtype does not match provenance")
    if record.get("logprobs_mode") != "raw_logprobs":
        raise S0ParityError("server record logprobs_mode must be raw_logprobs")
    launch_command = record.get("launch_command")
    if not isinstance(launch_command, str) or not launch_command.strip():
        raise S0ParityError("server record must include launch_command")
    served_model = record.get("served_model")
    if not isinstance(served_model, str) or not served_model.strip():
        raise S0ParityError("server record must include served_model")
    return {
        "served_model": served_model,
        "operator_attested": True,
        "remote_attestation": False,
        "raw_logprobs_confirmed": True,
        "dtype": record["dtype"],
        "logprobs_mode": record["logprobs_mode"],
        "launch_command_present": True,
        "launch_command_sha256": sha256_bytes(launch_command.encode()),
        "server_startup": "not_performed_by_this_cli",
        "record_sha256": sha256_json(record),
    }


def ensure_cuda_available() -> Any:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise S0ParityError("cuda_required_before_model_load: refusing to load backbone on CPU")
    return torch


def configure_torch_runtime(torch: Any) -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = TF32_ALLOWED
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = TF32_ALLOWED


def _tensor_to_device(value: Any, *, torch: Any, dtype: Any) -> Any:
    if hasattr(value, "to") and hasattr(value, "dtype"):
        if torch.is_floating_point(value):
            return value.to(device="cuda", dtype=dtype, non_blocking=False)
        return value.to(device="cuda", non_blocking=False)
    if isinstance(value, Mapping):
        return {k: _tensor_to_device(v, torch=torch, dtype=dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensor_to_device(v, torch=torch, dtype=dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(_tensor_to_device(v, torch=torch, dtype=dtype) for v in value)
    return value


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise S0ParityError(f"unsupported dtype: {dtype_name}")


def run_hf_pass(
    *,
    cases: Sequence[PreparationCase],
    processor: Any,
    paths: ParityPaths,
    dtype_name: str,
) -> list[RawReadout]:
    torch = ensure_cuda_available()
    configure_torch_runtime(torch)
    dtype = _torch_dtype(torch, dtype_name)
    runtime_requirements()
    transformers = importlib.import_module("transformers")
    model = None
    try:
        model_cls = transformers.Gemma4ForConditionalGeneration
        model = model_cls.from_pretrained(
            str(paths.model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        if paths.adapter_dir is not None:
            peft = importlib.import_module("peft")
            model = peft.PeftModel.from_pretrained(
                model, str(paths.adapter_dir), is_trainable=False
            )
        model.requires_grad_(False)
        model.eval()
        model.to("cuda")
        rows: list[RawReadout] = []
        with torch.inference_mode():
            for case in cases:
                prepared = prepare_case(
                    case, processor, paths.asset_root, profile=CANONICAL_HTTP_PROFILE
                )
                inputs = {
                    k: _tensor_to_device(v, torch=torch, dtype=dtype)
                    for k, v in prepared.inputs.items()
                }
                output = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                logits = output.logits[0, -1].float()
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_ids = list(prepared.answer_token_ids.values())
                raw = {int(token): float(logprobs[int(token)].item()) for token in token_ids}
                rows.append(row_from_prepared(case, prepared, raw))
                del inputs, output, logits, logprobs, prepared
                torch.cuda.empty_cache()
        return rows
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def run_vllm_pass(
    *,
    cases: Sequence[PreparationCase],
    processor: Any,
    asset_root: Path,
    served_model: str,
    upstream_url: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[RawReadout], list[dict[str, Any]]]:
    rows: list[RawReadout] = []
    summaries: list[dict[str, Any]] = []
    client = VllmChatClient(
        base_url=upstream_url,
        model=served_model,
        raw_logprobs_confirmed=True,
        transport=transport,
    )
    try:
        for case in cases:
            prepared = prepare_case(case, processor, asset_root, profile=CANONICAL_HTTP_PROFILE)
            logprobs = await client.read_logprobs(prepared)
            rows.append(row_from_prepared(case, prepared, logprobs.raw_logprobs))
            summaries.append(logprobs.summary)
    finally:
        await client.aclose()
    return rows, summaries


def report_base(
    action: str,
    *,
    provenance: Mapping[str, Any] | None,
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "lite-omnijev.s0_parity.report.v1",
        "action": action,
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "environment": environment_summary(),
        "source": source_identity(),
        "provenance": provenance,
        "cost": {"new_paid_resources": 0, "gpu_started_by_this_cli": False},
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    existing_output_error(path)
    write_json(path, report)


def failure_report(
    action: str,
    *,
    report_path: Path,
    started_at: str,
    started: float,
    provenance: Mapping[str, Any] | None,
    error: BaseException,
) -> int:
    report = report_base(
        action,
        provenance=provenance,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
    )
    report.update(
        {
            "status": "blocked",
            "exit_code": 2,
            "error": {"type": type(error).__name__, "message": str(error)},
            "acceptance": {"release_or_stage_acceptance": False},
        }
    )
    write_report(report_path, report)
    return 2


def run_hf_reference(args: argparse.Namespace) -> int:
    existing_output_error(args.report)
    started = time.monotonic()
    started_at = now_iso()
    provenance: Mapping[str, Any] | None = None
    try:
        paths = args_to_paths(args)
        provenance = build_provenance(paths, dtype=args.dtype)
        cases = load_cases(paths.manifest)
        processor = load_official_processor(paths.model_dir)
        rows_a = run_hf_pass(cases=cases, processor=processor, paths=paths, dtype_name=args.dtype)
        rows_b = run_hf_pass(cases=cases, processor=processor, paths=paths, dtype_name=args.dtype)
    except S0ParityError as exc:
        return failure_report(
            "hf-reference",
            report_path=args.report,
            started_at=started_at,
            started=started,
            provenance=provenance,
            error=exc,
        )
    repeat = compare_exact("hf_reference_repeatability", rows_a, rows_b)
    status = "reference-ready" if repeat["passed"] else "failed"
    report = report_base(
        "hf-reference",
        provenance=provenance,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
    )
    report.update(
        {
            "status": status,
            "exit_code": 0 if repeat["passed"] else 2,
            "hf_repeatability": repeat,
            "rows": rows_to_json(rows_a),
            "acceptance": {
                "hf_processed_full_prepared_inputs": True,
                "canonical_http_only": True,
                "raw_logprobs_complete": repeat["passed"],
                "release_or_stage_acceptance": False,
            },
        }
    )
    write_report(args.report, report)
    return int(report["exit_code"])


def run_compare_vllm(args: argparse.Namespace) -> int:
    existing_output_error(args.report)
    started = time.monotonic()
    started_at = now_iso()
    paths = args_to_paths(args)
    provenance = build_provenance(paths, dtype=args.dtype)
    cases = load_cases(paths.manifest)
    processor = load_official_processor(paths.model_dir)
    reference_report = json.loads(args.reference.read_text())
    reference_rows, reference_validation = validate_reference_report(
        reference_report,
        provenance=provenance,
        cases=cases,
        processor=processor,
        asset_root=paths.asset_root,
    )
    server_record = json.loads(args.server_record.read_text())
    server = validate_server_record(server_record, provenance=provenance)
    rows_a, summaries_a = asyncio.run(
        run_vllm_pass(
            cases=cases,
            processor=processor,
            asset_root=paths.asset_root,
            served_model=server["served_model"],
            upstream_url=args.upstream_url,
        )
    )
    rows_b, summaries_b = asyncio.run(
        run_vllm_pass(
            cases=cases,
            processor=processor,
            asset_root=paths.asset_root,
            served_model=server["served_model"],
            upstream_url=args.upstream_url,
        )
    )
    repeat = compare_exact("vllm_repeatability", rows_a, rows_b)
    parity = compare_threshold("hf_reference_vs_vllm", reference_rows, rows_a)
    passed = bool(repeat["passed"] and parity["passed"])
    report = report_base(
        "compare-vllm",
        provenance=provenance,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
    )
    report.update(
        {
            "status": "passed" if passed else "failed",
            "exit_code": 0 if passed else 2,
            "server": server,
            "reference_validation": reference_validation,
            "vllm_repeatability": repeat,
            "parity": parity,
            "vllm_response_summaries_sha256": {
                "pass_a": sha256_json(summaries_a),
                "pass_b": sha256_json(summaries_b),
            },
            "rows": rows_to_json(rows_a),
            "acceptance": {
                "threshold_exclusive": THRESHOLD,
                "canonical_http_only": True,
                "server_remote_attestation": False,
                "release_or_stage_acceptance": False,
            },
        }
    )
    write_report(args.report, report)
    return int(report["exit_code"])


def synthetic_control_report(args: argparse.Namespace) -> int:
    existing_output_error(args.report)
    started = time.monotonic()
    started_at = now_iso()
    token_ids = [101, 202, 303]
    rows_a = [
        make_row(
            id_sha256=sha256_json({"synthetic_id": i}),
            input_sha256=sha256_json({"synthetic_input": i}),
            media_sha256=[sha256_json({"synthetic_media": i})],
            token_ids=token_ids,
            raw_logprobs=[-2.0, -3.0, -4.0],
        )
        for i in range(REQUIRED_COHORT_SIZE)
    ]
    rows_b = [
        make_row(
            id_sha256=row.id_sha256,
            input_sha256=row.input_sha256,
            media_sha256=row.media_sha256,
            token_ids=row.token_ids,
            raw_logprobs=list(row.raw_logprobs),
        )
        for row in rows_a
    ]
    repeat = compare_exact("synthetic_repeatability", rows_a, rows_b)
    parity = compare_threshold("synthetic_reference_vs_candidate", rows_a, rows_b)
    provenance = {
        "version": S0_PARITY_VERSION,
        "profile": CANONICAL_HTTP_PROFILE,
        "synthetic_non_gpu_control": True,
        "threshold_exclusive": THRESHOLD,
        "cohort_size": REQUIRED_COHORT_SIZE,
        "execution_config": {"seed": SEED, "tf32_allowed": TF32_ALLOWED},
        "source": source_identity(),
    }
    passed = bool(repeat["passed"] and parity["passed"])
    report = report_base(
        "synthetic-control",
        provenance=provenance,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - started,
    )
    report.update(
        {
            "status": "passed" if passed else "failed",
            "exit_code": 0 if passed else 2,
            "synthetic_non_gpu_acceptance": False,
            "limitation": (
                "Exercises S0 parity row math, repeatability, report refusal, "
                "and report schema only; it is not HF/vLLM or GPU evidence."
            ),
            "repeatability": repeat,
            "parity": parity,
        }
    )
    write_report(args.report, report)
    return int(report["exit_code"])


def args_to_paths(args: argparse.Namespace) -> ParityPaths:
    return ParityPaths(
        manifest=args.manifest,
        asset_root=args.asset_root,
        model_dir=args.model_dir,
        asset_manifest=args.asset_manifest,
        adapter_dir=args.adapter_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--asset-root", type=Path, required=True)
        p.add_argument("--model-dir", type=Path, required=True)
        p.add_argument("--asset-manifest", type=Path, required=True)
        p.add_argument("--adapter-dir", type=Path)
        p.add_argument("--dtype", choices=sorted(ALLOWED_DTYPES), default="bfloat16")
        p.add_argument("--report", type=Path, required=True)

    hf = sub.add_parser(
        "hf-reference", help="Two CUDA HF passes; frozen non-holdout reference report"
    )
    add_common(hf)
    hf.set_defaults(func=run_hf_reference)

    cmp = sub.add_parser(
        "compare-vllm", help="Compare native vLLM chat logprobs against a non-holdout HF reference"
    )
    add_common(cmp)
    cmp.add_argument("--reference", type=Path, required=True)
    cmp.add_argument("--server-record", type=Path, required=True)
    cmp.add_argument("--upstream-url", default="http://127.0.0.1:8001")
    cmp.set_defaults(func=run_compare_vllm)

    syn = sub.add_parser("synthetic-control", help="Write a CPU-only synthetic control-flow report")
    syn.add_argument("--report", type=Path, required=True)
    syn.set_defaults(func=synthetic_control_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except S0ParityError as exc:
        print(f"s0_parity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
