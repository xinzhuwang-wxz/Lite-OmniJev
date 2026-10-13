"""Audited conversion for Valen nested training checkpoints."""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from train.checkpoint import file_sha256
from train.teacher_head import (
    ISOLATED_CONVERSION_TIMEOUT_SECONDS,
    SAFETENSORS_HEADER_RESERVE_BYTES,
    IsolationError,
    _assert_socket_denied,
    _copy_regular_nofollow,
    _drop_to_nobody,
    _scrubbed_env,
    _set_no_new_privs,
    install_network_deny_seccomp,
)

SCHEMA_VERSION = 1
SOURCE_REPO = "Valen-Team/Valen-Preview-0923"
SOURCE_REVISION = "30096aaabb2689614c83a79f64f38c865c2ce1c1"
SOURCE_CODE_REPO = "Liuziyu77/Valen"
SOURCE_CODE_REVISION = "42a6a10e127e9248a6d7824299551b78af9fcb97"
BASE_REPO = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CHECKPOINT_SHA256 = "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012"
SOURCE_FILE_SHA256S = {
    "configs/train/qwen/rlcd_vision_top.json": (
        "4c7aa7018a67c86ce9492862a604de9bff7cea0ebcceeb15dae86a2dc3c5d54b"
    ),
    "valen/modeling/qwen/builder.py": (
        "34cd6bb85e9283f2aa026b0f29655ddd8e46e419451e09eb177ae29b0878dfb7"
    ),
    "valen/modeling/qwen/heads.py": (
        "63b5970a8bfa121eff94d50ffd35e1fa8387f3af5abccc5f5b75381fb084fc34"
    ),
    "valen/training/checkpoint.py": (
        "b3739da6db4202414da637b821ad99c97c13e618680b8dcc64893c65296ac438"
    ),
}

MAX_STAGE_DIAGNOSTICS_BYTES = 64 * 1024
ALLOWED_STAGE_NAMES = {
    "source_hash",
    "load_weights_only",
    "audit_tensors",
    "save_safetensors",
    "reload_safetensors",
    "roundtrip_equal",
    "output_hash",
}
ALLOWED_STAGE_NUMERIC_FIELDS = {
    "started_seconds",
    "duration_seconds",
    "elapsed_seconds",
    "checkpoint_size",
    "tensor_count",
    "output_size",
}

THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}


def _valen_scrubbed_env() -> dict[str, str]:
    env = _scrubbed_env()
    env.update(THREAD_LIMIT_ENV)
    return env


def _configure_torch_threads(torch: Any) -> dict[str, int | str]:
    for key, value in THREAD_LIMIT_ENV.items():
        os.environ[key] = value
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    interop: int | str = "unknown"
    try:
        interop = int(torch.get_num_interop_threads())
    except RuntimeError:
        pass
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": interop,
    }


class StageDiagnostics:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.started = time.monotonic()
        self.stages: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.context: dict[str, Any] = {}

    def update_context(self, **items: Any) -> None:
        self.context.update(items)
        self._write("running")

    def start(self, name: str, **items: Any) -> None:
        now = time.monotonic()
        self.current = {"name": name, "started_seconds": round(now - self.started, 6), **items}
        self._write("running")

    def finish(self, name: str, **items: Any) -> None:
        now = time.monotonic()
        current = self.current if self.current and self.current.get("name") == name else None
        started = (
            float(current.get("started_seconds", now - self.started))
            if current
            else now - self.started
        )
        entry = {
            "name": name,
            "started_seconds": round(started, 6),
            "duration_seconds": round((now - self.started) - started, 6),
            **(current or {}),
            **items,
        }
        self.stages.append(entry)
        self.current = None
        self._write("running")

    def complete(self) -> None:
        self._write("complete")

    def snapshot(self, status: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - self.started, 6),
            "stages": self.stages,
            "context": self.context,
        }
        if self.current is not None:
            current = dict(self.current)
            current["elapsed_seconds"] = round(
                time.monotonic() - self.started - float(current["started_seconds"]), 6
            )
            payload["current_stage"] = current
        return payload

    def _write(self, status: str) -> None:
        if self.path is None:
            return
        payload = self.snapshot(status)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        temporary.replace(self.path)


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return cast(int | float, value)


def _read_regular_nofollow_bytes(path: Path, *, max_bytes: int) -> bytes | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            return None
        if status.st_size < 0 or status.st_size > max_bytes:
            return None
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            return None
        return payload
    except OSError:
        return None
    finally:
        os.close(fd)


def _sanitize_stage_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    if not isinstance(name, str) or name not in ALLOWED_STAGE_NAMES:
        return None
    sanitized: dict[str, Any] = {"name": name}
    for key in ALLOWED_STAGE_NUMERIC_FIELDS:
        number = _safe_number(value.get(key))
        if number is not None:
            sanitized[key] = number
    return sanitized


def _sanitize_stage_diagnostics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != SCHEMA_VERSION:
        return None
    status_value = value.get("status")
    if status_value not in {"running", "complete"}:
        return None
    stages_value = value.get("stages")
    if not isinstance(stages_value, list):
        return None
    stages: list[dict[str, Any]] = []
    for stage in stages_value:
        sanitized_stage = _sanitize_stage_entry(stage)
        if sanitized_stage is None:
            return None
        stages.append(sanitized_stage)
    sanitized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status_value,
        "stages": stages,
    }
    elapsed = _safe_number(value.get("elapsed_seconds"))
    if elapsed is not None:
        sanitized["elapsed_seconds"] = elapsed
    current_stage = value.get("current_stage")
    if current_stage is not None:
        sanitized_current = _sanitize_stage_entry(current_stage)
        if sanitized_current is None:
            return None
        sanitized["current_stage"] = sanitized_current
    context = value.get("context")
    if isinstance(context, dict):
        sanitized_context: dict[str, Any] = {}
        for key in ("torch_num_threads", "torch_num_interop_threads"):
            number = _safe_number(context.get(key))
            if number is not None:
                sanitized_context[key] = number
        torch_load_mmap = context.get("torch_load_mmap")
        if isinstance(torch_load_mmap, bool):
            sanitized_context["torch_load_mmap"] = torch_load_mmap
        if sanitized_context:
            sanitized["context"] = sanitized_context
    return sanitized


def _read_stage_diagnostics(path: Path) -> dict[str, Any] | None:
    payload = _read_regular_nofollow_bytes(path, max_bytes=MAX_STAGE_DIAGNOSTICS_BYTES)
    if payload is None:
        return None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _sanitize_stage_diagnostics(value)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path.name}")
    return value


def _torch_dtype_from_safetensors(dtype: str) -> str:
    mapping = {
        "BF16": "torch.bfloat16",
        "F32": "torch.float32",
        "F16": "torch.float16",
        "I64": "torch.int64",
        "I32": "torch.int32",
        "I16": "torch.int16",
        "I8": "torch.int8",
        "U8": "torch.uint8",
        "BOOL": "torch.bool",
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return mapping[dtype]


def safetensors_shape_map(base_dir: Path) -> dict[str, dict[str, Any]]:
    index = _json(base_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("base model lacks a safetensors weight_map")
    safe_open = importlib.import_module("safetensors").safe_open
    keys_by_file: dict[str, list[str]] = {}
    for key, filename in sorted(weight_map.items()):
        if not isinstance(key, str) or not isinstance(filename, str):
            raise ValueError("invalid base model weight_map entry")
        keys_by_file.setdefault(filename, []).append(key)
    result: dict[str, dict[str, Any]] = {}
    for filename, keys in sorted(keys_by_file.items()):
        with safe_open(str(base_dir / filename), framework="pt", device="cpu") as handle:
            for key in keys:
                slice_ = handle.get_slice(key)
                result[key] = {
                    "shape": list(slice_.get_shape()),
                    "dtype": _torch_dtype_from_safetensors(slice_.get_dtype()),
                }
    return result


def _dtype_size(dtype: str) -> int:
    sizes = {
        "torch.float64": 8,
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int64": 8,
        "torch.int32": 4,
        "torch.int16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }
    if dtype not in sizes:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    return sizes[dtype]


def tensor_payload_bytes(tensors: Mapping[str, Mapping[str, Any]]) -> int:
    total = 0
    for name, spec in tensors.items():
        shape = spec.get("shape")
        dtype = spec.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise ValueError(f"invalid tensor spec: {name}")
        numel = 1
        for dim in shape:
            if not isinstance(dim, int) or dim < 0:
                raise ValueError(f"invalid tensor shape: {name}")
            numel *= dim
        total += numel * _dtype_size(dtype)
    return total


def _group_for_key(key: str) -> str:
    if key.startswith("head."):
        return "head"
    if "lora_" in key:
        return "lora"
    if ".visual.merger." in key:
        return "merger"
    if ".visual.blocks." in key:
        return "vision"
    raise ValueError(f"unclassified Valen tensor: {key}")


def _summarize(tensors: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key, spec in tensors.items():
        group = _group_for_key(key)
        entry = summary.setdefault(group, {"count": 0, "numel": 0, "dtypes": {}})
        entry["count"] += 1
        numel = 1
        for dim in spec["shape"]:
            numel *= int(dim)
        entry["numel"] += numel
        dtypes = entry["dtypes"]
        dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
    return summary


def derive_expected_tensors(
    *,
    base_config: dict[str, Any],
    valen_config: dict[str, Any],
    metadata: dict[str, Any],
    base_shapes: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if valen_config.get("stage") != "vision_top" or valen_config.get("method") != "rlcd":
        raise ValueError("Valen preview spec supports only rlcd vision_top")
    if valen_config.get("lora_rank") != 32 or valen_config.get("projection_dim") != 256:
        raise ValueError("unexpected Valen preview LoRA/head dimensions")
    text = base_config.get("text_config")
    vision = base_config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("base config lacks text/vision sections")
    hidden = int(text["hidden_size"])
    vision_depth = int(vision["depth"])
    rank = int(valen_config["lora_rank"])
    projection = int(valen_config["projection_dim"])
    tensors: dict[str, dict[str, Any]] = {}

    for block in range(vision_depth - 4, vision_depth):
        prefix = f"model.visual.blocks.{block}."
        for key, spec in base_shapes.items():
            if key.startswith(prefix):
                tensors["backbone." + key.removeprefix("model.")] = {
                    "shape": list(spec["shape"]),
                    "dtype": spec["dtype"],
                }
    for key, spec in base_shapes.items():
        if key.startswith("model.visual.merger."):
            tensors["backbone." + key.removeprefix("model.")] = {
                "shape": list(spec["shape"]),
                "dtype": spec["dtype"],
            }
    for key, spec in base_shapes.items():
        if not (
            key.startswith("model.language_model.layers.")
            and key.endswith(".weight")
            and ".conv1d." not in key
            and len(spec["shape"]) == 2
        ):
            continue
        out_features, in_features = [int(dim) for dim in spec["shape"]]
        stem = "backbone.language_model.base_model.model." + key.removeprefix(
            "model.language_model."
        ).removesuffix(".weight")
        tensors[f"{stem}.lora_A.default.weight"] = {
            "shape": [rank, in_features],
            "dtype": "torch.float32",
        }
        tensors[f"{stem}.lora_B.default.weight"] = {
            "shape": [out_features, rank],
            "dtype": "torch.float32",
        }
    tensors["head.decision.weight"] = {"shape": [projection, hidden], "dtype": "torch.float32"}
    tensors["head.candidate.weight"] = {"shape": [projection, hidden], "dtype": "torch.float32"}

    summary = _summarize(tensors)
    expected_counts = metadata.get("trainable_parameters")
    if not isinstance(expected_counts, dict):
        raise ValueError("training metadata lacks trainable parameter counts")
    for group in ("vision", "merger", "lora", "head"):
        if int(summary.get(group, {}).get("numel", 0)) != expected_counts.get(group):
            raise ValueError(f"derived {group} tensor count does not match metadata")
    return dict(sorted(tensors.items()))


def derive_spec(checkpoint_dir: Path, base_dir: Path) -> dict[str, Any]:
    config = _json(checkpoint_dir / "config.json")
    metadata = _json(checkpoint_dir / "training_metadata.json")
    base_config = _json(base_dir / "config.json")
    tensors = derive_expected_tensors(
        base_config=base_config,
        valen_config=config,
        metadata=metadata,
        base_shapes=safetensors_shape_map(base_dir),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_code_repo": SOURCE_CODE_REPO,
        "source_code_revision": SOURCE_CODE_REVISION,
        "source_file_sha256s": SOURCE_FILE_SHA256S,
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": file_sha256(checkpoint_dir / "config.json"),
        "training_metadata_sha256": file_sha256(checkpoint_dir / "training_metadata.json"),
        "base_config_sha256": file_sha256(base_dir / "config.json"),
        "tensor_count": len(tensors),
        "tensor_payload_bytes": tensor_payload_bytes(tensors),
        "groups": _summarize(tensors),
        "tensors": tensors,
    }


def load_expected_spec(path: Path) -> dict[str, dict[str, Any]]:
    value = _json(path)
    tensors = value.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Valen spec lacks tensors")
    expected: dict[str, dict[str, Any]] = {}
    for key, spec in tensors.items():
        if not isinstance(key, str) or not isinstance(spec, dict):
            raise ValueError("invalid Valen tensor spec")
        shape, dtype = spec.get("shape"), spec.get("dtype")
        valid_shape = isinstance(shape, list) and all(
            isinstance(dim, int) and dim >= 0 for dim in shape
        )
        if not valid_shape:
            raise ValueError(f"invalid Valen tensor shape: {key}")
        if not isinstance(dtype, str):
            raise ValueError(f"invalid Valen tensor dtype: {key}")
        expected[key] = {"shape": shape, "dtype": dtype}
    return expected


def _audit_weights(weights: Any, expected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    if not isinstance(weights, dict):
        raise ValueError("Valen checkpoint weights must be a dict")
    if set(weights) != set(expected):
        raise ValueError("Valen checkpoint tensor keys mismatch")
    report: dict[str, Any] = {}
    for key in sorted(weights):
        tensor = weights[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Valen checkpoint contains non-tensor weight: {key}")
        if tensor.layout != torch.strided:
            raise ValueError(f"Valen checkpoint tensor layout is unsupported: {key}")
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        if shape != expected[key]["shape"] or dtype != expected[key]["dtype"]:
            raise ValueError(f"Valen checkpoint tensor spec mismatch: {key}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Valen checkpoint tensor has nonfinite values: {key}")
        report[key] = {
            "shape": shape,
            "dtype": dtype,
            "numel": int(tensor.numel()),
        }
    return report


def convert_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    expected_spec: Mapping[str, Mapping[str, Any]],
    diagnostics_output: Path | None = None,
) -> dict[str, Any]:
    diagnostics = StageDiagnostics(diagnostics_output)
    if checkpoint.name != "checkpoint.pt":
        raise ValueError("Valen source must be checkpoint.pt")
    if output.suffix != ".safetensors":
        raise ValueError("Valen output must be safetensors")
    if expected_size is not None and checkpoint.stat().st_size != expected_size:
        raise ValueError("Valen checkpoint size mismatch")
    diagnostics.start("source_hash", checkpoint_size=checkpoint.stat().st_size)
    actual_sha256 = file_sha256(checkpoint)
    diagnostics.finish("source_hash")
    if actual_sha256 != expected_sha256:
        raise ValueError("Valen checkpoint SHA256 mismatch")
    torch = importlib.import_module("torch")
    diagnostics.update_context(**_configure_torch_threads(torch), torch_load_mmap=True)
    diagnostics.start("load_weights_only")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    diagnostics.finish("load_weights_only")
    if not isinstance(payload, dict) or "weights" not in payload:
        raise ValueError("Valen checkpoint lacks nested weights")
    diagnostics.start("audit_tensors")
    weights = payload["weights"]
    tensor_report = _audit_weights(weights, expected_spec)
    diagnostics.finish("audit_tensors", tensor_count=len(tensor_report))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    safetensors = importlib.import_module("safetensors.torch")
    diagnostics.start("save_safetensors")
    safetensors.save_file(
        weights,
        str(temporary),
        metadata={
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_sha256": actual_sha256,
            "schema_version": str(SCHEMA_VERSION),
            "payload": "weights",
        },
    )
    diagnostics.finish("save_safetensors", output_size=temporary.stat().st_size)
    diagnostics.start("reload_safetensors")
    converted = safetensors.load_file(str(temporary), device="cpu")
    diagnostics.finish("reload_safetensors")
    if set(converted) != set(tensor_report):
        temporary.unlink(missing_ok=True)
        raise ValueError("Valen safetensors round-trip keys mismatch")
    diagnostics.start("roundtrip_equal")
    for key, tensor in weights.items():
        converted_tensor = converted[key]
        if list(converted_tensor.shape) != tensor_report[key]["shape"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Valen safetensors round-trip shape mismatch: {key}")
        if str(converted_tensor.dtype) != tensor_report[key]["dtype"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Valen safetensors round-trip dtype mismatch: {key}")
        if not bool(torch.equal(tensor, converted_tensor)):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Valen tensor round-trip mismatch: {key}")
    diagnostics.finish("roundtrip_equal", tensor_count=len(tensor_report))
    temporary.replace(output)
    diagnostics.start("output_hash", output_size=output.stat().st_size)
    output_sha256 = file_sha256(output)
    diagnostics.finish("output_hash")
    diagnostics.complete()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "converted",
        "converted_at": datetime.now(UTC).isoformat(),
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_path": str(checkpoint),
        "source_size": checkpoint.stat().st_size,
        "source_sha256": actual_sha256,
        "output_path": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": output_sha256,
        "stage_timings_seconds": diagnostics.stages,
        "runtime": diagnostics.context,
        "top_level_keys": sorted(str(key) for key in payload),
        "tensor_count": len(tensor_report),
        "groups": _summarize(tensor_report),
        "tensors": tensor_report,
    }


def _validate_safetensors_header(path: Path, expected: Mapping[str, Mapping[str, Any]]) -> None:
    safe_open = importlib.import_module("safetensors").safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(expected):
            raise IsolationError("copied Valen safetensors keys mismatch")
        for key, spec in expected.items():
            slice_ = handle.get_slice(key)
            if list(slice_.get_shape()) != spec["shape"]:
                raise IsolationError("copied Valen safetensors shape mismatch")
            if _torch_dtype_from_safetensors(slice_.get_dtype()) != spec["dtype"]:
                raise IsolationError("copied Valen safetensors dtype mismatch")


def _empty_staging_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=output.name + ".",
        suffix=".staging",
        dir=output.parent,
    )
    os.close(fd)
    staging = Path(name)
    staging.unlink()
    return staging


def child_chrooted_convert(argv: list[str], *, jail_root: Path, host_canary: str) -> dict[str, Any]:
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise IsolationError("Valen chroot conversion requires Linux root")
    for key, value in THREAD_LIMIT_ENV.items():
        os.environ[key] = value
    importlib.import_module("torch")
    importlib.import_module("torch.utils.serialization")
    importlib.import_module("numpy.ctypeslib")
    importlib.import_module("safetensors.torch")
    install_network_deny_seccomp()
    os.chdir(jail_root)
    os.chroot(jail_root)
    os.chdir("/")
    _set_no_new_privs()
    _drop_to_nobody()
    secret_denied = _host_canary_denied(host_canary)
    socket_denied = _assert_socket_denied()
    kwargs = parse_conversion_args(argv)
    report = convert_checkpoint(**cast(Any, kwargs))
    report["isolation"] = {
        "mode": "chroot+dropuid+seccomp",
        "filesystem_boundary": "chroot",
        "uid": os.getuid(),
        "gid": os.getgid(),
        "network_denied": socket_denied,
        "host_canary_denied": secret_denied,
        "no_new_privs": True,
    }
    report["expected_spec_provided"] = True
    return report


def _host_canary_denied(host_canary: str) -> bool:
    try:
        Path(host_canary).read_bytes()
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return True
    raise IsolationError("host canary remained readable after Valen chroot")


def parse_conversion_args(argv: list[str]) -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-spec", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path)
    args = parser.parse_args(argv)
    return {
        "checkpoint": args.checkpoint,
        "output": args.output,
        "expected_sha256": args.expected_sha256,
        "expected_size": args.expected_size,
        "expected_spec": load_expected_spec(args.expected_spec),
        "diagnostics_output": args.diagnostics_output,
    }


def _jail_args(args: list[str]) -> tuple[list[str], Path, Path, Path]:
    parsed = parse_conversion_args(args)
    checkpoint = cast(Path, parsed["checkpoint"])
    output = cast(Path, parsed["output"])
    spec_path: Path | None = None
    for index, item in enumerate(args):
        if item == "--expected-spec":
            spec_path = Path(args[index + 1])
            break
    if spec_path is None:
        raise IsolationError("Valen conversion requires expected spec")
    jail_args = args.copy()
    replacements = {
        "--checkpoint": "/input/checkpoint.pt",
        "--output": "/output/weights.safetensors",
        "--expected-spec": "/input/expected-spec.json",
        "--diagnostics-output": "/output/stage-diagnostics.json",
    }
    if "--diagnostics-output" not in jail_args:
        jail_args.extend(["--diagnostics-output", "/output/stage-diagnostics.json"])
    for index, item in enumerate(jail_args):
        if item in replacements:
            jail_args[index + 1] = replacements[item]
    return jail_args, checkpoint, output, spec_path


def isolated_convert(args: list[str]) -> dict[str, Any]:
    jail_args, checkpoint, output, spec_path = _jail_args(args)
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise IsolationError("Valen isolated conversion requires Linux root chroot support")
    expected = load_expected_spec(spec_path)
    max_output_bytes = tensor_payload_bytes(expected) + SAFETENSORS_HEADER_RESERVE_BYTES * 4
    evidence = {
        "expected_spec_sha256": file_sha256(spec_path),
        "config_sha256": file_sha256(checkpoint.parent / "config.json"),
        "training_metadata_sha256": file_sha256(checkpoint.parent / "training_metadata.json"),
    }
    with tempfile.TemporaryDirectory(prefix="valen-checkpoint-jail-") as temporary:
        jail = Path(temporary)
        input_dir, output_dir = jail / "input", jail / "output"
        jail.chmod(0o755)
        input_dir.mkdir()
        output_dir.mkdir()
        input_dir.chmod(0o755)
        output_dir.chmod(0o777)
        shutil.copy2(checkpoint, input_dir / "checkpoint.pt")
        shutil.copy2(spec_path, input_dir / "expected-spec.json")
        host_canary = jail.parent / f"{jail.name}.host-canary"
        host_canary.write_text("secret-canary")
        command = [
            sys.executable,
            "-m",
            "scripts.valen_checkpoint",
            "_child-convert",
            "--jail-root",
            str(jail),
            "--host-canary",
            str(host_canary),
            *jail_args,
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    env=_valen_scrubbed_env(),
                    timeout=ISOLATED_CONVERSION_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                diagnostics = _read_stage_diagnostics(output_dir / "stage-diagnostics.json")
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "exit_code": 1,
                    "blocker": "isolation_timeout",
                    "timeout_seconds": ISOLATED_CONVERSION_TIMEOUT_SECONDS,
                    "stage_diagnostics": diagnostics,
                    **evidence,
                }
            if completed.returncode != 0:
                raise IsolationError("Valen isolated conversion child failed")
            report = dict(json.loads(completed.stdout))
            if "tensors" in report:
                report.pop("tensors")
                report["tensor_manifest_redacted"] = True
            diagnostics = _read_stage_diagnostics(output_dir / "stage-diagnostics.json")
            if diagnostics is not None:
                report["stage_diagnostics"] = diagnostics
            converted = output_dir / "weights.safetensors"
            staging = _empty_staging_path(output)
            try:
                _copy_regular_nofollow(converted, staging, max_bytes=max_output_bytes)
                staging_sha256 = file_sha256(staging)
                if staging_sha256 != report.get("output_sha256"):
                    raise IsolationError("copied Valen safetensors hash mismatch")
                _validate_safetensors_header(staging, expected)
                staging.replace(output)
            except Exception:
                staging.unlink(missing_ok=True)
                staging.with_suffix(staging.suffix + ".tmp").unlink(missing_ok=True)
                raise
            report["output_path"] = str(output)
            report["output_size"] = output.stat().st_size
            report["output_sha256"] = file_sha256(output)
            report.update(evidence)
            return report
        finally:
            host_canary.unlink(missing_ok=True)
