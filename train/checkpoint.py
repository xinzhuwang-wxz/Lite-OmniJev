"""Safe checkpoint serialization for the bounded P0 smoke run."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, cast

from scripts.assets import write_json

SCHEMA_VERSION = 1


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step-{step:04d}"


def _tensor_io() -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    safetensors = importlib.import_module("safetensors.torch")
    return torch, (safetensors.load_file, safetensors.save_file)


def _json_safe(value: Any, tensors: dict[str, Any], prefix: str) -> Any:
    torch, _ = _tensor_io()
    if isinstance(value, torch.Tensor):
        key = prefix.replace("/", "__")
        tensors[key] = value.detach().cpu()
        return {"__tensor__": key}
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, tensors, f"{prefix}/{key}") for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, tensors, f"{prefix}/{index}") for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _restore_json_safe(value: Any, tensors: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__tensor__"}:
            return tensors[value["__tensor__"]]
        restored: dict[Any, Any] = {}
        for key, item in value.items():
            restored[int(key) if key.isdigit() else key] = _restore_json_safe(item, tensors)
        return restored
    if isinstance(value, list):
        return [_restore_json_safe(item, tensors) for item in value]
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model: Any, tensors: dict[str, Any]) -> None:
    parameters = {name: parameter for name, parameter in model.named_parameters()}
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(tensors) != expected:
        raise ValueError("checkpoint trainable parameter set does not match model")
    for name, tensor in tensors.items():
        parameter = parameters[name]
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    config: dict[str, Any],
    model_revision: str,
    data_sha256: str,
    asset_hashes: dict[str, str],
    order: list[int],
    losses: list[float],
    model: Any,
    optimizer: Any,
) -> Path:
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    destination = checkpoint_dir(output_dir, step)
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    tmp = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch, (_, save_file) = _tensor_io()
    try:
        model_tensors = _trainable_state(model)
        save_file(model_tensors, str(tmp / "model.safetensors"))
        optimizer_tensors: dict[str, Any] = {}
        optimizer_meta = _json_safe(optimizer.state_dict(), optimizer_tensors, "optimizer")
        save_file(optimizer_tensors, str(tmp / "optimizer.safetensors"))
        rng_tensors = {"torch_cpu_rng": torch.random.get_rng_state().cpu()}
        cuda_rng = []
        if torch.cuda.is_available():
            cuda_rng = [tensor.cpu() for tensor in torch.cuda.get_rng_state_all()]
            rng_tensors.update(
                {f"torch_cuda_rng_{index}": tensor for index, tensor in enumerate(cuda_rng)}
            )
        save_file(rng_tensors, str(tmp / "rng.safetensors"))
        write_json(
            tmp / "checkpoint.json",
            {
                "schema_version": SCHEMA_VERSION,
                "step": step,
                "config_sha256": stable_hash(config),
                "config": config,
                "model_revision": model_revision,
                "data_sha256": data_sha256,
                "asset_hashes_sha256": stable_hash(asset_hashes),
                "asset_hashes": asset_hashes,
                "order_sha256": stable_hash(order),
                "order": order,
                "losses": losses,
                "optimizer": optimizer_meta,
                "python_random_state": _json_safe(random.getstate(), {}, "python_random"),
                "cuda_rng_count": len(cuda_rng),
            },
        )
        os.replace(tmp, destination)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return destination


def load_checkpoint(path: Path) -> dict[str, Any]:
    torch, (load_file, _) = _tensor_io()
    metadata = json.loads((path / "checkpoint.json").read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    model_tensors = load_file(str(path / "model.safetensors"))
    optimizer_tensors = load_file(str(path / "optimizer.safetensors"))
    rng_tensors = load_file(str(path / "rng.safetensors"))
    metadata["model_tensors"] = model_tensors
    metadata["optimizer_state"] = _restore_json_safe(metadata["optimizer"], optimizer_tensors)
    metadata["torch_cpu_rng"] = rng_tensors["torch_cpu_rng"]
    cuda_rng_count = int(metadata.get("cuda_rng_count", 0))
    metadata["torch_cuda_rng"] = [
        rng_tensors[f"torch_cuda_rng_{index}"] for index in range(cuda_rng_count)
    ]
    metadata["_torch"] = torch
    return cast(dict[str, Any], metadata)


def verify_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config: dict[str, Any],
    model_revision: str,
    data_sha256: str,
    asset_hashes: dict[str, str],
    order: list[int],
) -> int:
    if checkpoint["config_sha256"] != stable_hash(config) or checkpoint["config"] != config:
        raise ValueError("checkpoint config does not match current run")
    if checkpoint["model_revision"] != model_revision:
        raise ValueError("checkpoint model revision does not match current run")
    if checkpoint["data_sha256"] != data_sha256:
        raise ValueError("checkpoint data hash does not match current run")
    if checkpoint["asset_hashes_sha256"] != stable_hash(asset_hashes):
        raise ValueError("checkpoint asset hashes do not match current run")
    if checkpoint["order_sha256"] != stable_hash(order) or checkpoint["order"] != order:
        raise ValueError("checkpoint sample order does not match current run")
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("checkpoint step is invalid")
    return step


def restore_checkpoint(model: Any, optimizer: Any, checkpoint: dict[str, Any]) -> None:
    torch = checkpoint["_torch"]
    restore_trainable_state(model, checkpoint["model_tensors"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng"].cpu())
    if checkpoint["torch_cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in checkpoint["torch_cuda_rng"]])
    random.setstate(_tuple_tree(checkpoint["python_random_state"]))
