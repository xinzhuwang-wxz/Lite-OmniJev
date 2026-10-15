"""Bounded CPU effective-delta rank-16 approximation; never retrains adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

PATTERN = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.lora_([AB])\.weight$"
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def approximate(
    a1: torch.Tensor, b1: torch.Tensor, a2: torch.Tensor, b2: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Optimal rank-16 approximation in Frobenius norm via a small core SVD."""
    ac = torch.cat((a1.double() * math.sqrt(1 - alpha), a2.double() * math.sqrt(alpha)))
    bc = torch.cat((b1.double() * math.sqrt(1 - alpha), b2.double() * math.sqrt(alpha)), dim=1)
    qb, rb = torch.linalg.qr(bc, mode="reduced")
    qa, ra = torch.linalg.qr(ac.T, mode="reduced")
    u, s, vt = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    root = s[:16].sqrt()
    b = ((qb @ u[:, :16]) * root).float()
    a = (root[:, None] * (vt[:16] @ qa.T)).float()
    total = float(s.square().sum())
    discarded = float(s[16:].square().sum())
    # Include float32 serialization error without materializing the dense delta.
    # The concatenated factors encode (Bnew Anew - Bc Ac).
    error_b = torch.cat((b.double(), -bc), dim=1)
    error_a = torch.cat((a.double(), ac), dim=0)
    error_sq = float(torch.trace((error_b.T @ error_b) @ (error_a @ error_a.T)))
    error_sq = max(0.0, error_sq)
    residual = math.sqrt(error_sq / total) if total else math.sqrt(error_sq)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not math.isfinite(residual):
        raise ValueError("nonfinite fused tensor or residual")
    return (
        a,
        b,
        {
            "discarded_spectral_energy": discarded,
            "relative_frobenius_error": residual,
            "spectral_relative_frobenius_error": math.sqrt(discarded / total) if total else 0.0,
        },
    )


def read_adapter(
    directory: Path, expected_sha: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, str]]:
    weights = directory / "adapter_model.safetensors"
    config_path = directory / "adapter_config.json"
    digest = sha256(weights)
    if not re.fullmatch("[0-9a-fA-F]{64}", expected_sha) or digest != expected_sha.lower():
        raise ValueError("weights SHA256 mismatch")
    config = json.loads(config_path.read_text())
    if (
        config.get("peft_type") != "LORA"
        or config.get("r") != 16
        or config.get("lora_alpha") != 32
        or config.get("bias") != "none"
        or config.get("use_rslora", False) is not False
        or config.get("modules_to_save")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
        or config.get("use_dora", False)
        or config.get("layer_replication")
        or config.get("fan_in_fan_out", False)
        or config.get("lora_bias", False)
        or config.get("target_parameters")
    ):
        raise ValueError("unsupported PEFT configuration")
    tensors = load_file(str(weights), device="cpu")
    if not tensors:
        raise ValueError("empty adapter")
    for key, tensor in tensors.items():
        match = PATTERN.fullmatch(key)
        if (
            not match
            or tensor.ndim != 2
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("unsupported key, shape, dtype or nonfinite tensor")
        if tensor.shape[0 if match[1] == "A" else 1] != 16:
            raise ValueError("rank must be 16")
        counterpart = (
            key.replace(".lora_A.", ".lora_B.")
            if match[1] == "A"
            else key.replace(".lora_B.", ".lora_A.")
        )
        if counterpart not in tensors:
            raise ValueError("missing paired LoRA tensor")
    return config, tensors, {"weights_sha256": digest, "config_sha256": sha256(config_path)}


def fuse(
    first: Path,
    second: Path,
    first_sha: str,
    second_sha: str,
    output: Path,
    alpha: float,
    seed: int = 0,
) -> dict[str, Any]:
    if alpha not in (0.5, 0.25):
        raise ValueError("alpha must be 0.5 or 0.25")
    # Exclusive directory creation protects originals and all existing output.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    start = time.monotonic()
    report: dict[str, Any] = {
        "status": "rejected",
        "alpha": alpha,
        "seed": seed,
        "source_paths": [str(first.resolve()), str(second.resolve())],
        "expected_weights_sha256": [first_sha, second_sha],
        "cost": None,
        "method": "effective delta optimal rank16 approximation; not exact mixing or retraining",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "layers": {},
    }
    try:
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        c1, t1, h1 = read_adapter(first, first_sha)
        c2, t2, h2 = read_adapter(second, second_sha)
        report["sources"] = [h1, h2]
        if c1 != c2:
            raise ValueError("semantic adapter configurations differ")
        if t1.keys() != t2.keys() or any(t1[k].shape != t2[k].shape for k in t1):
            raise ValueError("adapter keys or shapes differ")
        result = {}
        for key in sorted(t1):
            if ".lora_A." not in key:
                continue
            bk = key.replace(".lora_A.", ".lora_B.")
            a, b, metrics = approximate(t1[key], t1[bk], t2[key], t2[bk], alpha)
            result[key], result[bk] = a, b
            report["layers"][key] = metrics
        maximum = max(m["relative_frobenius_error"] for m in report["layers"].values())
        report["max_relative_frobenius_error"] = maximum
        report["promotion_allowed"] = maximum <= 1e-3
        report["status"] = "ready_for_gpu_evaluation" if maximum <= 1e-3 else "rejected_residual"
        weight_path = output / "adapter_model.safetensors"
        save_file(result, str(weight_path))
        config_path = output / "adapter_config.json"
        config_path.write_bytes((first / "adapter_config.json").read_bytes())
        for path in (weight_path, config_path):
            path.chmod(0o600)
        report["output"] = {
            "weights_sha256": sha256(weight_path),
            "config_sha256": sha256(config_path),
        }
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["promotion_allowed"] = False
        raise
    finally:
        report["duration_seconds"] = time.monotonic() - start
        path = output / "fusion_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        path.chmod(0o600)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-sha256", required=True)
    parser.add_argument("--second-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, choices=(0.5, 0.25), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = fuse(
            args.first,
            args.second,
            args.first_sha256,
            args.second_sha256,
            args.output,
            args.alpha,
            args.seed,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0 if report["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
