"""CPU research initializer: standard decoderLoRA16→64 with zero added B columns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SOURCE_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
CONFIG_SHA = "a4fc244afa7a308262478aad466e250592fc15176d447ea12390d8bf321539ad"
MODULE = re.compile(
    r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def validate_config(config):
    required = {
        "r": 16,
        "lora_alpha": 32,
        "peft_type": "LORA",
        "bias": "none",
        "lora_bias": False,
        "use_rslora": False,
        "use_dora": False,
        "use_qalora": False,
        "fan_in_fan_out": False,
        "lora_dropout": 0.0,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "init_lora_weights": True,
        "inference_mode": True,
    }
    for key, value in required.items():
        require(key in config and config[key] == value, "unsupported adapter behavior: " + key)
    harmless = {
        "base_model_name_or_path",
        "auto_mapping",
        "peft_version",
        "revision",
        "task_type",
        "megatron_core",
        "qalora_group_size",
        "target_modules",
    }
    empty = {
        "alora_invocation_tokens",
        "arrow_config",
        "corda_config",
        "ensure_weight_tying",
        "eva_config",
        "exclude_modules",
        "kasa_config",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_ga_config",
        "megatron_config",
        "monteclora_config",
        "target_parameters",
        "trainable_token_indices",
        "use_bdlora",
        "velora_config",
    }
    require(not set(config) - set(required) - harmless - empty, "unknown config fields")
    for key in empty:
        require(config.get(key) in (None, False, {}), "nonstandard extension: " + key)
    targets = config["target_modules"]
    require(
        isinstance(targets, list)
        and len(targets) == len(set(targets))
        and targets
        and all(MODULE.fullmatch(m) for m in targets),
        "decoder targets only",
    )
    return targets


def bit_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def expand_tensors(tensors, targets):
    require(
        set(tensors)
        == {
            f"base_model.model.{module}.lora_{kind}.weight"
            for module in targets
            for kind in ["A", "B"]
        },
        "only exact decoder A/B tensors",
    )
    generator = torch.Generator(device="cpu").manual_seed(42)
    result = {}
    for module in sorted(targets):
        names = [f"base_model.model.{module}.lora_{kind}.weight" for kind in ["A", "B"]]
        a, b = [tensors[n] for n in names]
        require(
            a.device.type == b.device.type == "cpu"
            and a.ndim == b.ndim == 2
            and a.shape[0] == b.shape[1] == 16,
            "rank16shapes",
        )
        require(
            a.dtype == b.dtype and a.dtype in [torch.float32, torch.float16, torch.bfloat16],
            "floating A/B dtype",
        )
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), "finite source")
        expanded_a = torch.empty((64, a.shape[1]), dtype=a.dtype)
        expanded_b = torch.zeros((b.shape[0], 64), dtype=b.dtype)
        expanded_a[:16].copy_(a)
        torch.nn.init.kaiming_uniform_(expanded_a[16:], a=math.sqrt(5), generator=generator)
        expanded_b[:, :16].copy_(b)
        require(
            bit_equal(expanded_a[:16], a) and bit_equal(expanded_b[:, :16], b),
            "old prefix bit exact",
        )
        require(
            bool(torch.isfinite(expanded_a).all()) and bool(torch.isfinite(expanded_b).all()),
            "finite expanded",
        )
        require(torch.count_nonzero(expanded_b[:, 16:]).item() == 0, "new B zero")
        result.update(zip(names, [expanded_a, expanded_b], strict=True))
    return result


def convert(source, out):
    started = time.monotonic()
    torch.set_num_threads(1)
    require(not out.exists(), "output exists")
    weights = source / "adapter_model.safetensors"
    config_path = source / "adapter_config.json"
    require(
        sha(weights) == SOURCE_SHA and sha(config_path) == CONFIG_SHA, "fixed f2db source hashes"
    )
    config = json.loads(config_path.read_bytes())
    targets = validate_config(config)
    tensors = load_file(str(weights), device="cpu")
    expanded = expand_tensors(tensors, targets)
    new_config = {**config, "r": 64, "lora_alpha": 128}
    require(
        {k for k in config if config[k] != new_config[k]} == {"r", "lora_alpha"},
        "only rank/alpha differ",
    )
    require(
        config["lora_alpha"] / config["r"] == new_config["lora_alpha"] / new_config["r"] == 2,
        "scale unchanged",
    )
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    os.umask(0o077)
    out.mkdir(mode=0o700)
    save_file(expanded, str(out / "adapter_model.safetensors"), metadata=metadata)
    (out / "adapter_config.json").write_text(json.dumps(new_config, indent=2) + "\n")
    require(sha(weights) == SOURCE_SHA and sha(config_path) == CONFIG_SHA, "source changed")
    result = {
        "status": "converted_research_initializer",
        "exit_code": 0,
        "gpu_used": False,
        "training_authorized": False,
        "production_replacement": False,
        "cost": None,
        "source_path": str(source),
        "source_weights_sha256": SOURCE_SHA,
        "source_config_sha256": CONFIG_SHA,
        "output_path": str(out),
        "output_sha256": {p.name: sha(p) for p in out.iterdir()},
        "old_rank": 16,
        "new_rank": 64,
        "old_alpha": 32,
        "new_alpha": 128,
        "scale": 2,
        "modules": len(targets),
        "tensor_count": len(expanded),
        "seed": 42,
        "checks": {
            "old_A_B_prefix_bit_exact": True,
            "new_B_zero": True,
            "finite": True,
            "same_module_set": True,
            "source_hashes_unchanged": True,
        },
        "mathematical_delta": "2*[B,0]@[A;Anew] = 2*B@A; no giant BA materialization",
        "gpu_parity": "pending; requires actual legal/joint probability maximum difference<=1e-3",
        "standard_peft_format": True,
        "hf_vllm_runtime_compatibility": "not_tested",
        "transform_sha256": sha(Path(__file__)),
        "duration_seconds": time.monotonic() - started,
    }
    (out / "rank-expansion.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    existed = a.output.exists()
    try:
        result = convert(a.source.resolve(), a.output.resolve())
    except Exception as exc:
        result = {"status": "rejected", "exit_code": 1, "error": str(exc), "gpu_used": False}
        if not existed:
            a.output.mkdir(mode=0o700, exist_ok=True)
            (a.output / "failure.json").write_text(json.dumps(result) + "\n")
    print(json.dumps(result))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
