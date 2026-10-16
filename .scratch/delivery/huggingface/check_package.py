"""Offline allowlist check; never uploads or declares final release readiness."""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

ALLOWED = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "release-manifest.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "examples/infer.py",
    "examples/reproduce.md",
}
REQUIRED = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "release-manifest.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "examples/infer.py",
    "examples/reproduce.md",
}
KEY = re.compile(
    r"base_model\.model\.model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.lora_([AB])\.weight"
)
SECRET = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    rb'(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*=\s*["\x27][^"\x27]+)'
)


def check(root: Path) -> dict:
    errors = []
    found = set()
    hashes = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append("symlink:" + rel)
            continue
        if path.is_dir():
            if rel != "examples":
                errors.append("unexpected_directory:" + rel)
            continue
        found.add(rel)
        if rel not in ALLOWED:
            errors.append("not_allowlisted:" + rel)
            continue  # Never read rejected files (including credentials/media/private data).
        if rel == "adapter_model.safetensors":
            try:
                with path.open("rb") as h:
                    size = struct.unpack("<Q", h.read(8))[0]
                    if not 0 < size <= 4_000_000:
                        raise ValueError("header_size")
                    header = json.loads(h.read(size))
                tensors = {k: v for k, v in header.items() if k != "__metadata__"}
                if not tensors or any(not KEY.fullmatch(k) for k in tensors):
                    raise ValueError("non_decoder_lora_tensor")
                for k, value in tensors.items():
                    shape = value["shape"]
                    if (
                        len(shape) != 2
                        or value["dtype"] not in {"F32", "BF16", "F16"}
                        or 16 != shape[0 if KEY.fullmatch(k)[1] == "A" else 1]
                    ):
                        raise ValueError("lora_shape_dtype")
                for k in tensors:
                    mate = (
                        k.replace(".lora_A.", ".lora_B.")
                        if ".lora_A." in k
                        else k.replace(".lora_B.", ".lora_A.")
                    )
                    if mate not in tensors:
                        raise ValueError("unpaired_lora")
            except (ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError):
                errors.append("invalid_adapter_header")
                continue
        else:
            if path.stat().st_size > 300_000:
                errors.append("oversized_text:" + rel)
                continue
            body = path.read_bytes()
            if SECRET.search(body):
                errors.append("credential_pattern:" + rel)
                continue
        with path.open("rb") as h:
            hashes[rel] = hashlib.file_digest(h, "sha256").hexdigest()
    missing = sorted(REQUIRED - found)
    return {
        "status": "rejected"
        if errors
        else ("inventory_incomplete" if missing else "structure_checked"),
        "errors": errors,
        "missing": missing,
        "files_sha256": hashes,
        "release_ready": False,
        "quality_gates": "pending_external_stage_acceptance",
        "namespace_verified": False,
        "uploaded": False,
        "gpu_used": False,
        "cost": None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    a = p.parse_args()
    result = check(a.package)
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result["errors"] else 0)
