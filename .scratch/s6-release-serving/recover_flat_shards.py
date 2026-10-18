"""Validate completed interrupted remap bytes before atomically publishing its index.

Uses only stdlib and bounded buffers; preserves compact sources and existing index.
This recovers the specific remap layout, not arbitrary checkpoints.
"""

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path


def header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated header: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        if length > 100_000_000:
            raise ValueError("invalid header length")
        raw = stream.read(length)
        result = json.loads(raw)
    return result, 8 + length


def digest_tensor(path, offset, info):
    start, end = info["data_offsets"]
    if start < 0 or end < start or offset + end > path.stat().st_size:
        raise ValueError(f"invalid tensor extent: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset + start)
        remaining = end - start
        while remaining:
            block = stream.read(min(8 * 1024**2, remaining))
            if not block:
                raise ValueError("truncated tensor")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def recover(source, base, output, report):
    started = time.time()
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    source_map = source_index["weight_map"]
    config = json.loads((source / "config.json").read_text())
    tc = config.get("text_config", config)
    kinds = tc["layer_types"]
    first_shared = tc["num_hidden_layers"] - tc.get("num_kv_shared_layers", 0)
    wanted_header, _ = header(base / "model.safetensors")
    wanted = set(wanted_header) - {"__metadata__"}
    files = sorted(output.glob("__tmp_model-*-of-XXXX.safetensors"))
    if not files:
        raise ValueError("no completed temporary shards")
    found = {}
    headers = {}
    tensor_bytes = 0
    for path in files:
        contents, offset = header(path)
        for name, info in contents.items():
            if name == "__metadata__":
                continue
            if name in found:
                raise ValueError(f"duplicate tensor: {name}")
            found[name] = (path, offset, info)
            tensor_bytes += info["data_offsets"][1] - info["data_offsets"][0]
    if set(found) != wanted:
        raise ValueError(
            f"key mismatch: missing={len(wanted - set(found))}, extra={len(set(found) - wanted)}"
        )
    dead = 0
    for number, (name, (path, offset, info)) in enumerate(sorted(found.items()), 1):
        canonical = name
        if canonical not in source_map:
            parts = name.split(".")
            layer = int(parts[parts.index("layers") + 1])
            member = parts[parts.index("self_attn") + 1]
            if layer < first_shared or member not in ("k_norm", "k_proj", "v_proj"):
                raise ValueError(f"unsupported alias: {name}")
            target = max(i for i in range(first_shared) if kinds[i] == kinds[layer])
            canonical = name.replace(f".layers.{layer}.", f".layers.{target}.")
            dead += 1
        src_path = source / source_map[canonical]
        if src_path not in headers:
            headers[src_path] = header(src_path)
        src_header, src_offset = headers[src_path]
        src_info = src_header[canonical]
        if any(info[key] != src_info[key] for key in ("dtype", "shape")):
            raise ValueError(f"dtype/shape mismatch: {name}")
        if digest_tensor(path, offset, info) != digest_tensor(src_path, src_offset, src_info):
            raise ValueError(f"byte mismatch: {name}")
        if number % 250 == 0:
            print(f"verified {number}/{len(found)}", flush=True)
    # All validation finishes before the first rename. Unique flat names avoid
    # replacing unrelated existing shards; original compact files are untouched.
    names = {
        p: f"model-flat-{i:05d}-of-{len(files):05d}.safetensors" for i, p in enumerate(files, 1)
    }
    if any((output / name).exists() for name in names.values()):
        raise ValueError("destination exists; refusing overwrite")
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        backup = output / "model.safetensors.index.before-flat.json"
        if backup.exists():
            raise ValueError("index backup already exists")
        backup.write_bytes(index_path.read_bytes())
    for path, name in names.items():
        path.rename(output / name)
    result_index = {
        "metadata": {"total_size": tensor_bytes},
        "weight_map": {name: names[item[0]] for name, item in sorted(found.items())},
    }
    temporary_index = output / "model.safetensors.index.json.tmp"
    temporary_index.write_text(json.dumps(result_index, indent=2) + "\n")
    os.replace(temporary_index, index_path)
    hashes = {}
    for name in names.values():
        with (output / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    hashes[index_path.name] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    receipt = {
        "status": "verified",
        "exit_code": 0,
        "tensor_count": len(found),
        "canonical_count": len(found) - dead,
        "dead_alias_count": dead,
        "shards": len(files),
        "tensor_bytes": tensor_bytes,
        "all_tensor_bytes_equal_to_source": True,
        "source": str(source),
        "base": str(base),
        "output": str(output),
        "elapsed_seconds": time.time() - started,
        "sha256": hashes,
    }
    report.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("source", "base", "output", "report"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    recover(args.source, args.base, args.output, args.report)
