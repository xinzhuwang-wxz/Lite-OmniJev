import importlib.util
import json
import struct
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "recovery", Path(__file__).with_name("recover_flat_shards.py")
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def safefile(path, tensors):
    offset = 0
    metadata = {}
    payload = b""
    for name, data in tensors.items():
        metadata[name] = {
            "dtype": "F32",
            "shape": [len(data) // 4],
            "data_offsets": [offset, offset + len(data)],
        }
        payload += data
        offset += len(data)
    raw = json.dumps(metadata).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


@pytest.fixture
def layout(tmp_path):
    source, base, output = [tmp_path / name for name in ("source", "base", "output")]
    for path in (source, base, output):
        path.mkdir()
    canonical = "model.layers.0.self_attn.k_proj.weight"
    alias = "model.layers.1.self_attn.k_proj.weight"
    value = struct.pack("<f", 0.75)
    safefile(source / "one.safetensors", {canonical: value})
    safefile(base / "model.safetensors", {canonical: value, alias: value})
    safefile(output / "__tmp_model-00001-of-XXXX.safetensors", {canonical: value, alias: value})
    (source / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_kv_shared_layers": 1,
                "layer_types": ["full_attention"] * 2,
            }
        )
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {canonical: "one.safetensors"}})
    )
    return source, base, output, tmp_path / "receipt.json"


def test_exact_bytes_and_alias_publish_complete_index(layout):
    recovery.recover(*layout)
    receipt = json.loads(layout[3].read_text())
    assert receipt["tensor_count"] == 2
    assert receipt["dead_alias_count"] == 1
    assert receipt["tensor_bytes"] == 8
    index = json.loads((layout[2] / "model.safetensors.index.json").read_text())
    assert len(index["weight_map"]) == 2
    assert all((layout[2] / name).is_file() for name in index["weight_map"].values())


def test_corruption_rejects_before_publishing(layout):
    shard = next(layout[2].glob("__tmp*"))
    data = shard.read_bytes()
    shard.write_bytes(data[:-4] + struct.pack("<f", 0.25))
    with pytest.raises(ValueError, match="byte mismatch"):
        recovery.recover(*layout)
    assert shard.exists()
    assert not (layout[2] / "model.safetensors.index.json").exists()


def test_missing_alias_rejects_before_publishing(layout):
    shard = next(layout[2].glob("__tmp*"))
    safefile(shard, {"model.layers.0.self_attn.k_proj.weight": struct.pack("<f", 0.75)})
    with pytest.raises(ValueError, match="key mismatch"):
        recovery.recover(*layout)
    assert not layout[3].exists()


def test_existing_destination_is_not_overwritten(layout):
    destination = layout[2] / "model-flat-00001-of-00001.safetensors"
    destination.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="destination exists"):
        recovery.recover(*layout)
    assert destination.read_bytes() == b"preserve"
