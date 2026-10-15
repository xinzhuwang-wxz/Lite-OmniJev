"""Restore only source-identical norm aliases omitted by HF tied-weight serialization."""

import hashlib
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

root = Path("/root/autodl-tmp/s1-text-lora-001")
output = root / "artifacts/s3-merged-fp32-reference-001"
source = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2/model.safetensors"
)
index_path = output / "model.safetensors.index.json"
index_bytes = index_path.read_bytes()
index = json.loads(index_bytes)
with safe_open(str(source), framework="pt", device="cpu") as original:
    missing = set(original.keys()) - set(index["weight_map"])
    omitted_projections = {
        key
        for key in missing
        if re.fullmatch(r"model.language_model.layers.\d+.self_attn.[kv]_proj.weight", key)
    }
    assert len(omitted_projections) == 36
    missing -= omitted_projections
    assert len(missing) == 18
    assert all(
        re.fullmatch(r"model.language_model.layers.\d+.self_attn.k_norm.weight", k) for k in missing
    )
    tensors = {key: original.get_tensor(key).float().contiguous() for key in sorted(missing)}
backup = root / "artifacts/s3-serving-reference-001/pre-alias-index.json"
assert not backup.exists()
backup.write_bytes(index_bytes)
shard = output / "source-norm-aliases.safetensors"
assert not shard.exists()
save_file(tensors, str(shard), metadata={"format": "pt"})
for key in tensors:
    index["weight_map"][key] = shard.name
index["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in tensors.values())
index_path.write_text(json.dumps(index, indent=2) + "\n")
report = {
    "status": "aliases_restored",
    "exit_code": 0,
    "scope": "Explicit export-key repair only; serving parity still pending",
    "source_model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "restored_keys": sorted(tensors),
    "restored_count": len(tensors),
    "shared_projection_keys_not_modified": len(omitted_projections),
    "old_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
    "new_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    "alias_shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    "method": "Original norm tensors cast to float32; trained matrices unchanged",
}
(root / "reports/s3-merged-alias-repair-001.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
