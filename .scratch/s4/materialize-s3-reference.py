"""Materialize the same frozen base plus LoRA into a separate FP32 reference copy."""

import hashlib
import json
import time
from pathlib import Path

import peft
import torch
import transformers

root = Path("/root/autodl-tmp/s1-text-lora-001")
ref_path = root / "reports/s3-serving-hf1120-reference-001.json"
reference = json.loads(ref_path.read_text())
assert reference["status"] == "reference-ready" and reference["exit_code"] == 0
provenance = reference["provenance"]
base = Path(provenance["model_dir"])
adapter = root / "artifacts/s3-stable-probe-train-001/adapter"
output = root / "artifacts/s3-merged-fp32-reference-001"
assert not output.exists()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


started = time.time()
assert (
    sha(adapter / "adapter_model.safetensors")
    == "1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755"
)
for name, expected in provenance["model_files"].items():
    assert sha(base / name) == expected
assert torch.cuda.is_available()
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
    str(base),
    local_files_only=True,
    trust_remote_code=False,
    use_safetensors=True,
    torch_dtype=torch.float32,
    attn_implementation="sdpa",
)
model = peft.PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
model.requires_grad_(False)
model.eval()
model.to("cuda")
with torch.no_grad():
    merged = model.merge_and_unload(safe_merge=True)
assert not any("lora_" in name for name, _ in merged.named_parameters())
merged.save_pretrained(str(output), safe_serialization=True, max_shard_size="4GB")
transformers.AutoProcessor.from_pretrained(
    str(base), local_files_only=True, trust_remote_code=False
).save_pretrained(str(output))
config_path = output / "config.json"
config = json.loads(config_path.read_text())
dtype_changes = {}
for name in ["root", "text_config", "vision_config", "audio_config"]:
    section = config if name == "root" else config[name]
    dtype_changes[name] = {"before": section.get("dtype"), "after": "float32"}
    section["dtype"] = "float32"
config_path.write_text(json.dumps(config, indent=2) + "\n")
files = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
report = {
    "status": "materialized",
    "exit_code": 0,
    "scope": (
        "Independent derived numerical-reference artifact, not a new base or trained model; "
        "cross-engine parity pending"
    ),
    "base_model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "base_file_hashes": provenance["model_files"],
    "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
    "hf_reference_sha256": sha(ref_path),
    "script_sha256": sha(Path(__file__)),
    "method": "PEFT merge_and_unload(safe_merge=True), float32, TF32 disabled",
    "dtype_changes": dtype_changes,
    "output_files": files,
    "output_bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
    "duration_seconds": time.time() - started,
    "base_and_adapter_source_files_modified": False,
    "versions": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    },
}
(root / "reports/s3-merged-fp32-materialization-001.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
