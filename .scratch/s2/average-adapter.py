"""Export one bounded 50/50 checkpoint-average candidate; no model loading."""

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

root = Path("/root/autodl-tmp/s1-text-lora-001")
run = root / "artifacts/s2-text-lora-001/joint-v2-001"
parents = [run / "adapter-at-step-0800", run / "adapter"]
output = root / "artifacts/s2-text-lora-001/average-0800-1600-001/adapter"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


configs = [json.loads((p / "adapter_config.json").read_text()) for p in parents]
assert configs[0] == configs[1]
states = [load_file(str(p / "adapter_model.safetensors"), device="cpu") for p in parents]
assert set(states[0]) == set(states[1])
combined = {}
for name, left in states[0].items():
    right = states[1][name]
    assert left.shape == right.shape and left.dtype == right.dtype
    assert "lora_" in name and torch.isfinite(left).all() and torch.isfinite(right).all()
    combined[name] = ((left.float() + right.float()) * 0.5).to(left.dtype).contiguous()
output.mkdir(parents=True, exist_ok=False)
shutil.copyfile(parents[1] / "adapter_config.json", output / "adapter_config.json")
save_file(combined, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
assert set(load_file(str(output / "adapter_model.safetensors"))) == set(combined)
report = {
    "status": "completed",
    "exit_code": 0,
    "method": (
        "elementwise FP32 50/50 mean of LoRA factors, cast to original dtype; not mean of"
        " effective delta matrices"
    ),
    "inputs": {str(p): sha(p / "adapter_model.safetensors") for p in parents},
    "adapter_sha256": sha(output / "adapter_model.safetensors"),
    "config_sha256": sha(output / "adapter_config.json"),
    "script_sha256": sha(Path(__file__)),
    "tensors": len(combined),
    "training_performed": False,
    "quality_gate": "not_run",
}
(root / "reports/s2-average-adapter-001.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
