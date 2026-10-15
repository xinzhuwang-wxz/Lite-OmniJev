"""Export a single half-strength decision adapter by scaling only LoRA B."""

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

root = Path("/root/autodl-tmp/s1-text-lora-001")
parent = root / "artifacts/s2-text-lora-001/joint-v2-001/adapter"
output = root / "artifacts/s2-text-lora-001/half-strength-v2-001/adapter"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


assert (
    sha(parent / "adapter_model.safetensors")
    == "db45fb13e651acf07ed5901584960f7bfa60abb1f1a26c609a6a89511c35a4ee"
)
state = load_file(str(parent / "adapter_model.safetensors"))
scaled = 0
for name, value in state.items():
    assert torch.isfinite(value).all()
    if ".lora_B." in name:
        state[name] = (value.float() * 0.5).to(value.dtype).contiguous()
        scaled += 1
    else:
        assert ".lora_A." in name
assert scaled > 0 and scaled * 2 == len(state)
output.mkdir(parents=True, exist_ok=False)
shutil.copyfile(parent / "adapter_config.json", output / "adapter_config.json")
save_file(state, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
report = {
    "status": "completed",
    "exit_code": 0,
    "method": "LoRA B *= 0.5; A/config unchanged",
    "parent_sha256": sha(parent / "adapter_model.safetensors"),
    "adapter_sha256": sha(output / "adapter_model.safetensors"),
    "scaled_tensors": scaled,
    "training_performed": False,
    "quality_gate": "not_run",
    "script_sha256": sha(Path(__file__)),
}
(root / "reports/s2-half-strength-adapter-001.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
