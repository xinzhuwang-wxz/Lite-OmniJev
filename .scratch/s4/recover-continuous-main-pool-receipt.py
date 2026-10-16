"""Recover only the wrapper receipt after successful paused model execution."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-continuous-code-235ad68")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


failed_path = ROOT / "reports/s4-continuous-main-pool-launch-001.json"
failed = json.loads(failed_path.read_bytes())
assert failed["status"] == "failed" and failed["exit_code"] == 1
assert failed["preflight_exit_code"] == 0
prepared = ROOT / "artifacts/s4-continuous-main-pool-001"
output = ROOT / "artifacts/s4-continuous-main-pool-train-001"
final_path = ROOT / "reports/s4-continuous-main-pool-training-001.json"
snapshot = ROOT / "reports/s4-continuous-main-pool-runtime-step0010.json"
assert snapshot.read_bytes() == (output / "run.json").read_bytes()
raw = json.loads(snapshot.read_bytes())
final = json.loads(final_path.read_bytes())
assert final.pop("allocation_ledger_check") == {"ledger_rows": 1200, "verified_train": 1200}
assert final == raw
assert raw["status"] == "paused" and raw["exit_code"] == 0
assert raw["steps_completed"] == 10
assert raw["invocation"] == {
    "steps_requested": 10,
    "start_step": 0,
    "steps_completed": 10,
    "global_steps_target": 100,
    "next_draw": 120,
    "stop_reason": "invocation_steps_limit",
}
assert sha(output / "data.jsonl") == raw["data_sha256"]
assert sha(Path(raw["adapter_dir"]) / "adapter_model.safetensors") == raw["adapter_sha256"]
assert raw["init_adapter"] == failed["selected_init_adapter"]
assert Path(raw["checkpoint"]).is_dir()
receipt = dict(failed)
receipt.update(
    status="paused",
    exit_code=0,
    error=None,
    original_failed_launch=binding(failed_path),
    recovery="CLI adds only allocation_ledger_check; raw snapshot and all other fields verified",
    recovery_script_sha256=sha(Path(__file__)),
    exposure_binding={
        "base_exposure_report": binding(ROOT / "artifacts/s4-pilot-exposure-013/report.json"),
        "prepared_report": binding(prepared / "report.json"),
        "config_file": binding(prepared / "config-ce.json"),
        "runtime_data": binding(output / "data.jsonl"),
        "final_report": binding(final_path),
        "runtime_report_snapshot": binding(snapshot),
        "runtime_report_path": str(output / "run.json"),
        "source_files": [
            binding(SOURCE / name)
            for name in ["train/trainer.py", "train/checkpoint.py", "scripts/train_s1.py"]
        ],
        "start_step": 0,
        "end_step": 10,
        "resume_from": None,
    },
)
path = ROOT / "reports/s4-continuous-main-pool-launch-recovered-002.json"
with path.open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": receipt["status"], "receipt_sha256": sha(path)}))
