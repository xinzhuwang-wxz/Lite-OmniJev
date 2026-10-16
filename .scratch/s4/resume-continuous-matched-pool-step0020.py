"""Authorize only ten more optimizer steps with unchanged global identity."""

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-continuous-code-235ad68")
PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


previous_path = ROOT / "reports/s4-continuous-matched-pool-launch-001.json"
previous = json.loads(previous_path.read_bytes())
assert sha(previous_path) == "f9c4d0fa7e6db07103f377002a09a2ca78d1e82a2f68c45faf7ef1d27d61c307"
base_path = ROOT / "artifacts/s4-pilot-exposure-019/report.json"
assert sha(base_path) == "a72b12182192a75600afd58a3ef67d8d3238388a70e4017dfbae38e92d9f2a88"
base = json.loads(base_path.read_bytes())
assert base["exit_code"] == 0 and not base["unknown_update_bearing_inputs"]
screen_path = ROOT / "reports/s4-continuous-matched-pool-dev50-001.json"
screen = json.loads(screen_path.read_bytes())
assert screen["status"] == "completed" and screen["paired_input_hashes_verified"] == 50
assert screen["candidate"]["count"] == 50
prepared = ROOT / "artifacts/s4-continuous-matched-pool-001"
output = ROOT / "artifacts/s4-continuous-matched-pool-train-001"
raw_path = output / "run.json"
assert sha(raw_path) == previous["exposure_binding"]["runtime_report_snapshot"]["sha256"]
raw = json.loads(raw_path.read_bytes())
assert raw["status"] == "paused" and raw["steps_completed"] == 10
assert raw["adapter_sha256"] == screen["after_adapter"]
for item in previous["exposure_binding"]["source_files"]:
    assert sha(Path(item["path"])) == item["sha256"]
final_path = ROOT / "reports/s4-continuous-matched-pool-training-002.json"
snapshot = ROOT / "reports/s4-continuous-matched-pool-runtime-step0020.json"
launch_path = ROOT / "reports/s4-continuous-matched-pool-launch-002.json"
assert not any(path.exists() for path in [final_path, snapshot, launch_path])
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
command = [
    "timeout",
    "--signal=INT",
    "--kill-after=20s",
    "240",
    PYTHON,
    "-m",
    "scripts.train_s1",
    "run",
    "--config",
    str(prepared / "config-ce.json"),
    "--model-dir",
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
    "--asset-manifest",
    "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
    "--data",
    str(prepared / "train.jsonl"),
    "--allocation-ledger",
    str(prepared / "allocation-ledger.jsonl"),
    "--output-dir",
    str(output),
    "--init-adapter",
    previous["selected_init_adapter"]["path"],
    "--init-adapter-sha256",
    previous["selected_init_adapter"]["weights_sha256"],
    "--resume-from",
    raw["checkpoint"],
    "--invocation-steps",
    "10",
    "--report",
    str(final_path),
]
receipt = {
    "status": "running",
    "exit_code": 1,
    "quality_passed": False,
    "source_commit": previous["source_commit"],
    "source_archive_sha256": previous["source_archive_sha256"],
    "selected_init_adapter": previous["selected_init_adapter"],
    "parent_selection": previous["parent_selection"],
    "previous_launch": binding(previous_path),
    "screening_report": binding(screen_path),
    "command": command,
    "optimizer_boundary": "Restore complete step0010 AdamW and RNG; no reinitialization",
    "screening_stop": "DEV50 after step0020; no automatic further training",
    "cost_usd": None,
    "wrapper_sha256": sha(Path(__file__)),
}
env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
env["SWANLAB_API_KEY"] = Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip()
env["LITE_OMNIJEV_SWANLAB_MODE"] = "online"
deadline = datetime(2026, 10, 16, 4, 40, tzinfo=UTC).timestamp()
assert time.time() + 240 <= deadline, "cannot finish inside initial20min observation window"
receipt["observation_deadline_utc"] = "2026-10-16T04:40:00Z"
receipt["continuation_reason"] = "Bounded fixed-course trajectory; no quality promotion"
start = time.monotonic()
launch_path.write_text(json.dumps(receipt, indent=2) + "\n")
try:
    with (prepared / "train-invocation002.log").open("x") as log:
        result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt["exit_code"] = result.returncode
    with snapshot.open("xb") as stream:
        stream.write(raw_path.read_bytes())
    final = json.loads(final_path.read_bytes())
    current = json.loads(snapshot.read_bytes())
    assert final.pop("allocation_ledger_check") == {"ledger_rows": 1200, "verified_train": 1200}
    assert final == current
    receipt["status"] = current["status"]
    receipt["exposure_binding"] = {
        "base_exposure_report": binding(base_path),
        "prepared_report": binding(prepared / "report.json"),
        "config_file": binding(prepared / "config-ce.json"),
        "runtime_data": binding(output / "data.jsonl"),
        "final_report": binding(final_path),
        "runtime_report_snapshot": binding(snapshot),
        "runtime_report_path": str(raw_path),
        "source_files": previous["exposure_binding"]["source_files"],
        "start_step": 10,
        "end_step": current["steps_completed"],
        "resume_from": raw["checkpoint"],
    }
    assert result.returncode == 0 and current["status"] == "paused"
    assert current["steps_completed"] == 20 and len(current["losses"]) == 20
    assert current["losses"][:10] == raw["losses"]
    assert current["invocation"]["start_step"] == 10
    assert current["invocation"]["next_draw"] == 240
    assert (
        sha(Path(current["adapter_dir"]) / "adapter_model.safetensors") == current["adapter_sha256"]
    )
except Exception as error:
    receipt.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}")
finally:
    receipt["wall_seconds"] = time.monotonic() - start
    launch_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
raise SystemExit(receipt["exit_code"])
