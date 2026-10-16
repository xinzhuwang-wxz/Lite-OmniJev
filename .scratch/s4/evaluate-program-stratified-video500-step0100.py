"""Measure the program-stratified candidate on all 500 public video examples."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/s4-monitored-eval-code-001")
python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(Path("/root/autodl-tmp/omnijev-s4-monitored-eval-001.tar")) == (
    "b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516"
)
previous_path = root / "reports/s3-dev-eval-video-stable-001.json"
previous = json.loads(previous_path.read_bytes())
assert previous["status"] == "passed" and previous["samples"] == 500
assert previous["data_sha256"] == "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2"
prior_command = previous["command"]


def argument(flag):
    return prior_command[prior_command.index(flag) + 1]


data, media, model = [Path(argument(f)) for f in ["--data", "--media-root", "--model-dir"]]
assert sha(data) == previous["data_sha256"]
trained_path = root / "reports/s4-program-stratified-training-step0100.json"
trained = json.loads(trained_path.read_bytes())
assert trained["status"] == "completed" and trained["steps_completed"] == 100
adapter = Path(trained["adapter_dir"])
assert sha(trained_path) == "421b950611a8e8db3fa1b0ae75c960790ae56c29a80a218976c171fbdf67ae97"
assert trained["init_adapter"]["weights_sha256"] == (
    "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
)
assert (
    sha(adapter / "adapter_model.safetensors")
    == trained["adapter_sha256"]
    == ("e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd")
)
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
out = root / "artifacts/s4-program-stratified-video500-step0100"
out.mkdir(exist_ok=False)
command = [
    "timeout",
    "--signal=INT",
    "--kill-after=20s",
    "420",
    python,
    "-m",
    "scripts.s1_dev_eval",
    "--model-dir",
    str(model),
    "--adapter",
    str(adapter),
    "--data",
    str(data),
    "--passes",
    "1",
    "--video-frames",
    "16",
    "--output-dir",
    str(out / "eval"),
    "--report",
    str(out / "report.json"),
    "--media-root",
    str(media),
]
receipt = {
    "status": "running",
    "stage_gate": False,
    "budget_seconds": 420,
    "scope": "Complete original public DEV video500, one pass; no holdout/calibration",
    "source_commit": "74e96e2",
    "source_archive_sha256": "b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516",
    "data_sha256": sha(data),
    "training_report_sha256": sha(trained_path),
    "source_protocol_sha256": sha(previous_path),
    "adapter_sha256": trained["adapter_sha256"],
    "command": command,
    "runner_sha256": sha(Path(__file__)),
    "cost_usd": None,
}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
started = time.monotonic()
try:
    with (out / "run.log").open("x") as log:
        result = subprocess.run(
            command,
            cwd=source,
            env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    receipt["exit_code"] = result.returncode
    assert result.returncode == 0, "Full-cohort evaluator failed or exceeded budget"
    report = json.loads((out / "report.json").read_bytes())
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["samples"] == 500 and report["passes"] == 1 and report["video_num_frames"] == 16
    assert report["data_sha256"] == sha(data) == receipt["data_sha256"]
    assert report["adapter_sha256"] == trained["adapter_sha256"]
    receipt.update(
        status="completed", metrics=report["metrics"], report_sha256=sha(out / "report.json")
    )
except Exception as error:
    receipt.update(status="failed", exit_code=receipt.get("exit_code") or 1, error=str(error))
finally:
    receipt["wall_seconds"] = time.monotonic() - started
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
raise SystemExit(receipt["exit_code"])
