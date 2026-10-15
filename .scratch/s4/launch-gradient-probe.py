"""Run one identity-bound CPU preflight then a non-updating GPU diagnostic."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-gradient-probe-code-001")
ARCHIVE = Path("/root/autodl-tmp/omnijev-s4-gradient-probe-001.tar")
PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
PREP = ROOT / "artifacts/s4-crossmodal-retention-pilot-002"
receipt_path = ROOT / "reports/s4-gradient-probe-launch-001.json"
assert not receipt_path.exists()
assert (
    hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    == "f4f8fa30c5113b1a82493881507b17ea6308216fd27519f8a3a6e2975ec27502"
)
SOURCE.mkdir(exist_ok=False)
subprocess.run(["tar", "-xf", str(ARCHIVE), "-C", str(SOURCE)], check=True)
command = [
    PYTHON,
    "-m",
    "scripts.train_gradient_probe",
    "--config",
    str(PREP / "config-30.json"),
    "--data",
    str(PREP / "train.jsonl"),
    "--ledger",
    str(PREP / "allocation-ledger.jsonl"),
    "--model-dir",
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
    "--asset-manifest",
    "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
    "--init-adapter",
    str(ROOT / "artifacts/s4-joint-pilot-train-001/adapter"),
    "--expected-config-sha256",
    "6829564f3eb58122d166b1cee486b51244fabffc9a4cc862f72ed2bf904b7a46",
    "--expected-data-sha256",
    "c886baf002fb54e8786c3ec957e03a10e1eccf40363a8c4d5cee43e64616e8ef",
    "--expected-ledger-sha256",
    "9d083c679c300d612d9b2f50f7ece2c6a8cb24f377acb8941bb0e8299006eec1",
    "--expected-asset-manifest-sha256",
    "207ce3a8b2ddc8a3a31f199935842bbb42e72ea029febf14978d895d120b1461",
    "--expected-adapter-sha256",
    "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac",
    "--expected-adapter-config-sha256",
    "6eb4080806ed4e4c7e1cc304268468b08da355437df7b02db3e6a69bae97ff62",
    "--expected-code-revision",
    "unavailable",
]
receipt = {
    "status": "running",
    "source_commit": "d96d580",
    "archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
    "optimizer_updates": 0,
    "quality_passed": False,
    "commands": [],
}
start = time.monotonic()
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(SOURCE))
try:
    for phase in ["preflight", "gpu"]:
        if phase == "gpu":
            assert not subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip()
        output = ROOT / f"reports/s4-gradient-probe-{phase}-001.json"
        args = [
            "timeout",
            "--signal=INT",
            "--kill-after=20s",
            "180",
            *command,
            "--output",
            str(output),
        ]
        if phase == "gpu":
            args.append("--execute")
        receipt["commands"].append(args)
        receipt["active_phase"] = phase
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        with (ROOT / f"reports/s4-gradient-probe-{phase}-001.log").open("x") as log:
            result = subprocess.run(args, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt[phase + "_exit_code"] = result.returncode
        assert result.returncode == 0, phase + " failed"
        report = json.loads(output.read_text())
        assert report["exit_code"] == 0
        assert report["status"] == ("completed" if phase == "gpu" else "preflight_verified")
    receipt.update(status="completed", exit_code=0)
except Exception as exc:
    receipt.update(status="failed", exit_code=1, error=str(exc))
finally:
    receipt["wall_seconds"] = time.monotonic() - start
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
raise SystemExit(receipt["exit_code"])
