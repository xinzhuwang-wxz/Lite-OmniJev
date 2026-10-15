"""Verify prepared artifacts, preflight, then run one bounded gold-only candidate."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/s4-claim-pair-code-001")
prepared_dir = root / "artifacts/s4-claim-pair-pilot-001"
prepared_path = prepared_dir / "report.json"
prepared = json.loads(prepared_path.read_text())
assert prepared["status"] == "prepared_pending_pair_hashes" and prepared["exit_code"] == 0


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


for name, expected in prepared["output_sha256"].items():
    assert sha(prepared_dir / name) == expected, name
receipt_path = root / "reports/s4-claim-pair-ce0-launch-001.json"
assert not receipt_path.exists()
receipt = {
    "status": "preflight_running",
    "source_commit": "3fe0ab2",
    "source_archive_sha256": sha(Path("/root/autodl-tmp/omnijev-s4-claim-pair-001.tar")),
    "prepared_report_sha256": sha(prepared_path),
    "quality_passed": False,
    "cost_usd": None,
}
env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
start = time.monotonic()
try:
    assert prepared["expanded_isolation"] == prepared["pilot_isolation"] == "isolated"
    assert (
        receipt["source_archive_sha256"]
        == "9132c2d552af09aa4569c33df3b370fc61abe5c61eb34fa2e235b795a28f2378"
    )
    python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
    common = [
        "--config",
        str(prepared_dir / "config-ce0.json"),
        "--model-dir",
        "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
        "--asset-manifest",
        "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
        "--data",
        str(prepared_dir / "train.jsonl"),
        "--allocation-ledger",
        str(prepared_dir / "allocation-ledger.jsonl"),
        "--output-dir",
        str(root / "artifacts/s4-claim-pair-ce0-train-001"),
        "--init-adapter",
        prepared["init_adapter"]["path"],
        "--init-adapter-sha256",
        prepared["init_adapter"]["weights_sha256"],
    ]
    preflight = [
        python,
        "-m",
        "scripts.train_s1",
        "preflight",
        *common,
        "--report",
        str(root / "reports/s4-claim-pair-ce0-preflight-001.json"),
    ]
    preflight[preflight.index("--report") + 1] = str(
        root / "reports/s4-claim-pair-ce0-preflight-001.json"
    )
    with (prepared_dir / "ce0-preflight.log").open("x") as log:
        result = subprocess.run(
            preflight, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
        )
    receipt["preflight_exit_code"] = result.returncode
    assert result.returncode == 0, "CPU preflight failed"
    p = json.loads((root / "reports/s4-claim-pair-ce0-preflight-001.json").read_text())
    assert p["status"] == "ready" and not p.get("blockers")
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    env.update(
        SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
        LITE_OMNIJEV_SWANLAB_MODE="online",
    )
    command = [
        "timeout",
        "--signal=INT",
        "--kill-after=20s",
        "240",
        python,
        "-m",
        "scripts.train_s1",
        "run",
        *common,
        "--report",
        str(root / "reports/s4-claim-pair-ce0-training-001.json"),
    ]
    receipt.update(status="training", command=command)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    train_start = time.monotonic()
    with (prepared_dir / "ce0-train.log").open("x") as log:
        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(
        status="completed" if result.returncode == 0 else "failed",
        exit_code=result.returncode,
        training_wall_seconds=time.monotonic() - train_start,
    )
except Exception as error:
    receipt.update(status="failed", exit_code=1, error=str(error))
finally:
    receipt["total_wall_seconds"] = time.monotonic() - start
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
raise SystemExit(receipt["exit_code"])
