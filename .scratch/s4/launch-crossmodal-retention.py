"""Verify prepared artifacts, preflight, then run one bounded gold-only candidate."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/s4-reference-code-fix-001")
prepared_dir = root / "artifacts/s4-crossmodal-retention-pilot-002"
prepared_path = prepared_dir / "report.json"
prepared = json.loads(prepared_path.read_text())
assert prepared["status"] == "prepared" and prepared["exit_code"] == 0


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


for name, expected in prepared["output_sha256"].items():
    assert sha(prepared_dir / name) == expected, name
receipt_path = root / "reports/s4-crossmodal-retention-launch-001.json"
assert not receipt_path.exists()
receipt = {
    "status": "preflight_running",
    "source_commit": "44214af",
    "source_archive_sha256": sha(Path("/root/autodl-tmp/omnijev-s4-reference-fix.tar")),
    "prepared_report_sha256": sha(prepared_path),
    "quality_passed": False,
    "cost_usd": None,
}
env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
start = time.monotonic()
try:
    preflight = list(prepared["preflight_argv"])
    preflight[preflight.index("--report") + 1] = str(
        root / "reports/s4-crossmodal-retention-preflight-001.json"
    )
    with (prepared_dir / "preflight.log").open("x") as log:
        result = subprocess.run(
            preflight, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
        )
    receipt["preflight_exit_code"] = result.returncode
    assert result.returncode == 0, "CPU preflight failed"
    p = json.loads((root / "reports/s4-crossmodal-retention-preflight-001.json").read_text())
    assert p["status"] == "ready" and not p.get("blockers")
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    env.update(
        SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
        LITE_OMNIJEV_SWANLAB_MODE="online",
    )
    command = prepared["foreground_run_argv"]
    receipt.update(status="training", command=command)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    train_start = time.monotonic()
    with (prepared_dir / "train.log").open("x") as log:
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
