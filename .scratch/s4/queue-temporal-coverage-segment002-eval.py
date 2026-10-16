"""Observe this bounded launch and evaluate only its successful terminal output."""

import json
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
receipt = root / "reports/s4-temporal-coverage-launch-002.json"
started = time.monotonic()
while time.monotonic() - started < 480:
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["status"] in ["completed", "failed"]:
            assert result["status"] == "completed" and result["exit_code"] == 0, result
            break
    time.sleep(5)
else:
    raise SystemExit("Launch not terminal; no evaluation started")
with (root / "reports/s4-temporal-coverage-segment002-dev50-002.log").open("x") as log:
    result = subprocess.run(
        [
            "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python",
            "/root/autodl-tmp/evaluate-temporal-coverage-segment002-dev50.py",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=225,
    )
print(json.dumps({"stage": "coverage-segment002-dev50", "exit_code": result.returncode}))
raise SystemExit(result.returncode)
