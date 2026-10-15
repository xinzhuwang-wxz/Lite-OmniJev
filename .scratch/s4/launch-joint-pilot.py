"""Launch the already prepared 30-step pilot; no automatic continuation."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
code = Path("/root/autodl-tmp/s4-pilot-code-f592776")
python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
preflight_path = root / "reports/s4-joint-pilot-preflight-001.json"
preflight = json.loads(preflight_path.read_bytes())
assert preflight["status"] == "ready" and preflight["exit_code"] == 0
assert not preflight.get("blockers")
assert not (root / "artifacts/s4-joint-pilot-train-001").exists()
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
prepared = json.loads((root / "artifacts/s4-joint-pilot-001/report.json").read_bytes())
command = prepared["foreground_run_argv"]
index = command.index("uv")
command = command[:index] + [python] + command[index + 3 :]
env = os.environ.copy()
env.update(
    SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
    LITE_OMNIJEV_SWANLAB_MODE="online",
    PYTHONUNBUFFERED="1",
)
logpath = root / "artifacts/s4-joint-pilot-001/train.log"
receipt_path = root / "reports/s4-joint-pilot-launch-001.json"
assert not receipt_path.exists()
fd = os.open(logpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as log:
    process = subprocess.Popen(
        command,
        cwd=code,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
receipt = {
    "status": "launched",
    "pid": process.pid,
    "code_revision": "f592776",
    "source_archive_sha256": hashlib.sha256(
        Path("/root/autodl-tmp/omnijev-s4-pilot-code.tar").read_bytes()
    ).hexdigest(),
    "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
    "command": command,
    "wallclock_limit_seconds": 1800,
    "quality_status": "not_evaluated",
    "automatic_continuation": False,
}
with receipt_path.open("x") as out:
    json.dump(receipt, out, indent=2)
print(json.dumps({"status": "launched", "pid": process.pid, "wallclock_limit_seconds": 1800}))
