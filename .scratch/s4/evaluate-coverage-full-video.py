"""One complete public video-dev diagnostic under ADR0022 evidence-based screening authorization."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
train_path = root / "reports/s4-crossmodal-retention-training-001.json"
r = json.loads(train_path.read_text())
assert r["status"] == "completed" and r["exit_code"] == 0
assert r["adapter_sha256"] == "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818"
prior = root / "reports/s3-dev-eval-video-stable-001.json"
reference = json.loads(prior.read_text())
assert reference["status"] == "passed" and reference["samples"] == 500
argv = list(reference["command"])
assert (
    hashlib.sha256(Path(argv[argv.index("--data") + 1]).read_bytes()).hexdigest()
    == reference["data_sha256"]
)
adapter = root / "artifacts/s4-crossmodal-retention-train-001/adapter"
assert (
    hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    == r["adapter_sha256"]
)
output = root / "artifacts/s4-coverage-video-full-001"
report = root / "reports/s4-coverage-video-full-001.json"
assert not output.exists() and not report.exists()
for flag, value in [
    ("--adapter", str(adapter)),
    ("--passes", "1"),
    ("--output-dir", str(output)),
    ("--report", str(report)),
]:
    argv[argv.index(flag) + 1] = value
start = time.monotonic()
with (root / "reports/s4-coverage-full-video-001.log").open("x") as log:
    p = subprocess.run(argv, stdout=log, stderr=log, timeout=600)
result = {
    "scope": "one complete original public video dev500 diagnostic, not full S4 acceptance",
    "exit_code": p.returncode,
    "command": argv,
    "elapsed_seconds": time.monotonic() - start,
    "cost": None,
    "training_report_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
    "authorization": (
        "ADR0022 user-approved larger dev evaluation despite one-question screening miss"
    ),
    "reference_report_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
if p.returncode == 0:
    evaluated = json.loads(report.read_text())
    assert (
        evaluated["status"] == "passed"
        and evaluated["adapter_sha256"] == r["adapter_sha256"]
        and evaluated["data_sha256"] == reference["data_sha256"]
    )
    result.update(
        metrics=evaluated["metrics"], report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()
    )
with (root / "reports/s4-coverage-video-full-execution-001.json").open("x") as h:
    json.dump(result, h, indent=2)
print(json.dumps(result))
raise SystemExit(p.returncode)
