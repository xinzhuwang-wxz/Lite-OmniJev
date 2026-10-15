"""One complete public video-dev diagnostic after all fixed retention screens pass."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", choices=["050", "025"], required=True)
args = parser.parse_args()
root = Path("/root/autodl-tmp/s1-text-lora-001")
retention = root / f"reports/s4-fusion{args.alpha}-retention-001.json"
for _ in range(90):
    if retention.exists() and json.loads(retention.read_text())["status"] == "completed":
        break
    time.sleep(5)
else:
    raise SystemExit("retention_not_terminal")
r = json.loads(retention.read_text())
assert r["qualified_for_full_dev"], "retention_failed"
prior = root / "reports/s3-dev-eval-video-stable-001.json"
reference = json.loads(prior.read_text())
assert reference["status"] == "passed" and reference["samples"] == 500
argv = list(reference["command"])
assert (
    hashlib.sha256(Path(argv[argv.index("--data") + 1]).read_bytes()).hexdigest()
    == reference["data_sha256"]
)
adapter = root / f"artifacts/s4-fusion{args.alpha}-002"
fusion = json.loads((adapter / "fusion_report.json").read_text())
assert fusion["promotion_allowed"] and fusion["output"]["weights_sha256"] == r["adapter_sha256"]
output = root / f"artifacts/s4-fusion{args.alpha}-video-full-001"
report = root / f"reports/s4-fusion{args.alpha}-video-full-001.json"
assert not output.exists() and not report.exists()
for flag, value in [
    ("--adapter", str(adapter)),
    ("--passes", "1"),
    ("--output-dir", str(output)),
    ("--report", str(report)),
]:
    argv[argv.index(flag) + 1] = value
start = time.monotonic()
with (adapter / "full-video.log").open("x") as log:
    p = subprocess.run(argv, stdout=log, stderr=log, timeout=600)
result = {
    "scope": "one complete original public video dev500 diagnostic, not full S4 acceptance",
    "exit_code": p.returncode,
    "command": argv,
    "elapsed_seconds": time.monotonic() - start,
    "cost": None,
    "retention_sha256": hashlib.sha256(retention.read_bytes()).hexdigest(),
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
with (root / f"reports/s4-fusion{args.alpha}-video-full-execution-001.json").open("x") as h:
    json.dump(result, h, indent=2)
print(json.dumps(result))
raise SystemExit(p.returncode)
