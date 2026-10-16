"""Run pinned paired CPU diagnostics for coverage versus CE0 on video150."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

source = Path("/root/autodl-tmp/compare-temporal-curriculum-001.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "f2fa4821d37f46694baa94b573db562ffa3108b0d29953e00a6bcf6dad5e0aff"
)
spec = importlib.util.spec_from_file_location("paired", source)
assert spec and spec.loader
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
root = Path("/root/autodl-tmp/s1-text-lora-001")
paired.CANDIDATE = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
output = root / "reports/s4-temporal-coverage-video150-paired-001.json"
sys.argv = [
    str(source),
    "--baseline-summary",
    str(root / "artifacts/s4-claim-ce-retention-001/receipt.json"),
    "--candidate-report",
    str(root / "artifacts/s4-temporal-coverage-video150-001/video/report.json"),
    "--output",
    str(output),
]
assert paired.main() == 0
report = json.loads(output.read_text())
report["wrapper_binding"] = {
    "candidate_sha256": paired.CANDIDATE,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
output.write_text(json.dumps(report, indent=2) + "\n")
