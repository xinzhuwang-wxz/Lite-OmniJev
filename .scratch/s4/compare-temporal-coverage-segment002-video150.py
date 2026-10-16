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
paired.BASELINE = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
paired.CANDIDATE = "250ef3e3c8a5359e6d7b6aaaeafa6d12b253603669a4b9f8a7a70c52c4b72707"
output = root / "reports/s4-temporal-coverage-video150-paired-002.json"
sys.argv = [
    str(source),
    "--baseline-summary",
    str(root / "artifacts/s4-temporal-coverage-video150-001/receipt.json"),
    "--candidate-report",
    str(root / "artifacts/s4-temporal-coverage-video150-002/video/report.json"),
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
