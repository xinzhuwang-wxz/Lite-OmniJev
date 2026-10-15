"""CPU comparison of the frozen Brier candidate with both CE controls."""

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
paired.CANDIDATE = "20ea65d4eaee5f0e4b6591ad02dacf94b00f5ef0331a7a97a1f877f40db716e2"
for name, baseline, summary in [
    (
        "ce0",
        "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed",
        root / "artifacts/s4-claim-ce-retention-001/receipt.json",
    ),
    (
        "matched-ce",
        "97201e304d26353833845078d1d392fd5e8d966d8d5e3952b2c4d7efb0ebd170",
        root / "artifacts/s4-temporal-curriculum-video150-001/receipt.json",
    ),
]:
    paired.BASELINE = baseline
    output = root / f"reports/s4-temporal-brier-vs-{name}-001.json"
    sys.argv = [
        str(source),
        "--baseline-summary",
        str(summary),
        "--candidate-report",
        str(root / "artifacts/s4-temporal-brier-video150-001/video/report.json"),
        "--output",
        str(output),
    ]
    assert paired.main() == 0
    report = json.loads(output.read_text())
    report["comparison_binding"] = {
        "baseline_adapter_sha256": baseline,
        "candidate_adapter_sha256": paired.CANDIDATE,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Pinned identities replace defaults; all validation remains active.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
