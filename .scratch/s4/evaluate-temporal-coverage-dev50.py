"""Screen one fresh-family training segment on fixed DEV temporal50."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/s4-motion-code-142e744")
runner = source / ".scratch/s4/evaluate-video-motion-coverage.py"
assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
    "e392f10b7738f9cf776087dfb7cd40b35377d974f986fc5cd9ad47bb77b51246"
)


def read(p):
    return json.loads(p.read_text())


trained = read(root / "reports/s4-temporal-coverage-training-001.json")
prepared = read(root / "artifacts/s4-temporal-coverage-pilot-001/report.json")
assert trained["status"] == "completed" and trained["exit_code"] == 0
assert trained["steps_completed"] == 10
assert trained["data_sha256"] == prepared["output_sha256"]["train.jsonl"]
assert trained["init_adapter"]["weights_sha256"] == (
    "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"
)
assert trained["config"]["sample_count"] == 120
assert trained["config"]["accumulation"] == 12
assert trained["config"].get("pair_loss_weight", 0) == 0
assert trained["config"].get("brier_weight", 0) == 0
os.chdir(source)
sys.path.insert(0, str(source))
spec = importlib.util.spec_from_file_location("motion", runner)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ADAPTER = root / "artifacts/s4-temporal-coverage-train-001/adapter"
module.ADAPTER_SHA = trained["adapter_sha256"]
out = root / "artifacts/s4-temporal-coverage-dev50-001"
for flags in [[], ["--execute", "--policy", "uniform"]]:
    sys.argv = [str(runner), "--output", str(out), *flags]
    assert module.main() == 0
baseline = root / "artifacts/s4-motion-coverage-pilot-001/uniform"
old_report, new_report = read(baseline / "receipt.json"), read(out / "uniform/receipt.json")
assert old_report["status"] == new_report["status"] == "completed"
for key in old_report["identity"]:
    if key not in ["adapter_sha256", "adapter_config_sha256"]:
        assert old_report["identity"][key] == new_report["identity"][key], key
old = [json.loads(x) for x in (baseline / "records.jsonl").read_text().splitlines()]
new = [json.loads(x) for x in (out / "uniform/records.jsonl").read_text().splitlines()]
assert len(old) == len(new) == 50
for a, b in zip(old, new, strict=True):
    for key in ["id", "input_sha256", "input_summary_sha256", "answer_token_ids"]:
        assert a[key] == b[key], (a["id"], key)
report = {
    "status": "completed",
    "stage_gate": False,
    "scope": "Fixed temporal DEV50 after fresh-family coverage; screening only",
    "baseline": old_report["metrics"],
    "candidate": new_report["metrics"],
    "paired_input_hashes_verified": 50,
    "before_adapter": old_report["identity"]["adapter_sha256"],
    "after_adapter": trained["adapter_sha256"],
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "baseline_receipt_sha256": hashlib.sha256((baseline / "receipt.json").read_bytes()).hexdigest(),
    "candidate_receipt_sha256": hashlib.sha256(
        (out / "uniform/receipt.json").read_bytes()
    ).hexdigest(),
}
(root / "reports/s4-temporal-coverage-dev50-001.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
