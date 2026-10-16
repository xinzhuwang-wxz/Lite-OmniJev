"""One pilot arm: fixed original100 or supplemental50 dev, bounded to 180s."""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import time
from pathlib import Path

from data.schema import Sample
from train.trainer import data_bytes

parser = argparse.ArgumentParser()
parser.set_defaults(arm="coverage")
parser.add_argument("--source", choices=("original", "supplemental"), default="original")
args = parser.parse_args()
ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
os.chdir("/root/autodl-tmp/s4-claim-pair-code-001")
MODULE = Path("/root/autodl-tmp/s4-cross-batch-code-001/evaluate-coverage-crossmodal-full.py")
spec = importlib.util.spec_from_file_location("cross_batch", MODULE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
count = 100 if args.source == "original" else 50
output = ROOT / f"artifacts/s4-temporal-coverage-{args.source}{count}-001"
output.mkdir(exist_ok=False)
start = time.monotonic()
report = {
    "status": "failed",
    "exit_code": 1,
    "arm": args.arm,
    "stage_gate": False,
    "budget_seconds": 180,
    "source": args.source,
    "count": count,
}


def interrupted(signum, frame):
    raise TimeoutError(f"evaluation interrupted {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGALRM, interrupted)
try:
    train_path = ROOT / "reports/s4-temporal-coverage-training-001.json"
    train = json.loads(train_path.read_text())
    assert (
        train["status"] == "completed"
        and train["exit_code"] == 0
        and train["steps_completed"] == 10
    )
    assert train["config"]["sample_count"] == 120 and train["config"]["accumulation"] == 12
    assert train["config"].get("pair_loss_weight", 0) == 0
    assert (
        train["init_adapter"]["weights_sha256"]
        == "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"
    )
    prep = ROOT / "artifacts/s4-temporal-coverage-pilot-001"
    assert (
        module.sha(prep / "train.jsonl")
        == "276afa595a42379dd01c50353dc8e0ad253d43c70367682b59775d8f9ced730f"
    )
    samples = [
        Sample.model_validate_json(line) for line in (prep / "train.jsonl").read_text().splitlines()
    ]
    assert hashlib.sha256(data_bytes(samples)).hexdigest() == train["data_sha256"]
    adapter = ROOT / "artifacts/s4-temporal-coverage-train-001/adapter"
    assert module.sha(adapter / "adapter_model.safetensors") == train["adapter_sha256"]
    data, media = module.cross_source(args.source)
    selected = module.full_samples(args.source, data)[:count]
    if args.source == "original":
        paired_path = ROOT / "artifacts/s4-claim-pair-ce0-original100-001/records.jsonl"
        expected_paired_sha = "2c4d954003d40cd9155136a640ffd6d3ed288c2bf17dfd361fd0ec9546532217"
    else:
        paired_path = ROOT / "artifacts/s4-cross-batch50-supplemental-coverage-001/records.jsonl"
        expected_paired_sha = "899f2326c3388964cda4afc38846b7a2cf3982676eab0a0c07fe3f22be02d660"
    assert module.sha(paired_path) == expected_paired_sha
    rows = [json.loads(x) for x in paired_path.read_text().splitlines()]
    paired = {r["id"]: r for r in rows}
    assert len(paired) == len(rows) == count and set(paired) == {s.id for s in selected}
    report.update(
        training_report_sha256=module.sha(train_path),
        data_sha256=module.sha(data),
        paired_records_sha256=module.sha(paired_path),
        runner_sha256=module.sha(Path(__file__)),
        evaluator_sha256=module.sha(MODULE),
        adapter_sha256=train["adapter_sha256"],
    )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    signal.alarm(180)
    report.update(module.full_forward(selected, media, adapter, output, paired))
    report["paired_inputs_verified"] = count
except Exception as error:
    report.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}")
finally:
    signal.alarm(0)
    report["duration_seconds"] = time.monotonic() - start
    report["records_sidecar"] = module.baseline.sidecar_evidence(output / "records.jsonl")
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
raise SystemExit(report["exit_code"])
