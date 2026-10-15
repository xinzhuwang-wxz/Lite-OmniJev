"""One completed pilot arm, 100 paired public-dev originals, bounded to 180s."""

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
parser.add_argument("--arm", choices=("ce0", "pair01"), required=True)
args = parser.parse_args()
ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
MODULE = Path("/root/autodl-tmp/s4-cross-batch-code-001/evaluate-coverage-crossmodal-full.py")
spec = importlib.util.spec_from_file_location("cross_batch", MODULE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output = ROOT / f"artifacts/s4-claim-pair-{args.arm}-original100-001"
output.mkdir(exist_ok=False)
start = time.monotonic()
report = {
    "status": "failed",
    "exit_code": 1,
    "arm": args.arm,
    "stage_gate": False,
    "budget_seconds": 180,
    "source": "original",
    "count": 100,
}


def interrupted(signum, frame):
    raise TimeoutError(f"evaluation interrupted {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGALRM, interrupted)
try:
    train_path = ROOT / f"reports/s4-claim-pair-{args.arm}-training-001.json"
    train = json.loads(train_path.read_text())
    assert (
        train["status"] == "completed"
        and train["exit_code"] == 0
        and train["steps_completed"] == 10
    )
    assert train["config"]["sample_count"] == 120 and train["config"]["accumulation"] == 12
    assert train["config"].get("pair_loss_weight", 0) == (0 if args.arm == "ce0" else 0.1)
    assert (
        train["init_adapter"]["weights_sha256"]
        == "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818"
    )
    prep = ROOT / "artifacts/s4-claim-pair-pilot-001"
    assert (
        module.sha(prep / "train.jsonl")
        == "72c9bbf0b3cd8b931feeb4ce3b63128634d714dc1caadeb2117a27aa583e81fe"
    )
    samples = [
        Sample.model_validate_json(line) for line in (prep / "train.jsonl").read_text().splitlines()
    ]
    assert hashlib.sha256(data_bytes(samples)).hexdigest() == train["data_sha256"]
    adapter = ROOT / f"artifacts/s4-claim-pair-{args.arm}-train-001/adapter"
    assert module.sha(adapter / "adapter_model.safetensors") == train["adapter_sha256"]
    data, media = module.cross_source("original")
    selected = module.full_samples("original", data)[:100]
    paired_path = ROOT / "artifacts/s4-cross-batch100-original-coverage-001/records.jsonl"
    assert (
        module.sha(paired_path)
        == "580c7d7155fb083080a7273510b5640bf00640e563e935a1204c2093239bb3be"
    )
    rows = [json.loads(x) for x in paired_path.read_text().splitlines()]
    paired = {r["id"]: r for r in rows}
    assert len(paired) == len(rows) == 100 and set(paired) == {s.id for s in selected}
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
    report["paired_inputs_verified"] = 100
except Exception as error:
    report.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}")
finally:
    signal.alarm(0)
    report["duration_seconds"] = time.monotonic() - start
    report["records_sidecar"] = module.baseline.sidecar_evidence(output / "records.jsonl")
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
raise SystemExit(report["exit_code"])
