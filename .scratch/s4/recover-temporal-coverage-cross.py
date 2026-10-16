"""Recover complete stored cross100 records after a final metadata path failure."""

import importlib.util
import json
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
path = Path("/root/autodl-tmp/s4-cross-batch-code-001/evaluate-coverage-crossmodal-full.py")
spec = importlib.util.spec_from_file_location("cross", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
out = root / "artifacts/s4-temporal-coverage-original100-001"
failed = json.loads((out / "receipt.json").read_text())
assert failed["status"] == "failed" and failed["exit_code"] == 1
assert failed["error"] == (
    "FileNotFoundError: [Errno 2] No such file or directory: 'scripts/s0_development_baselines.py'"
)
assert module.sha(out / "records.jsonl") == failed["records_sidecar"]["sha256"]
assert module.sha(path) == failed["evaluator_sha256"]
old_path = root / "artifacts/s4-claim-pair-ce0-original100-001/records.jsonl"
assert module.sha(old_path) == failed["paired_records_sha256"]
old = [json.loads(line) for line in old_path.read_text().splitlines()]
rows = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
data, _ = module.cross_source("original")
samples = module.full_samples("original", data)[:100]
assert len(rows) == len(old) == len(samples) == 100
assert module.sha(data) == failed["data_sha256"]
for sample, before, after in zip(samples, old, rows, strict=True):
    assert sample.id == before["id"] == after["id"]
    for key in ["input_sha256", "input_summary_sha256", "answer_token_ids", "target_index"]:
        assert before[key] == after[key], (sample.id, key)
metrics = module.baseline.score_records(rows)
progress = json.loads((out / "progress.json").read_text())
assert progress["completed"] == progress["total"] == metrics["count"] == 100
assert progress["metrics"] == metrics
adapter = root / "artifacts/s4-temporal-coverage-train-001/adapter/adapter_model.safetensors"
assert module.sha(adapter) == failed["adapter_sha256"]
source = Path(module.baseline.__file__).resolve().parent.parent
result = {
    "status": "recovered_completed",
    "exit_code": 0,
    "stage_gate": False,
    "metrics": metrics,
    "paired_inputs_verified": 100,
    "original_receipt_sha256": module.sha(out / "receipt.json"),
    "records_sha256": module.sha(out / "records.jsonl"),
    "progress_sha256": module.sha(out / "progress.json"),
    "adapter_sha256": module.sha(adapter),
    "data_sha256": module.sha(data),
    "observed_job_wall_seconds": failed["duration_seconds"],
    "code_sha256_revalidated": {
        p: module.sha(source / p) for p in module.baseline.REQUIRED_CODE_FILES
    },
    "recovery_scope": (
        "CPU rescore/revalidation only; original failed receipt preserved; no GPU rerun"
    ),
    "gpu_peak_metadata": None,
    "recovery_script_sha256": module.sha(Path(__file__)),
}
with (out / "receipt-recovered-002.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
