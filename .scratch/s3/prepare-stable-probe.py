"""Prepare a bounded S3 replay pilot; this script never starts GPU training."""

import hashlib
import json
import random
from pathlib import Path

from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig

root = Path("/root/autodl-tmp/s1-text-lora-001")
output = root / "artifacts/s3-stable-probe-001"
output.mkdir(exist_ok=False)
cohorts = {
    "text": "s1-text-pools-v1-001",
    "image": "s2-media-pools-image-001",
    "screenshot": "s2-media-pools-screenshot-001",
    "audio": "s3-pools-audio-001",
    "video": "s3-pools-video-001",
}
selected, ledgers, inputs = [], [], {}
for cohort, directory in cohorts.items():
    source = root / "artifacts" / directory / "run-a"
    path = source / "train.jsonl"
    frozen = json.loads((source / "report.json").read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen["status"] == "passed" and sha == frozen["outputs"]["train_jsonl_sha256"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(row["meta"]["pool"] == "train" for row in rows)
    subset = sorted(
        rows, key=lambda r: hashlib.sha256(("s3-probe:42:" + r["id"]).encode()).hexdigest()
    )[:200]
    assert len(subset) == 200
    selected.extend(subset)
    ledgers.append(source / "allocation-ledger.jsonl")
    inputs[cohort] = {"train_sha256": sha, "selected": len(subset)}
random.Random(42).shuffle(selected)
assert len({row["id"] for row in selected}) == 1000
path = output / "train.jsonl"
path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in selected))
check = verify_data_is_train_pool(path, ledgers)
config = json.loads((root / "artifacts/s3-joint-pool-001/main-config.json").read_text())
config.update(
    sample_count=1000,
    steps=100,
    accumulation=10,
    learning_rate=2e-5,
    midway_step=50,
    checkpoint_every=50,
    preparation_mode="lazy",
    preparation_cache_bytes=134217728,
    image_max_soft_tokens=1120,
)
LoopConfig.model_validate(config)
(output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
report = {
    "status": "prepared",
    "training_started": False,
    "requires_s2_acceptance": True,
    "scope": "CPU-only balanced train-pool preparation, not stage acceptance",
    "inputs": inputs,
    "rows": 1000,
    "ledger_check": check,
    "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
    "ledgers": [str(p) for p in ledgers],
    "init_adapter_if_s2_passes": "artifacts/s2-text-lora-001/joint-v2-001/adapter",
    "init_adapter_sha256": "db45fb13e651acf07ed5901584960f7bfa60abb1f1a26c609a6a89511c35a4ee",
}
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "reports/s3-stable-probe-preparation-001.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
