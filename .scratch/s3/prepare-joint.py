"""Prepare S3 frozen train union and balanced probe; never read reserved samples."""

import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

from data.schema import Sample
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig

root = Path("/root/autodl-tmp/s1-text-lora-001")
started = time.time()
cohorts = {
    "text": "s1-text-pools-v1-001",
    "image": "s2-media-pools-image-001",
    "screenshot": "s2-media-pools-screenshot-001",
    "audio": "s3-pools-audio-001",
    "video": "s3-pools-video-001",
}
expected_counts = dict(zip(cohorts, [10750, 1389, 420, 326, 8299], strict=True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(rows):
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()


inputs, ledgers, rows, probe, versions = {}, [], [], [], []
for cohort, directory in cohorts.items():
    source = root / "artifacts" / directory / "run-a"
    data = source / "train.jsonl"
    ledger = source / "allocation-ledger.jsonl"
    evidence = json.loads((source / "report.json").read_text())
    assert evidence["status"] == "passed"
    assert sha(data) == evidence["outputs"]["train_jsonl_sha256"]
    assert sha(ledger) == evidence["outputs"]["allocation_ledger_sha256"]
    assert sha(data) == sha(source.parent / "run-b/train.jsonl")
    current = [json.loads(line) for line in data.read_text().splitlines() if line]
    assert len(current) == expected_counts[cohort]
    for row in current:
        sample = Sample.model_validate(row)
        assert sample.meta.pool == "train"
        assert sample.meta.modality == [("image" if cohort == "screenshot" else cohort)]
    versions.extend(sorted({r["meta"]["data_version"] for r in current}))
    inputs[cohort] = {
        "path": str(data),
        "sha256": sha(data),
        "count": len(current),
        "ledger_sha256": sha(ledger),
        "report_sha256": sha(source / "report.json"),
    }
    ledgers.append(ledger)
    rows.extend(current)
    probe.extend(
        sorted(
            current, key=lambda r: hashlib.sha256(("s3-probe:42:" + r["id"]).encode()).hexdigest()
        )[:100]
    )
assert len({r["id"] for r in rows}) == len(rows) == 21184
random.Random(42).shuffle(rows)
random.Random(42).shuffle(probe)
output = root / "artifacts/s3-joint-pool-001"
output.mkdir(exist_ok=True)
for name, selected in [("joint-train.jsonl", rows), ("probe-500.jsonl", probe)]:
    target = output / name
    payload = encode(selected)
    if target.exists():
        assert target.read_bytes() == payload, "refuse to overwrite different frozen data"
    else:
        target.write_bytes(payload)
    verify_data_is_train_pool(target, ledgers)
config = {
    "seed": 42,
    "sample_count": 21184,
    "steps": 1000,
    "accumulation": 10,
    "max_length": 4096,
    "learning_rate": 0.0001,
    "rank": 16,
    "alpha": 32,
    "dtype": "bfloat16",
    "modality": "mixed",
    "video_num_frames": 16,
    "midway_step": 500,
    "checkpoint_every": 100,
    "expected_data_versions": versions,
    "media_roots": [
        (
            "/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources/aokvqa/co"
            "co-ccby-publicdomain-images"
        ),
        (
            "/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/gui-odyssey-source-audit"
            "/media/app_split_train_under19_5gb/screenshots"
        ),
        (
            "/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207/appro"
            "ved-audio-002"
        ),
        "/root/autodl-tmp/s3-media-candidates-001/video/media",
    ],
}
for name, cfg in [
    ("main", config),
    (
        "probe",
        {**config, "sample_count": 500, "steps": 30, "midway_step": 15, "checkpoint_every": 15},
    ),
]:
    LoopConfig.model_validate(cfg)
    (output / f"{name}-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
report = {
    "status": "passed",
    "exit_code": 0,
    "scope": (
        "CPU pool/config preparation only; S2 prerequisite remains; no S3 training or med"
        "ia forward acceptance"
    ),
    "inputs": inputs,
    "total_rows": len(rows),
    "probe_rows": len(probe),
    "probe_per_cohort": 100,
    "ledger_check": verify_data_is_train_pool(output / "joint-train.jsonl", ledgers),
    "joint_sha256": sha(output / "joint-train.jsonl"),
    "probe_sha256": sha(output / "probe-500.jsonl"),
    "config_hashes": {name: sha(output / f"{name}-config.json") for name in ["main", "probe"]},
    "script_sha256": sha(Path(__file__)),
    "code_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "seed": 42,
    "holdout_content_accessed": False,
    "gpu_used": False,
    "cost": None,
    "duration_seconds": time.time() - started,
}
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
