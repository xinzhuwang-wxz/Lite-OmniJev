"""CPU-only S4 joint pilot preparation; never launches model or GPU commands.

Run from the repository with --root ABS --model-dir ABS --asset-manifest ABS.
Entry: uv run python .scratch/s4/prepare-joint-pilot.py
The existing trainer cannot pause at midway_step or extend a checkpoint config.
Only a 30-step candidate is runnable; evaluate it before deciding continuation.
GNU timeout in the emitted foreground command supplies the 1800-second limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, smoke_samples, verified_init_adapter

ADAPTER_SHA = "1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755"
VERSION = "crossmodal-train-pilot-v1"
COHORTS = {
    "text": "s1-text-pools-v1-001",
    "image": "s2-media-pools-image-001",
    "screenshot": "s2-media-pools-screenshot-001",
    "audio": "s3-pools-audio-001",
    "video": "s3-pools-video-001",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, category: str) -> None:
    if not condition:
        raise ValueError(category)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, payload: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)


def prepare(root: Path, output: Path, *, model_dir: Path, asset_manifest: Path) -> dict[str, Any]:
    started = time.monotonic()
    require(model_dir.is_absolute() and asset_manifest.is_absolute(), "absolute_model_paths")
    require(root.is_absolute() and output.is_absolute(), "absolute_paths_required")
    require(not output.exists(), "refuse_existing_output")
    stable = root / "artifacts/s3-stable-probe-001"
    stable_report_path = stable / "report.json"
    stable_report = json.loads(stable_report_path.read_bytes())
    stable_data, stable_config = stable / "train.jsonl", stable / "config.json"
    require(stable_report["status"] == "prepared", "stable_not_prepared")
    require(sha(stable_data) == stable_report["data_sha256"], "stable_data_sha")
    require(sha(stable_config) == stable_report["config_sha256"], "stable_config_sha")
    old_rows = rows(stable_data)
    require(len(old_rows) == 1000, "stable_count")
    ledgers, input_hashes, pools = [], {}, {}
    for cohort, directory in COHORTS.items():
        source = root / "artifacts" / directory / "run-a"
        ledger = source / "allocation-ledger.jsonl"
        report_path = source / "report.json"
        evidence = json.loads(report_path.read_bytes())
        require(evidence["status"] == "passed", "source_report")
        require(sha(ledger) == evidence["outputs"]["allocation_ledger_sha256"], "ledger_sha")
        ledger_rows = rows(ledger)
        ids = {r["item_id"] for r in ledger_rows if r["pool"] == "train"}
        cohort_rows = [r for r in old_rows if r["id"] in ids]
        require(len(cohort_rows) == 200, "replay_cohort_count")
        require(
            all(
                r["meta"]["modality"] == ["image" if cohort == "screenshot" else cohort]
                for r in cohort_rows
            ),
            "replay_physical_modality",
        )
        for row in cohort_rows:
            require(row["id"] not in pools, "cohort_membership_collision")
            pools[row["id"]] = cohort
        ledgers.append(ledger)
        input_hashes[cohort] = {"ledger_sha256": sha(ledger), "report_sha256": sha(report_path)}
    require(len(pools) == 1000, "unclassified_replay")
    cross = root / "artifacts/s4-crossmodal-train-001"
    cross_report_path = cross / "provenance.json"
    cross_report = json.loads(cross_report_path.read_bytes())
    require(
        cross_report["schema_version"] == VERSION and cross_report["count"] == 200,
        "crossmodal_provenance",
    )
    for key, name in (
        ("train", "train.jsonl"),
        ("allocation", "allocation.jsonl"),
        ("components", "components.jsonl"),
        ("inventory", "inventory-manifest.json"),
    ):
        require(sha(cross / name) == cross_report["output_sha256"][key], "crossmodal_output_sha")
    cross_rows = rows(cross / "train.jsonl")
    require(len(cross_rows) == 200, "crossmodal_count")
    for row in cross_rows:
        sample = Sample.model_validate(row)
        require(
            sample.meta.modality == ["image", "audio"] and sample.meta.data_version == VERSION,
            "crossmodal_contract",
        )
        require(sample.id not in pools, "duplicate_component_id")
        pools[sample.id] = "crossmodal"
        for media in sample.prefix.media:
            relative = Path(media.uri)
            require(
                not relative.is_absolute() and ".." not in relative.parts, "crossmodal_media_path"
            )
            require(sha(cross / relative) == media.sha256, "crossmodal_media_sha")
    ledgers.append(cross / "allocation.jsonl")
    selected = old_rows + cross_rows
    require(len({r["id"] for r in selected}) == 1200, "duplicate_sample_id")
    require(
        all(Sample.model_validate(r).meta.pool == "train" for r in selected), "non_train_sample"
    )
    # Invert the trainer's seed42 permutation to retain strict six-way draws.
    # Each six-draw round includes every cohort once; the first30 steps draw50/class.
    rng = random.Random(42)
    buckets = {
        cohort: [row for row in selected if pools[row["id"]] == cohort]
        for cohort in [*COHORTS, "crossmodal"]
    }
    for bucket in buckets.values():
        rng.shuffle(bucket)
    desired: list[dict[str, Any]] = []
    for index in range(200):
        cohort_order = list(buckets)
        rng.shuffle(cohort_order)
        desired.extend(buckets[cohort][index] for cohort in cohort_order)
    trainer_order = list(range(1200))
    random.Random(42).shuffle(trainer_order)
    for index, input_index in enumerate(trainer_order):
        selected[input_index] = desired[index]
    cohort_counts = Counter(pools[r["id"]] for r in selected)
    require(cohort_counts == dict.fromkeys([*COHORTS, "crossmodal"], 200), "six_way_balance")
    config = json.loads(stable_config.read_bytes())
    roots = list(
        config.get("media_roots") or ([config["media_root"]] if config.get("media_root") else [])
    )
    require(bool(roots), "missing_stable_media_roots")
    config.pop("media_root", None)
    if str(cross) not in roots:
        roots.append(str(cross))
    config.update(
        sample_count=1200,
        steps=30,
        accumulation=10,
        learning_rate=2e-5,
        midway_step=30,
        checkpoint_every=30,
        seed=42,
        modality="mixed",
        media_roots=roots,
        expected_data_versions=sorted({r["meta"]["data_version"] for r in selected}),
    )
    config.pop("expected_data_version", None)
    validated = LoopConfig.model_validate(config)
    adapter = root / "artifacts/s3-stable-probe-train-001/adapter"
    adapter_identity = verified_init_adapter(adapter, ADAPTER_SHA)
    output.mkdir(mode=0o700, parents=True)
    data_path = output / "train.jsonl"
    write(
        data_path,
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected),
    )
    # Preserve the frozen rows, including labels and pool, without normalizing source data.
    require(
        {r["id"]: r for r in rows(data_path)} == {r["id"]: r for r in old_rows + cross_rows},
        "sample_mutation",
    )
    ledger_check = verify_data_is_train_pool(data_path, ledgers)
    loaded = smoke_samples(
        validated, data_path, expected_data_versions=config["expected_data_versions"]
    )
    require(len(loaded) == 1200, "trainer_intake")
    # Keep complete frozen ledgers so conflicting pool declarations remain visible.
    ledger_path = output / "allocation-ledger.jsonl"
    write(ledger_path, "".join(path.read_text().rstrip("\n") + "\n" for path in ledgers))
    verify_data_is_train_pool(data_path, [ledger_path])
    config_path = output / "config-30.json"
    write(config_path, json.dumps(config, indent=2) + "\n")
    run_dir = root / "artifacts/s4-joint-pilot-train-001"
    common = [
        "--config",
        str(config_path),
        "--model-dir",
        str(model_dir),
        "--asset-manifest",
        str(asset_manifest),
        "--data",
        str(data_path),
        "--allocation-ledger",
        str(ledger_path),
        "--output-dir",
        str(run_dir),
        "--report",
        str(root / "reports/s4-joint-pilot-train-001.json"),
        "--init-adapter",
        str(adapter),
        "--init-adapter-sha256",
        ADAPTER_SHA,
    ]
    order = list(range(1200))
    random.Random(42).shuffle(order)
    draws = Counter(pools[selected[i]["id"]] for i in order[:300])
    require(draws == dict.fromkeys([*COHORTS, "crossmodal"], 50), "draw_balance")
    report = {
        "status": "prepared",
        "exit_code": 0,
        "training_started": False,
        "gpu_used": False,
        "scope": "CPU preparation; 30-step candidate before evaluation, not stage acceptance",
        "planned_optimizer_steps": 60,
        "candidate_optimizer_steps": 30,
        "accumulation": 10,
        "wallclock_limit_seconds": 1800,
        "wallclock_mechanism": (
            "foreground GNU timeout --signal=INT --kill-after=30s; trainer has no native "
            "wallclock field"
        ),
        "pause_resume": (
            "midway_step saves then continues; strict config checkpoint identity prevents "
            "changing steps 30 to 60; no 60-step runnable config emitted"
        ),
        "continuation": (
            "Evaluate the completed 30-step adapter first. Resume with the same 30-step "
            "config cannot extend training; a second initialized candidate would reset "
            "optimizer and needs an explicit execution decision."
        ),
        "balance": dict(cohort_counts),
        "first_30_step_draw_counts": dict(draws),
        "balance_scope": (
            "200 rows per cohort; inverse trainer shuffle gives 50 draws/class in30 steps"
        ),
        "ledger_check": ledger_check,
        "init_adapter": adapter_identity,
        "input_sha256": {
            **input_hashes,
            "stable_data": sha(stable_data),
            "stable_config": sha(stable_config),
            "stable_report": sha(stable_report_path),
            "crossmodal_provenance": sha(cross_report_path),
            "crossmodal_train": sha(cross / "train.jsonl"),
            "crossmodal_allocation": sha(cross / "allocation.jsonl"),
        },
        "output_sha256": {
            "train": sha(data_path),
            "config": sha(config_path),
            "allocation": sha(ledger_path),
        },
        "model_asset_preflight": "not_run; execute emitted preflight before GPU run",
        "preflight_argv": ["uv", "run", "python", "-m", "scripts.train_s1", "preflight", *common],
        "foreground_run_argv": [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            "1800s",
            "uv",
            "run",
            "python",
            "-m",
            "scripts.train_s1",
            "run",
            *common,
        ],
        "monitoring": (
            "SwanLab uses existing environment; run_dir/run.json updates every optimizer "
            "step, with losses/steps_completed; checkpoint and midway adapter are saved "
            "at30"
        ),
        "script_sha256": sha(Path(__file__)),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or "unavailable",
        "intake_code_sha256": sha(Path("train/trainer.py")),
        "python": platform.python_version(),
        "duration_seconds": time.monotonic() - started,
        "cost": None,
    }
    write(output / "report.json", json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.root,
        args.output_dir or args.root / "artifacts/s4-joint-pilot-001",
        model_dir=args.model_dir,
        asset_manifest=args.asset_manifest,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "balance": result["balance"],
                "candidate_optimizer_steps": 30,
                "training_started": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
