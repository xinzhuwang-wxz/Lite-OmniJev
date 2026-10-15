"""CPU-only bounded video exposure pilot; no GPU or reserved-pool inspection.

Full inventory is passed to the existing aggregate audit without exposing records.
Prior overlapping current TRAIN artifacts become historical in the new manifest,
otherwise audit_manifest rejects duplicated current IDs. Original files stay intact.
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

from data.schema import HardTarget, Sample
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples, verified_init_adapter

INIT_SHA = "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac"
COHORTS = {
    "text": "s1-text-pools-v1-001",
    "image": "s2-media-pools-image-001",
    "screenshot": "s2-media-pools-screenshot-001",
    "audio": "s3-pools-audio-001",
    "video": "s3-pools-video-001",
}
QUOTAS = {"temporal": 90, "motion": 30, "existence": 30}


def require(value: bool, category: str) -> None:
    if not value:
        raise ValueError(category)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")


def kind(row: dict[str, Any]) -> str:
    words = set(row["questions"][0]["instruction"].lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else "motion"
        if words & {"moving", "stationary"}
        else "existence"
    )


def rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(("s4-video-pilot:42:" + row["id"]).encode()).hexdigest()


def choose_video(
    candidates: list[dict[str, Any]], seen: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ids = {r["id"] for r in seen}
    families = {r["meta"]["family_id"] for r in seen}
    media = {m["sha256"] for r in seen for m in r["prefix"]["media"]}
    eligible = []
    for row in candidates:
        sample = Sample.model_validate(row)
        require(
            sample.meta.pool == "train"
            and sample.meta.modality == ["video"]
            and len(sample.prefix.media) == 1
            and sample.prefix.media[0].type == "video"
            and len(sample.questions) == 1,
            "video_contract",
        )
        q = sample.questions[0]
        require(
            q.type == "noul"
            and isinstance(q.target, HardTarget)
            and q.target.value in {"yes", "no"},
            "video_target",
        )
        if (
            sample.id not in ids
            and sample.meta.family_id not in families
            and sample.prefix.media[0].sha256 not in media
        ):
            eligible.append(row)
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    used_families: set[str] = set()
    used_media: set[str] = set()
    # One deterministic global hash pass avoids quota-order preference for shared media.
    for row in sorted(eligible, key=rank):
        group, target = kind(row), row["questions"][0]["target"]["value"]
        family, physical = row["meta"]["family_id"], row["prefix"]["media"][0]["sha256"]
        if (
            counts[(group, target)] >= QUOTAS[group] // 2
            or family in used_families
            or physical in used_media
        ):
            continue
        selected.append(row)
        used_families.add(family)
        used_media.add(physical)
        counts[(group, target)] += 1
    require(
        all(
            counts[(group, target)] == count // 2
            for group, count in QUOTAS.items()
            for target in ("yes", "no")
        ),
        "unique_video_quota_capacity",
    )
    return selected, {
        "input": len(candidates),
        "eligible_after_lineage_id_family_media": len(eligible),
        "selected": len(selected),
    }


def arrange(buckets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    require(
        len(buckets["video"]) == 150
        and all(len(values) == 30 for key, values in buckets.items() if key != "video"),
        "draw_bucket_counts",
    )
    rng = random.Random(42)
    for values in buckets.values():
        rng.shuffle(values)
    desired = []
    for step in range(30):
        batch = buckets["video"][step * 5 : step * 5 + 5] + [
            values[step] for key, values in buckets.items() if key != "video"
        ]
        rng.shuffle(batch)
        desired.extend(batch)
    order = list(range(300))
    random.Random(42).shuffle(order)
    result = list(desired)
    for draw, index in enumerate(order):
        result[index] = desired[draw]
    require([result[index] for index in order] == desired, "trainer_permutation")
    return result


def prepare(root: Path, output: Path, *, model_dir: Path, asset_manifest: Path) -> dict[str, Any]:
    started = time.monotonic()
    require(
        all(p.is_absolute() for p in (root, output, model_dir, asset_manifest)), "absolute_paths"
    )
    require(not output.exists(), "output_exists")
    output.mkdir(mode=0o700, parents=True)
    try:
        train_report_path = root / "reports/s4-joint-pilot-train-001.json"
        train_report = json.loads(train_report_path.read_bytes())
        require(
            train_report["status"] == "completed"
            and train_report["exit_code"] == 0
            and train_report["steps_completed"] == 30
            and train_report["adapter_sha256"] == INIT_SHA,
            "first_pilot_completion",
        )
        require(model_dir.name == train_report["model_revision"], "frozen_model_revision")
        adapter = root / "artifacts/s4-joint-pilot-train-001/adapter"
        identity = verified_init_adapter(adapter, INIT_SHA)
        previous = root / "artifacts/s4-joint-pilot-001"
        previous_report = json.loads((previous / "report.json").read_bytes())
        for key, filename in (
            ("train", "train.jsonl"),
            ("config", "config-30.json"),
            ("allocation", "allocation-ledger.jsonl"),
        ):
            require(
                sha(previous / filename) == previous_report["output_sha256"][key],
                "previous_preparation_sha",
            )
        replay = rows(previous / "train.jsonl")
        require(len(replay) == 1200 and len({r["id"] for r in replay}) == 1200, "previous_count")
        previous_samples = [Sample.model_validate(row) for row in replay]
        require(
            hashlib.sha256(data_bytes(previous_samples)).hexdigest() == train_report["data_sha256"],
            "first_training_data_sha",
        )
        stable = root / "artifacts/s3-stable-probe-001"
        stable_report = json.loads((stable / "report.json").read_bytes())
        require(sha(stable / "train.jsonl") == stable_report["data_sha256"], "stable_train_sha")
        require(
            sha(stable / "report.json") == previous_report["input_sha256"]["stable_report"],
            "stable_report_binding",
        )
        seen = [row for row in rows(stable / "train.jsonl") if row["meta"]["modality"] == ["video"]]
        require(
            len(seen) == 200 and all(r["meta"]["pool"] == "train" for r in seen),
            "known_lineage_video_count",
        )
        prior_order = list(range(1200))
        random.Random(42).shuffle(prior_order)
        seen_ids = {row["id"] for row in seen}
        require(
            all(
                replay[index]["id"] in seen_ids
                for index in prior_order[:300]
                if replay[index]["meta"]["modality"] == ["video"]
            ),
            "first_pilot_video_lineage",
        )
        paths, membership, input_hashes = [], {}, {}
        for cohort, directory in COHORTS.items():
            source = root / "artifacts" / directory / "run-a"
            source_report = json.loads((source / "report.json").read_bytes())
            ledger = source / "allocation-ledger.jsonl"
            require(
                source_report["status"] == "passed"
                and sha(ledger) == source_report["outputs"]["allocation_ledger_sha256"],
                "source_ledger_sha",
            )
            paths.append(ledger)
            for row in rows(ledger):
                if row["pool"] == "train":
                    require(row["item_id"] not in membership, "source_membership_collision")
                    membership[row["item_id"]] = cohort
            input_hashes[cohort] = {"report": sha(source / "report.json"), "ledger": sha(ledger)}
        video_source = root / "artifacts/s3-pools-video-001/run-a"
        video_report = json.loads((video_source / "report.json").read_bytes())
        require(
            sha(video_source / "train.jsonl") == video_report["outputs"]["train_jsonl_sha256"],
            "video_source_sha",
        )
        video_rows = rows(video_source / "train.jsonl")
        require(len(video_rows) == 8299, "video_source_count")
        videos, capacity = choose_video(video_rows, seen)
        cross = root / "artifacts/s4-crossmodal-train-001"
        require(
            sha(cross / "provenance.json")
            == previous_report["input_sha256"]["crossmodal_provenance"],
            "previous_crossmodal_binding",
        )
        cross_provenance = json.loads((cross / "provenance.json").read_bytes())
        for key, name in (
            ("allocation", "allocation.jsonl"),
            ("inventory", "inventory-manifest.json"),
        ):
            require(
                sha(cross / name) == cross_provenance["output_sha256"][key], "crossmodal_binding"
            )
        paths.append(cross / "allocation.jsonl")
        for row in rows(cross / "allocation.jsonl"):
            require(
                row["pool"] == "train" and row["item_id"] not in membership, "crossmodal_ledger"
            )
            membership[row["item_id"]] = "crossmodal"
        buckets = {"video": videos}
        for cohort in ("text", "image", "screenshot", "audio", "crossmodal"):
            pool = [row for row in replay if membership.get(row["id"]) == cohort]
            require(len(pool) == 200, "replay_cohort_count")
            buckets[cohort] = sorted(pool, key=rank)[:30]
        selected = arrange(buckets)
        require(len({r["id"] for r in selected}) == 300, "duplicate_selection")
        require(all(Sample.model_validate(r).meta.pool == "train" for r in selected), "non_train")
        data_path = output / "train.jsonl"
        write(
            data_path,
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected),
        )
        require(rows(data_path) == selected, "sample_mutation")
        ledger_path = output / "allocation-ledger.jsonl"
        write(ledger_path, "".join(path.read_text().rstrip("\n") + "\n" for path in paths))
        ledger_check = verify_data_is_train_pool(data_path, [ledger_path])
        config = json.loads((previous / "config-30.json").read_bytes())
        config.update(
            sample_count=300,
            steps=30,
            accumulation=10,
            learning_rate=2e-5,
            seed=42,
            checkpoint_every=15,
            midway_step=15,
            expected_data_versions=sorted({r["meta"]["data_version"] for r in selected}),
        )
        require(
            config["image_max_soft_tokens"] == 1120
            and config["video_num_frames"] == 16
            and config["preparation_mode"] == "lazy",
            "frozen_budget",
        )
        config.pop("expected_data_version", None)
        validated = LoopConfig.model_validate(config)
        smoke_samples(validated, data_path, expected_data_versions=config["expected_data_versions"])
        write(output / "config-30.json", config)
        inventory_path = cross / "inventory-manifest.json"
        inventory = Manifest.model_validate_json(inventory_path.read_bytes())
        # Duplicate current IDs cannot coexist. Preserve old full artifacts as historical.
        selected_ids = {row["id"] for row in selected}
        superseded = []
        version_cohorts: dict[str, str] = {}
        for row in selected:
            cohort = "video" if row["meta"]["modality"] == ["video"] else membership[row["id"]]
            version = row["meta"]["data_version"]
            require(
                version not in version_cohorts or version_cohorts[version] == cohort,
                "data_version_cohort_conflict",
            )
            version_cohorts[version] = cohort
        combined = inventory.model_dump(mode="json", exclude_none=True)
        for index, artifact in enumerate(inventory.samples):
            if artifact.expected_pool == "train" and artifact.exposure == "current":
                overlap = any(
                    json.loads(line)["id"] in selected_ids for line in bound_lines(artifact)
                )
                if overlap:
                    combined["samples"][index]["exposure"] = "historical"
                    superseded.append({"path": artifact.path, "sha256": artifact.sha256})
        combined["samples"].append(
            {
                "path": str(data_path),
                "sha256": sha(data_path),
                "cohorts_by_data_version": version_cohorts,
                "exposure": "current",
                "expected_pool": "train",
            }
        )
        write(output / "inventory-manifest.json", combined)
        audit = audit_manifest(output / "inventory-manifest.json")
        write(output / "inventory-audit.json", audit.model_dump(mode="json"))
        require(audit.outcome == "isolated", "full_inventory_rejected")
        run_dir = root / "artifacts/s4-video-pilot-train-001"
        common = [
            "--config",
            str(output / "config-30.json"),
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
            str(root / "reports/s4-video-pilot-train-001.json"),
            "--init-adapter",
            str(adapter),
            "--init-adapter-sha256",
            INIT_SHA,
        ]
        order = list(range(300))
        random.Random(42).shuffle(order)
        exposure = [selected[index] for index in order]
        draw_groups = []
        for start in range(0, 300, 10):
            counts = Counter(
                "video" if r["meta"]["modality"] == ["video"] else membership[r["id"]]
                for r in exposure[start : start + 10]
            )
            require(
                counts
                == {
                    "video": 5,
                    "text": 1,
                    "image": 1,
                    "screenshot": 1,
                    "audio": 1,
                    "crossmodal": 1,
                },
                "per_step_balance",
            )
            draw_groups.append(dict(counts))
        result = {
            "status": "prepared",
            "exit_code": 0,
            "gpu_used": False,
            "training_started": False,
            "optimizer": "fresh AdamW initialized from S4 first pilot; not resume",
            "init_adapter": identity,
            "scope": (
                "Known lineage exclusion only; no all-history unseen claim or quality admission"
            ),
            "video_capacity": capacity,
            "video_quota": dict(
                Counter(kind(row) + ":" + row["questions"][0]["target"]["value"] for row in videos)
            ),
            "per_step_draw_counts": draw_groups,
            "total_unique_draws": 300,
            "ledger_check": ledger_check,
            "inventory_audit": "isolated",
            "superseded_current_train_artifacts": superseded,
            "supersession_reason": (
                "Overlapping current TRAIN IDs become historical only in derived manifest; "
                "original files, hashes and pools preserved; not gold readmission"
            ),
            "input_sha256": {
                **input_hashes,
                "previous_report": sha(previous / "report.json"),
                "previous_data": sha(previous / "train.jsonl"),
                "training_report": sha(train_report_path),
                "stable_train": sha(stable / "train.jsonl"),
                "video_source": sha(video_source / "train.jsonl"),
                "crossmodal_provenance": sha(cross / "provenance.json"),
                "inventory": sha(inventory_path),
            },
            "output_sha256": {
                name: sha(output / name)
                for name in (
                    "train.jsonl",
                    "config-30.json",
                    "allocation-ledger.jsonl",
                    "inventory-manifest.json",
                    "inventory-audit.json",
                )
            },
            "preflight_argv": [
                "uv",
                "run",
                "python",
                "-m",
                "scripts.train_s1",
                "preflight",
                *common,
            ],
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
            "wallclock_limit_seconds": 1800,
            "model_asset_preflight": "not_run",
            "script_sha256": sha(Path(__file__)),
            "code_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or "unavailable",
            "intake_code_sha256": sha(Path("train/trainer.py")),
            "python": platform.python_version(),
            "duration_seconds": time.monotonic() - started,
            "cost": None,
        }
        write(output / "report.json", result)
        return result
    except Exception:
        if not (output / "report.json").exists():
            write(
                output / "report.json",
                {
                    "status": "rejected",
                    "exit_code": 1,
                    "categories": ["cpu_preparation_rejected"],
                    "gpu_used": False,
                    "duration_seconds": time.monotonic() - started,
                    "cost": None,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        args.root,
        args.output_dir or args.root / "artifacts/s4-video-pilot-001",
        model_dir=args.model_dir,
        asset_manifest=args.asset_manifest,
    )
    print(json.dumps({"status": report["status"], "total_unique_draws": 300, "gpu_used": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
