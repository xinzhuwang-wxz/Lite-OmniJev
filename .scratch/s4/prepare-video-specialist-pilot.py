"""CPU-only source-preserving video-specialist research preparation."""

from __future__ import annotations

import argparse
import collections

# Reuse the already verified source/inventory preparation surface.
import importlib.util
import json
import random
import time
from pathlib import Path

from scripts.gold_inventory import Manifest, audit_manifest, bound_lines
from scripts.s1_text_pools import canonical_sample_sha
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples

path = Path(__file__).with_name("prepare-native-event-condition-pilot.py")
spec = importlib.util.spec_from_file_location("native_specialist_source", path)
assert spec and spec.loader
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
h = native.h
require = native.require
SOURCE_REPORT_SHA = "401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696"


def actual_draws(samples, seed=42):
    require(
        len(samples) == 60 and len({s.meta.family_id for s in samples}) == 60,
        "requires60unique original video families",
    )
    require(
        all(s.meta.pool == "train" and s.meta.modality == ["video"] for s in samples),
        "video TRAIN only",
    )
    order = list(range(60))
    random.Random(seed).shuffle(order)
    return [samples[order[i % 60]] for i in range(120)]


def specialist_config(original, version):
    result = dict(original)
    require(
        result["steps"] == 10 and result["accumulation"] == 12 and result["seed"] == 42,
        "native loop contract",
    )
    result.update(
        sample_count=60,
        modality="video",
        expected_data_version=version,
        expected_data_versions=None,
    )
    config = LoopConfig.model_validate(result)
    require(
        not config.brier_weight and not config.reference_kl_weight and not config.pair_loss_weight,
        "pure CE only",
    )
    return result


def prepare(root, out, exposure, exposure_sha):
    started = time.monotonic()
    require(not out.exists(), "output already exists")
    native.check_no_updates()
    er_path = exposure / "report.json"
    require(h.sha(er_path) == exposure_sha, "latest exposure SHA")
    er = h.read(er_path)
    require(
        er["status"] == "known_pilot_exposures_reconciled"
        and er["exit_code"] == 0
        and er["unknown_update_bearing_inputs"] == 0,
        "complete latest exposure required",
    )
    h.bound_outputs(exposure, er, ["inventory-manifest.json"])
    base_manifest = exposure / "inventory-manifest.json"
    base_sha = h.sha(base_manifest)
    inventory = Manifest.model_validate_json(base_manifest.read_bytes())
    source = root / "artifacts/s4-native-event-condition-pilot-001"
    require(h.sha(source / "report.json") == SOURCE_REPORT_SHA, "native source receipt")
    source_report = h.read(source / "report.json")
    require(
        source_report["status"] == "data_ready_only"
        and source_report["inventory_isolation"] == "isolated",
        "native preparation source",
    )
    samples = native.bound_rows(source)
    video = [s for s in samples if s.meta.modality == ["video"]]
    draws = actual_draws(video)
    require(
        collections.Counter(s.id for s in draws) == {s.id: 2 for s in video}, "exact two passes"
    )
    require(
        collections.Counter(s.questions[0].target.value for s in video) == {"yes": 30, "no": 30},
        "native60 gold balance",
    )
    versions = {s.meta.data_version for s in video}
    require(len(versions) == 1, "single original video version")
    ledger = h.exact_ledger(video, h.rows(source / "allocation-ledger.jsonl"))
    config = specialist_config(h.read(source / "config-ce.json"), next(iter(versions)))
    out.mkdir(mode=0o700)
    h.write(out / "train.jsonl", data_bytes(video).decode())
    h.write(
        out / "allocation-ledger.jsonl",
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ledger),
    )
    h.write(out / "config-ce.json", config)
    verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    loaded = smoke_samples(
        LoopConfig.model_validate(config),
        out / "train.jsonl",
        expected_data_version=next(iter(versions)),
    )
    require(data_bytes(loaded) == data_bytes(video), "canonical samples unchanged")
    h.write(
        out / "draw-recipe.json",
        {
            "seed": 42,
            "steps": 10,
            "accumulation": 12,
            "actual_draws": 120,
            "unique_samples": 60,
            "draws": [
                {
                    "draw": i,
                    "step": i // 12,
                    "offset": i % 12,
                    "item_id": s.id,
                    "family_id": s.meta.family_id,
                    "sample_sha256": canonical_sample_sha(s),
                }
                for i, s in enumerate(draws)
            ],
        },
    )
    combined = inventory.model_dump(mode="json", exclude_none=True)
    ids = {s.id for s in video}
    for index, artifact in enumerate(inventory.samples):
        if (
            artifact.expected_pool == "train"
            and artifact.exposure == "current"
            and any(json.loads(line)["id"] in ids for line in bound_lines(artifact))
        ):
            combined["samples"][index]["exposure"] = "historical"
    combined["samples"].append(
        {
            "path": str(out / "train.jsonl"),
            "sha256": h.sha(out / "train.jsonl"),
            "expected_pool": "train",
            "exposure": "current",
            "cohorts_by_data_version": {next(iter(versions)): "video"},
        }
    )
    h.write(out / "inventory-manifest.json", combined)
    native.check_no_updates()
    audit = audit_manifest(out / "inventory-manifest.json")
    h.write(out / "inventory-audit.json", audit.model_dump(mode="json"))
    native.check_no_updates()
    require(h.sha(er_path) == exposure_sha and h.sha(base_manifest) == base_sha, "base changed")
    require(audit.outcome == "isolated", "full inventory isolation")
    result = {
        "status": "data_ready_only",
        "exit_code": 0,
        "training_authorized": False,
        "parent_model": None,
        "pending_parent_selection": True,
        "gpu_used": False,
        "cost": None,
        "scope": "video-specialist research; not production single-LoRA promotion",
        "comparison": "native mixed10: video exposure2x plus replay removal; not single-factor",
        "unique_video_families": 60,
        "actual_video_draws": 120,
        "draws_per_sample": 2,
        "new_source_families": 0,
        "inventory_isolation": audit.outcome,
        "source_report_sha256": SOURCE_REPORT_SHA,
        "source_data_sha256": source_report["output_sha256"]["train.jsonl"],
        "source_video_canonical_sha256": h.digest(data_bytes(video)),
        "draw_order_canonical_sha256": h.digest(data_bytes(draws)),
        "exposure_base": {
            "path": str(exposure),
            "report_sha256": exposure_sha,
            "manifest_sha256": base_sha,
        },
        "helper_sha256": h.sha(Path(__file__)),
        "native_helper_sha256": h.sha(path),
        "duration_seconds": time.monotonic() - started,
        "output_sha256": {p.name: h.sha(p) for p in out.iterdir() if p.is_file()},
    }
    h.write(out / "report.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--exposure-dir", type=Path, required=True)
    ap.add_argument("--exposure-report-sha256", required=True)
    args = ap.parse_args()
    existed = args.output_dir.exists()
    try:
        result = prepare(
            args.root.resolve(),
            args.output_dir.resolve(),
            args.exposure_dir.resolve(),
            args.exposure_report_sha256,
        )
    except Exception as exc:
        result = {
            "status": "rejected",
            "exit_code": 1,
            "error": str(exc),
            "training_authorized": False,
            "gpu_used": False,
        }
        if not existed:
            args.output_dir.mkdir(mode=0o700, exist_ok=True)
            if not (args.output_dir / "failure.json").exists():
                h.write(args.output_dir / "failure.json", result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
