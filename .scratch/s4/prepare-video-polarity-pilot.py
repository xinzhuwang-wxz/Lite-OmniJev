"""CPU-only gold-derived video polarity/control prototype; never starts training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

from data.schema import HardTarget, Sample, validate_pool_isolation
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines
from scripts.s1_text_pools import canonical_sample_sha
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples

VIDEO_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"
WRAPPER = "For this video, is the correct yes/no answer to the following question 'no'?\n\n"
NAMESPACE = "s4-video-polarity:42:"


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "claim_prepare", Path(__file__).with_name("prepare-claim-pair-pilot.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def video_gold(sample: Sample) -> str:
    require(
        sample.meta.pool == "train"
        and sample.meta.modality == ["video"]
        and len(sample.prefix.media) == 1
        and sample.prefix.media[0].type == "video"
        and len(sample.questions) == 1,
        "source_train_video_noul",
    )
    q = sample.questions[0]
    require(
        q.type == "noul" and isinstance(q.target, HardTarget) and q.target.value in ["yes", "no"],
        "source_hard_yes_no_only",
    )
    assert isinstance(q.target, HardTarget)
    return str(q.target.value)


def select_families(samples: list[Sample]) -> list[Sample]:
    # Fixed hash selection; no developer labels or historical-unseen claim.
    buckets: dict[str, list[Sample]] = {"yes": [], "no": []}
    used = set()
    for sample in sorted(
        samples,
        key=lambda s: (
            hashlib.sha256((NAMESPACE + canonical_sample_sha(s)).encode()).hexdigest(),
            s.id,
        ),
    ):
        label = video_gold(sample)
        if sample.meta.family_id in used or len(buckets[label]) == 15:
            continue
        buckets[label].append(sample)
        used.add(sample.meta.family_id)
    require(all(len(v) == 15 for v in buckets.values()), "balanced30_family_capacity")
    # Three families/step: alternate2yes1no and1yes2no; exactly15/15 overall.
    result = []
    for step in range(10):
        for label in ["yes", "yes", "no"] if step % 2 == 0 else ["no", "no", "yes"]:
            result.append(buckets[label].pop(0))
    return result


def variants(source: Sample, arm: str) -> tuple[Sample, Sample]:
    y = video_gold(source)
    require(arm in {"control", "polarity"}, "unknown_arm")
    result = []
    for offset in range(2):
        payload = source.model_dump(mode="json")
        payload["id"] = f"video-polarity-{arm}:{canonical_sample_sha(source)}:{offset}"
        payload["meta"]["data_version"] = f"clevrer-train-video-noul-polarity-{arm}-v1"
        if arm == "polarity" and offset == 1:
            payload["questions"][0]["instruction"] = WRAPPER + source.questions[0].instruction
            payload["questions"][0]["target"] = {
                "kind": "hard",
                "value": "yes" if y == "no" else "no",
                "source": "gold-derived-noul-no-complement-v1",
            }
        derived = Sample.model_validate(payload)
        require(
            derived.prefix == source.prefix
            and derived.meta.family_id == source.meta.family_id
            and derived.meta.source_sha256 == source.meta.source_sha256,
            "preserve_component_identity",
        )
        result.append(derived)
    return result[0], result[1]


def prepare(
    root: Path, out: Path, exposure_report_sha: str, replay_report_sha: str
) -> dict[str, Any]:
    started = time.monotonic()
    h = helper()
    require(root.is_absolute() and out.is_absolute() and not out.exists(), "new_absolute_output")
    base = root / "artifacts/s4-pilot-exposure-009"
    replay = root / "artifacts/s4-temporal-curriculum-pilot-001"
    source = root / "artifacts/s3-pools-video-001/run-a"
    require(
        h.sha(base / "report.json") == exposure_report_sha
        and h.sha(replay / "report.json") == replay_report_sha,
        "receipt_sha",
    )
    er, rr, sr = [h.read(p / "report.json") for p in [base, replay, source]]
    require(
        er["status"] == "known_pilot_exposures_reconciled"
        and er["exit_code"] == 0
        and er["unknown_update_bearing_inputs"] == 0,
        "settled_exposure",
    )
    require(
        rr["status"] == "ready_for_preflight"
        and rr["source_admission"] == "verified_existing_scope"
        and rr["inventory_isolation"] == "isolated",
        "existing_admitted_scope",
    )
    h.bound_outputs(base, er, ["inventory-manifest.json"])
    h.bound_outputs(
        replay, rr, ["train.jsonl", "allocation-ledger.jsonl", "config-ce.json", "draw-recipe.json"]
    )
    require(
        h.sha(source / "report.json") == rr["input_report_sha256"]["video_source"]
        and sr["status"] == "passed"
        and sr["exit_code"] == 0,
        "source_receipt",
    )
    require(
        h.sha(source / "train.jsonl") == sr["outputs"]["train_jsonl_sha256"] == VIDEO_SHA
        and h.sha(source / "allocation-ledger.jsonl") == sr["outputs"]["allocation_ledger_sha256"],
        "full8299_binding",
    )
    inventory = Manifest.model_validate_json((base / "inventory-manifest.json").read_bytes())
    originals = [Sample.model_validate(r) for r in h.rows(source / "train.jsonl")]
    require(len(originals) == 8299, "source8299")
    selected = select_families(originals)
    require(
        all(v == "video" for v in h.cohorts(selected, inventory).values()),
        "admitted_source_inventory_match",
    )
    require(len({s.prefix.media[0].sha256 for s in selected}) == 30, "unique30_physical_video")
    source_ledger = h.exact_ledger(
        selected, [r for r in h.rows(source / "allocation-ledger.jsonl") if r["pool"] == "train"]
    )
    replay_samples = [Sample.model_validate(r) for r in h.rows(replay / "train.jsonl")]
    require(
        len(replay_samples) == 120 and all(s.meta.pool == "train" for s in replay_samples),
        "replay120_train",
    )
    order = list(range(120))
    random.Random(42).shuffle(order)
    replay_draws = [replay_samples[i] for i in order]
    replay_versions = h.cohorts(replay_samples, inventory)
    require(
        all(
            [
                replay_versions[s.meta.data_version]
                for s in replay_draws[step * 12 + 6 : step * 12 + 12]
            ]
            == ["crossmodal", "crossmodal", "audio", "image", "screenshot", "text"]
            for step in range(10)
        ),
        "fixed_replay_offsets",
    )
    replay_ledger = h.rows(replay / "allocation-ledger.jsonl")
    from scripts.gold_media_bundle import media_path

    for s in selected:
        require(
            h.sha(media_path([Path(sr["source"]["media_root"])], s.prefix.media[0].uri))
            == s.prefix.media[0].sha256,
            "selected_media_sha",
        )
    out.mkdir(mode=0o700)
    reports = {}
    for arm in ["control", "polarity"]:
        folder = out / arm
        folder.mkdir(mode=0o700)
        draws, ledger, recipe = [], [], []
        for step in range(10):
            for family_offset in range(3):
                n = step * 3 + family_offset
                parent = selected[n]
                for offset, s in enumerate(variants(parent, arm)):
                    draws.append(s)
                    ledger.append(
                        {
                            **source_ledger[n],
                            "item_id": s.id,
                            "derived_sample_sha256": canonical_sample_sha(s),
                            "parent_item_id": parent.id,
                            "parent_sample_sha256": canonical_sample_sha(parent),
                            "transform": "no-complement"
                            if arm == "polarity" and offset
                            else "identity",
                            "data_version": s.meta.data_version,
                        }
                    )
                    recipe.append(
                        {
                            "draw": step * 12 + family_offset * 2 + offset,
                            "item_id": s.id,
                            "family_id": s.meta.family_id,
                            "sample_sha256": canonical_sample_sha(s),
                            "parent_sample_sha256": canonical_sample_sha(parent),
                            "variant": offset,
                            "arm": arm,
                        }
                    )
            tail = replay_draws[step * 12 + 6 : step * 12 + 12]
            draws.extend(tail)
            ledger.extend(h.exact_ledger(tail, replay_ledger))
        validate_pool_isolation(draws)
        require(len(draws) == len({s.id for s in draws}) == 120, "unique120_draws")
        arranged = h.inverse_order(draws)
        h.write(folder / "train.jsonl", data_bytes(arranged).decode())
        h.write(
            folder / "allocation-ledger.jsonl",
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in ledger),
        )
        versions = {
            s.meta.data_version: (
                "video" if s.meta.modality == ["video"] else replay_versions[s.meta.data_version]
            )
            for s in draws
        }
        combined = inventory.model_dump(mode="json", exclude_none=True)
        wanted = {s.id for s in draws}
        for index, a in enumerate(inventory.samples):
            if (
                a.expected_pool == "train"
                and a.exposure == "current"
                and any(json.loads(line)["id"] in wanted for line in bound_lines(a))
            ):
                combined["samples"][index]["exposure"] = "historical"
        combined["samples"].append(
            {
                "path": str(folder / "train.jsonl"),
                "sha256": h.sha(folder / "train.jsonl"),
                "expected_pool": "train",
                "exposure": "current",
                "cohorts_by_data_version": versions,
            }
        )
        h.write(folder / "inventory-manifest.json", combined)
        audit = audit_manifest(folder / "inventory-manifest.json")
        h.write(folder / "inventory-audit.json", audit.model_dump(mode="json"))
        require(audit.outcome == "isolated", "full_inventory_isolation")
        config = {**h.read(replay / "config-ce.json"), "expected_data_versions": sorted(versions)}
        validated = LoopConfig.model_validate(config)
        require(
            validated.steps == 10
            and validated.accumulation == 12
            and validated.seed == 42
            and validated.sample_count == 120
            and validated.brier_weight
            == validated.reference_kl_weight
            == validated.pair_loss_weight
            == 0,
            "fixed_ce_config",
        )
        smoke_samples(validated, folder / "train.jsonl", expected_data_versions=sorted(versions))
        check = verify_data_is_train_pool(
            folder / "train.jsonl", [folder / "allocation-ledger.jsonl"]
        )
        h.write(folder / "config-ce.json", config)
        h.write(
            folder / "draw-recipe.json",
            {
                "video": recipe,
                "same_physical_families_across_arms": True,
                "training_order_seed": 42,
                "replay_source_sha256": h.sha(replay / "train.jsonl"),
            },
        )
        reports[arm] = {
            "status": "cpu_prepared_pending_root_training_decision",
            "inventory_isolation": audit.outcome,
            "ledger_check": check,
            "video_original_gold": dict(Counter(video_gold(s) for s in selected)),
            "output_sha256": {p.name: h.sha(p) for p in folder.iterdir() if p.is_file()},
        }
    assert h.__file__ is not None
    report = {
        "status": "cpu_prepared_pending_root_training_decision",
        "exit_code": 0,
        "gpu_used": False,
        "training_authorized": False,
        "arm_wallclock_limit_seconds": 240,
        "families": 30,
        "video_rows_per_arm": 60,
        "total_draws_per_arm": 120,
        "source_sha256": VIDEO_SHA,
        "source_ledger_sha256": h.sha(source / "allocation-ledger.jsonl"),
        "source_report_sha256": h.sha(source / "report.json"),
        "exposure_report_sha256": exposure_report_sha,
        "replay_report_sha256": replay_report_sha,
        "source_admission": "verified_existing_scope",
        "admission_trace": rr["admission_trace"],
        "question_transform": WRAPPER,
        "historical_unseen": "not_claimed",
        "init_selection": "pending_root_decision; config contains no selected adapter",
        "arms": reports,
        "helper_sha256": h.sha(Path(__file__)),
        "cost": None,
        "duration_seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "seed": 42,
        "shared_helper_sha256": h.sha(Path(h.__file__)),
    }
    h.write(out / "report.json", report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "output-dir"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--exposure-report-sha256", required=True)
    ap.add_argument("--replay-report-sha256", required=True)
    a = ap.parse_args()
    existed = a.output_dir.exists()
    try:
        r = prepare(a.root, a.output_dir, a.exposure_report_sha256, a.replay_report_sha256)
    except Exception as e:
        if not existed and a.output_dir.exists() and not (a.output_dir / "report.json").exists():
            helper().write(
                a.output_dir / "report.json",
                {
                    "status": "rejected",
                    "exit_code": 1,
                    "category": str(e) if isinstance(e, ValueError) else type(e).__name__,
                    "gpu_used": False,
                },
            )
        print(json.dumps({"status": "rejected", "exit_code": 1}))
        return 1
    print(json.dumps({k: r[k] for k in ["status", "families", "total_draws_per_arm", "gpu_used"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
