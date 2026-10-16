"""CPU-only fixed global TRAIN pool; no parent selection or training authorization."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from collections import Counter
from pathlib import Path

from data.schema import HardTarget, Sample
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines, digest
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples, verified_media_file

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
EXPOSURE_SHA = "5872a667ef0935436430e22f8c041c85f4e4f2c0c4ffc1e0076e4a9e6525b15e"
LINEAGE_SHA = "cb9574e06fc9950198f703ab9107d6d2d42824a5d2399ded2a26a146dd88cbfa"
FIRST_SHA = "1e7f877e04ea311d29ebc1fe1711f31aa2cc91034f8794b666850bcf5f2fff19"
PRIOR_SHA = "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b"
VIDEO_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"
NAMESPACE = "s4-continuous-main-pool-001:42:"
OLD = ("audio", "image", "screenshot", "text")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def helper(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kind(s):
    words = set(s.questions[0].instruction.lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else "motion"
        if words & {"moving", "stationary"}
        else "existence"
    )


def select_video(samples, excluded):
    require(
        all(
            s.meta.pool == "train"
            and s.meta.modality == ["video"]
            and len(s.questions) == 1
            and isinstance(s.questions[0].target, HardTarget)
            and s.questions[0].target.value in ("yes", "no")
            for s in samples
        ),
        "video_train_gold",
    )
    used = set(excluded)
    result = {}
    for task, count in [("temporal", 250), ("motion", 25), ("existence", 25)]:
        for target in ("yes", "no"):
            candidates = sorted(
                (s for s in samples if kind(s) == task and s.questions[0].target.value == target),
                key=lambda s: digest((NAMESPACE + s.id).encode()),
            )
            bucket = []
            for s in candidates:
                if s.meta.family_id not in used:
                    bucket.append(s)
                    used.add(s.meta.family_id)
                if len(bucket) == count:
                    break
            require(len(bucket) == count, f"video_capacity:{task}:{target}")
            result[task, target] = bucket
    return result


def unique_families(samples, count, cohort):
    require(all(s.meta.pool == "train" for s in samples), "old_train_only")
    used = set()
    chosen = []
    for s in sorted(samples, key=lambda x: digest((NAMESPACE + cohort + x.id).encode())):
        if s.meta.family_id not in used:
            chosen.append(s)
            used.add(s.meta.family_id)
        if len(chosen) == count:
            break
    require(len(chosen) == count, f"old_capacity:{cohort}")
    return chosen


def layout(video, cross, old):
    require(len(cross) == 20 and len({s.id for s in cross}) == 20, "cross_capacity")
    require(all(len(old[c]) == 100 for c in OLD), "old_layout_capacity")
    result = []
    for step in range(100):
        # Each half balances both minor task labels; every block has 3yes/3no.
        half = step // 50
        local = step % 50
        even_yes = 3 if half == 0 else 2
        yes_count = even_yes if local % 2 == 0 else 5 - even_yes
        yes_start = 125 * half + 5 * (local // 2) + (even_yes if local % 2 else 0)
        no_start = 125 * half + 5 * (local // 2) + (5 - even_yes if local % 2 else 0)
        result.extend(video["temporal", "yes"][yes_start : yes_start + yes_count])
        result.extend(video["temporal", "no"][no_start : no_start + 5 - yes_count])
        task = "motion" if step % 2 == 0 else "existence"
        target = "no" if yes_count == 3 else "yes"
        result.append(video[task, target][(step % 50) // 2])
        result.extend(cross[2 * (step % 10) : 2 * (step % 10) + 2])
        result.extend(old[c][step] for c in OLD)
    require(len(result) == 1200, "layout_draw_count")
    for step in range(100):
        block = result[12 * step : 12 * step + 6]
        require(
            len(block) == 6
            and Counter(s.questions[0].target.value for s in block) == {"yes": 3, "no": 3},
            "video_block_balance",
        )
    require(
        len({s.meta.family_id for i, s in enumerate(result) if i % 12 < 6}) == 600,
        "video_family_uniqueness",
    )
    return result


def inverse_order(desired):
    order = list(range(len(desired)))
    random.Random(42).shuffle(order)
    arranged = list(desired)
    for draw, index in enumerate(order):
        arranged[index] = desired[draw]
    require([arranged[i] for i in order] == desired, "inverse_order")
    return arranged, order


def alias_draw(source, n, ledger):
    require(source.meta.pool == "train" and ledger["pool"] == "train", "alias_train_only")
    h = helper("prepare-claim-pair-pilot.py")
    require(
        ledger["item_id"] == source.id
        and ledger["derived_sample_sha256"] == h.canonical_sample_sha(source),
        "source_ledger",
    )
    sample = source.model_copy(update={"id": f"{NAMESPACE}draw:{n:04d}"})
    record = {
        **ledger,
        "item_id": sample.id,
        "derived_sample_sha256": h.canonical_sample_sha(sample),
        "source_item_id": source.id,
        "source_sample_sha256": h.canonical_sample_sha(source),
        "alias_only_gold_unchanged": True,
        "media_sha256": [m.sha256 for m in source.prefix.media],
    }
    return sample, record


def prepare(root, out):
    started = time.monotonic()
    h = helper("prepare-claim-pair-pilot.py")
    require(root.is_absolute() and out.is_absolute() and not out.exists(), "output_path")
    exposure = root / "artifacts/s4-pilot-exposure-013"
    require(h.sha(exposure / "report.json") == EXPOSURE_SHA, "latest_exposure_sha")
    er = h.read(exposure / "report.json")
    require(er["exit_code"] == 0 and er["unknown_update_bearing_inputs"] == 0, "exposure_status")
    h.bound_outputs(exposure, er, ["inventory-manifest.json"])
    inventory = Manifest.model_validate_json((exposure / "inventory-manifest.json").read_bytes())
    lineage_path = root / "reports/s4-ce0-video-exposure-001.json"
    require(h.sha(lineage_path) == LINEAGE_SHA, "lineage_sha")
    lineage = h.read(lineage_path)
    require(lineage["status"] == "completed" and lineage["exit_code"] == 0, "lineage_status")
    for stage in lineage["stages"]:
        require(
            h.sha(Path(stage["report"])) == stage["report_sha256"]
            and h.sha(Path(stage["data"])) == stage["data_sha256"],
            "lineage_binding",
        )
    first = root / "artifacts/s4-temporal-coverage-pilot-001"
    require(h.sha(first / "report.json") == FIRST_SHA, "first_sha")
    fr = h.read(first / "report.json")
    h.bound_outputs(first, fr, list(fr["output_sha256"]))
    first_rows = [Sample.model_validate(x) for x in h.rows(first / "train.jsonl")]
    excluded = set(lineage["lineage_unique_video_families"]) | {
        s.meta.family_id for s in first_rows if s.meta.modality == ["video"]
    }
    require(len(excluded) == 408, "f2db_lineage408")
    prior = root / "artifacts/s4-temporal-curriculum-pilot-001"
    require(h.sha(prior / "report.json") == PRIOR_SHA, "prior_sha")
    pr = h.read(prior / "report.json")
    for evidence in pr["admission_trace"]["files"]:
        require(h.sha(Path(evidence["repository_path"])) == evidence["sha256"], "admission_sha")
    jointpath = root / "reports/s3-joint-preparation-001.json"
    require(h.sha(jointpath) == pr["admission_trace"]["joint_preparation_sha256"], "joint_sha")
    joint = h.read(jointpath)
    require(
        joint["status"] == "passed" and joint["ledger_check"]["verified_train"] == 21184,
        "joint_status",
    )
    sources, ledgers, pins = {}, [], {}
    for cohort in (*OLD, "video"):
        info = joint["inputs"][cohort]
        path = Path(info["path"])
        ledger_path = path.with_name("allocation-ledger.jsonl")
        require(
            h.sha(path) == info["sha256"] and h.sha(ledger_path) == info["ledger_sha256"],
            "joint_source_binding",
        )
        sources[cohort] = [Sample.model_validate(x) for x in h.rows(path)]
        require(len(sources[cohort]) == info["count"], "source_count")
        require(all(s.meta.pool == "train" for s in sources[cohort]), "source_train")
        ledgers.extend(x for x in h.rows(ledger_path) if x["pool"] == "train")
        pins[cohort] = info
    require(
        pins["video"]["sha256"] == VIDEO_SHA and pins["video"]["count"] == 8299, "video_full_scope"
    )
    video = select_video(sources["video"], excluded)
    old = {c: unique_families(sources[c], 100, c) for c in OLD}
    cross = [s for s in first_rows if s.meta.modality == ["image", "audio"]]
    require(len(cross) == 20, "audited_cross_capacity")
    ledgers.extend(h.rows(first / "allocation-ledger.jsonl"))
    originals = layout(video, cross, old)
    unique_originals = list({s.id: s for s in originals}.values())
    versions = h.cohorts(unique_originals, inventory)
    source_ledger = {
        s.id: a
        for s, a in zip(unique_originals, h.exact_ledger(unique_originals, ledgers), strict=True)
    }
    desired, alias_ledger = [], []
    for n, source in enumerate(originals):
        s, a = alias_draw(source, n, source_ledger[source.id])
        desired.append(s)
        alias_ledger.append(a)
    config = {
        **h.read(first / "config-ce.json"),
        "sample_count": 1200,
        "steps": 100,
        "seed": 42,
        "accumulation": 12,
        "checkpoint_every": 10,
        "midway_step": 10,
        "expected_data_versions": sorted(versions),
    }
    require(
        config["rank"] == 16
        and config["alpha"] == 32
        and config["video_num_frames"] == 16
        and config["image_max_soft_tokens"] == 1120
        and not any(
            config.get(k, 0) for k in ("pair_loss_weight", "reference_kl_weight", "brier_weight")
        ),
        "frozen_ce_configuration",
    )
    # Verify every selected source media file, once, without downloading or decoding private pools.
    media_verified = {}
    roots = [Path(p) for p in config["media_roots"]]
    for s in unique_originals:
        for m in s.prefix.media:
            if m.sha256 not in media_verified:
                path = verified_media_file(roots[0], m.uri, m.sha256, extra_roots=roots[1:])
                require(h.sha(path) == m.sha256, "source_media_sha")
                media_verified[m.sha256] = str(path)
    components = {}
    wanted_cross = {s.id for s in cross}
    for artifact in inventory.component_ledgers:
        for line in bound_lines(artifact):
            record = json.loads(line)
            if record["item_id"] in wanted_cross and record["pool"] == "train":
                require(components.get(record["item_id"], record) == record, "component_conflict")
                components[record["item_id"]] = record
    require(set(components) == wanted_cross, "cross_components")
    alias_components = [
        {**components[o.id], "item_id": s.id}
        for o, s in zip(originals, desired, strict=True)
        if o.id in wanted_cross
    ]
    arranged, order = inverse_order(desired)
    out.mkdir(mode=0o700)
    h.write(out / "train.jsonl", data_bytes(arranged).decode())
    h.write(
        out / "allocation-ledger.jsonl",
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in alias_ledger),
    )
    h.write(
        out / "components.jsonl",
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in alias_components),
    )
    h.write(out / "config-ce.json", config)
    smoke_samples(
        LoopConfig.model_validate(config),
        out / "train.jsonl",
        expected_data_versions=sorted(versions),
    )
    ledger_check = verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    combined = inventory.model_dump(mode="json", exclude_none=True)
    combined["samples"].append(
        {
            "path": str(out / "train.jsonl"),
            "sha256": h.sha(out / "train.jsonl"),
            "expected_pool": "train",
            "exposure": "current",
            "cohorts_by_data_version": versions,
        }
    )
    combined["component_ledgers"].append(
        {"path": str(out / "components.jsonl"), "sha256": h.sha(out / "components.jsonl")}
    )
    h.write(out / "inventory-manifest.json", combined)
    audit = audit_manifest(out / "inventory-manifest.json")
    h.write(out / "inventory-audit.json", audit.model_dump(mode="json"))
    require(audit.outcome == "isolated", "full_inventory_isolation")
    require(h.sha(exposure / "report.json") == EXPOSURE_SHA, "exposure_changed_during_build")
    h.write(
        out / "draw-recipe.json",
        {
            "seed": 42,
            "global_steps": 100,
            "accumulation": 12,
            "draws": [
                {
                    "draw": n,
                    "step": n // 12,
                    "offset": n % 12,
                    "data_index": order[n],
                    "item_id": s.id,
                    "source_item_id": o.id,
                    "source_sample_sha256": h.canonical_sample_sha(o),
                    "sample_sha256": h.canonical_sample_sha(s),
                    "family_id": s.meta.family_id,
                    "cohort": versions[s.meta.data_version],
                    "data_version": s.meta.data_version,
                    "video_heuristic": kind(o) if o.meta.modality == ["video"] else None,
                    "target": o.questions[0].target.model_dump(mode="json"),
                }
                for n, (s, o) in enumerate(zip(desired, originals, strict=True))
            ],
        },
    )
    result = {
        "status": "data_ready_only",
        "exit_code": 0,
        "parent": None,
        "pending_parent_selection": True,
        "training_authorized": False,
        "training_started": False,
        "gpu_used": False,
        "global_pool_draws": 1200,
        "global_steps_upper_bound": 100,
        "future_first_invocation_steps": 10,
        "future_invocation_wallclock_cap_seconds": 240,
        "actual_training_exposure_draws": 0,
        "pool_capacity_is_not_completed_exposure": True,
        "inventory_isolation": audit.outcome,
        "allocation_ledger_check": ledger_check,
        "per_step": {"video": 6, "crossmodal": 2, **{c: 1 for c in OLD}},
        "video_target_per_step": {"yes": 3, "no": 3},
        "video_unique_families": 600,
        "video_selected": {
            task: {t: len(video[task, t]) for t in ("yes", "no")}
            for task in ("temporal", "motion", "existence")
        },
        "old_unique_families": {c: len({s.meta.family_id for s in old[c]}) for c in OLD},
        "cross_replay": {
            "draws": 200,
            "unique_source_samples": 20,
            "unique_families": len({s.meta.family_id for s in cross}),
            "cycles": 10,
            "reason": "reuse explicitly audited temporal60 replay",
        },
        "source_original_unique_samples": len(unique_originals),
        "source_gold_unchanged": True,
        "selection_namespace": NAMESPACE,
        "excluded_families": sorted(excluded),
        "historical_exclusion_scope": (
            "new only to selected f2db lineage408; other failed branches may overlap"
        ),
        "video_rule": "existing heuristic word rule, not official annotation classification",
        "source_admission": "verified_existing_scope",
        "admission_trace": pr["admission_trace"],
        "source_pins": pins,
        "source_media_verified": len(media_verified),
        "input_report_sha256": {
            "exposure013": EXPOSURE_SHA,
            "lineage": LINEAGE_SHA,
            "first_coverage": FIRST_SHA,
            "temporal_curriculum": PRIOR_SHA,
        },
        "output_sha256": {p.name: h.sha(p) for p in out.iterdir() if p.is_file()},
        "script_sha256": h.sha(Path(__file__)),
        "helper_sha256": h.sha(Path(h.__file__)),
        "duration_seconds": time.monotonic() - started,
        "cost": None,
    }
    h.write(out / "report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/s4-continuous-main-pool-001"
    )
    a = p.parse_args()
    result = prepare(a.root.resolve(), a.output_dir.resolve())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "exit_code",
                    "inventory_isolation",
                    "video_unique_families",
                    "output_sha256",
                )
            }
        )
    )
