"""Official-program-stratified fixed TRAIN course; no labels, variants or pair objective."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import time
from pathlib import Path

from data.schema import Sample
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines
from scripts.gold_media_bundle import media_path
from scripts.s0_media_dev import source_hash_for_video
from scripts.s1_text_pools import canonical_sample_sha
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples

path = Path(__file__).with_name("prepare-native-event-condition-pilot.py")
spec = importlib.util.spec_from_file_location("matched_native_reuse", path)
assert spec and spec.loader
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
h = native.h
inspection = native.inspection
require = native.require
exclusions = native.exclusions
check_no_updates = native.check_no_updates
VERSION = native.VERSION
REPLAY_SHA = native.REPLAY_SHA
NAMESPACE = "42:s4-program-stratified-pool-001:"


GROUPS = ("before_after", "event_motion", "endpoint_motion", "exist")
EXTRA_EXCLUSIONS = {
    "s4-continuous-matched-pool-001": (
        "2c51e16cb88ab91dfb45dca84a2325b371ad259dce522a01fa50e042090cfa87"
    ),
    "s4-temporal-curriculum-pilot-001": (
        "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b"
    ),
    "s4-temporal-coverage-pilot-002": (
        "44b947dc30fcd7b9df4159354b2d054fbfa17dd8b783b0b841a70309df6bd400"
    ),
    "s4-temporal-coverage-pilot-003": (
        "66d2fc6a2f4440fe8fcc11ea9b5f981cc6ab5605f79d4638b9011df175b6da32"
    ),
    "s4-temporal-order-contrast-pilot-001": (
        "7870a39aec8f64939dabd4035f2416b4a1c454f8dd9ec7e88f9891ffbffc28ab"
    ),
}


def program_group(program):
    require(
        isinstance(program, list) and program and all(isinstance(t, str) for t in program),
        "official postfix list",
    )
    if "filter_before" in program or "filter_after" in program:
        return "before_after"
    if "events" in program and "get_frame" in program:
        return "event_motion"
    if "get_frame" in program:
        return "endpoint_motion"
    require(program[-1] == "exist", "remaining program must be existential")
    return "exist"


def select_layout(buckets):
    chosen = {}
    for group in GROUPS:
        chosen[group] = {}
        for gold in ["yes", "no"]:
            rows = buckets[group][gold]
            require(len(rows) >= 75, f"insufficient {group}/{gold}: {len(rows)}<75")
            chosen[group][gold] = sorted(
                rows, key=lambda r: h.digest((NAMESPACE + group + ":" + gold + ":" + r.id).encode())
            )[:75]
    selected = [r for group in GROUPS for gold in ["yes", "no"] for r in chosen[group][gold]]
    require(len({r.meta.family_id for r in selected}) == 600, "six hundred unique families")
    queues = {
        group: {gold: iter(chosen[group][gold]) for gold in ["yes", "no"]} for group in GROUPS
    }
    layout = []
    for cycle in range(50):
        even = cycle % 2 == 0
        first = [
            ("yes", "no"),
            ("yes", "no"),
            ("yes" if even else "no",),
            ("no" if even else "yes",),
        ]
        second = [
            ("yes" if even else "no",),
            ("no" if even else "yes",),
            ("yes", "no"),
            ("yes", "no"),
        ]
        for allocation in [first, second]:
            layout.append(
                [
                    next(queues[group][gold])
                    for group, golds in zip(GROUPS, allocation, strict=True)
                    for gold in golds
                ]
            )
    return selected, layout


def inverse_order(draws):
    require(len(draws) == 1200, "fixed1200storage")
    order = list(range(len(draws)))
    random.Random(42).shuffle(order)
    arranged = [None] * len(draws)
    for i, index in enumerate(order):
        arranged[index] = draws[i]
    require([arranged[i] for i in order] == draws, "inverse1200roundtrip")
    return arranged


def prepare(root, official, out, exposure, expected_exposure_sha):
    started = time.monotonic()
    require(not out.exists(), "output exists")
    check_no_updates()
    er = h.read(exposure / "report.json")
    require(h.sha(exposure / "report.json") == expected_exposure_sha, "exposure report SHA")
    require(
        er["status"] == "known_pilot_exposures_reconciled"
        and er["exit_code"] == 0
        and er["unknown_update_bearing_inputs"] == 0,
        "complete known exposure required",
    )
    h.bound_outputs(exposure, er, ["inventory-manifest.json"])
    inventory_sha = h.sha(exposure / "inventory-manifest.json")
    inventory = Manifest.model_validate_json((exposure / "inventory-manifest.json").read_bytes())
    original = root / "artifacts/s4-temporal-curriculum-pilot-001"
    require(h.sha(original / "report.json") == REPLAY_SHA, "replay report identity")
    rr = h.read(original / "report.json")
    require(rr["inventory_isolation"] == "isolated" and rr["exit_code"] == 0, "replay status")
    for name, digest in rr["output_sha256"].items():
        require(Path(name).name == name and h.sha(original / name) == digest, "replay output SHA")
    for evidence in rr["admission_trace"]["files"]:
        require(h.sha(Path(evidence["repository_path"])) == evidence["sha256"], "admission proof")
    source = root / "artifacts/s3-pools-video-001/run-a"
    pool = source / "train.jsonl"
    require(h.sha(pool) == inspection.POOL_SHA, "TRAIN pool SHA")
    source_report = h.read(source / "report.json")
    require(
        h.sha(source / "report.json") == rr["input_report_sha256"]["video_source"],
        "source receipt SHA",
    )
    require(
        h.sha(source / "allocation-ledger.jsonl")
        == source_report["outputs"]["allocation_ledger_sha256"],
        "source ledger SHA",
    )
    for path, digest in [
        (official / "questions/train.json", inspection.QUESTIONS_SHA),
        (official / "annotations/train/annotation_train.zip", inspection.ANNOTATIONS_SHA),
        (official / "README.txt", inspection.README_SHA),
    ]:
        require(h.sha(path) == digest, "official source SHA")
    source_samples = [Sample.model_validate(x) for x in h.rows(pool)]
    require(
        len(source_samples) == 8299 and all(s.meta.pool == "train" for s in source_samples),
        "full admitted TRAIN scope",
    )
    excluded, exclusion_report = exclusions(root)
    native_dir = root / "artifacts/s4-native-event-condition-pilot-001"
    require(
        h.sha(native_dir / "report.json")
        == "401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696",
        "native exclusion receipt",
    )
    native_rows = native.bound_rows(native_dir)
    native_families = {r.meta.family_id for r in native_rows if r.meta.modality == ["video"]}
    excluded |= native_families
    exclusion_report.update(
        native_course_families=len(native_families),
        union_with_native=len(excluded),
        scope="f2db plus actual continuous20steps/TRAIN12/native; not globally unseen",
    )
    matched_dir = root / "artifacts/s4-video-text-matched-pilot-001"
    require(
        h.sha(matched_dir / "report.json")
        == "3e764378147c06816a3f14fe71967f53d83fb50871237e094ceb19587419db93",
        "matched exclusion receipt",
    )
    matched_rows = native.bound_rows(matched_dir)
    excluded |= {r.meta.family_id for r in matched_rows if r.meta.modality == ["video"]}
    require(len(excluded) == 660, "explicit660exclusion union")
    exclusion_report["union_with_matched"] = len(excluded)
    extra_scope = []
    for name, digest in EXTRA_EXCLUSIONS.items():
        directory = root / "artifacts" / name
        require(h.sha(directory / "report.json") == digest, "specified prior pilot receipt")
        rows = native.bound_rows(directory)
        videos = [r for r in rows if r.meta.modality == ["video"]]
        h.exact_ledger(videos, h.rows(directory / "allocation-ledger.jsonl"))
        families = {r.meta.family_id for r in videos}
        extra_scope.append(
            {
                "artifact": name,
                "report_sha256": digest,
                "families": len(families),
                "new_exclusions": len(families - excluded),
                "scope": "prepared source families; not claiming all were optimizer-exposed",
            }
        )
        excluded |= families
    exclusion_report.update(
        specified_prior_pilots=extra_scope,
        final_union=len(excluded),
        scope="explicit source-bound pilot-family exclusions; not globally unseen",
    )
    eligible = [r for r in source_samples if r.meta.family_id not in excluded]
    scenes = inspection.selected_scenes(
        official / "questions/train.json", {int(r.meta.family_id.split(":")[1]) for r in eligible}
    )
    buckets = {g: {gold: [] for gold in ["yes", "no"]} for g in GROUPS}
    source_questions = {}
    classifications = {}
    for sample in eligible:
        index = int(sample.meta.family_id.split(":")[1])
        question = next(
            q
            for q in scenes[index]["questions"]
            if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
        )
        require(
            question["question"] == sample.questions[0].instruction
            and question["answer"].casefold() == sample.questions[0].target.value,
            "original text/gold",
        )
        require(
            source_hash_for_video({"scene_index": index}, question, sample.prefix.media[0].sha256)
            == sample.meta.source_sha256,
            "official source program/media",
        )
        group = program_group(question["program"])
        gold = sample.questions[0].target.value
        require(gold in {"yes", "no"}, "official binary gold")
        buckets[group][gold].append(sample)
        source_questions[sample.id] = question
        classifications[sample.id] = group
    capacity = {
        group: {gold: len(rows) for gold, rows in values.items()}
        for group, values in buckets.items()
    }
    out.mkdir(mode=0o700)
    summary = {
        "status": "insufficient_stratum_capacity",
        "exit_code": 1,
        "capacity_after_exclusions": capacity,
        "exclusions": exclusion_report,
        "training_authorized": False,
        "pending_parent_selection": True,
        "parent_model": None,
        "gpu_used": False,
        "cost": None,
    }
    if any(capacity[group][gold] < 75 for group in GROUPS for gold in ["yes", "no"]):
        h.write(out / "report.json", summary)
        return summary
    selected, video_groups = select_layout(buckets)
    summary.update(
        video_families=600,
        gold_counts={"yes": 300, "no": 300},
        stratum_counts={g: {"yes": 75, "no": 75} for g in GROUPS},
    )
    metadata = []
    media_verified = 0
    for sample in selected:
        require(
            h.sha(
                media_path(
                    [Path(source_report["source"]["media_root"])], sample.prefix.media[0].uri
                )
            )
            == sample.prefix.media[0].sha256,
            "selected media SHA",
        )
        media_verified += 1
        program = source_questions[sample.id]["program"]
        metadata.append(
            {
                "item_id": sample.id,
                "family_id": sample.meta.family_id,
                "program_stratum": classifications[sample.id],
                "official_program_sha256": h.digest(json.dumps(program, sort_keys=True).encode()),
                "official_source_sha256": sample.meta.source_sha256,
                "media_sha256": sample.prefix.media[0].sha256,
                "sample_sha256": canonical_sample_sha(sample),
                "original_gold": sample.questions[0].target.value,
                "has_collision_operator": "filter_collision" in program,
            }
        )
    generated_ledger = h.exact_ledger(
        selected, [r for r in h.rows(source / "allocation-ledger.jsonl") if r["pool"] == "train"]
    )
    h.write(
        out / "program-strata-metadata.json",
        {
            "target_source": "original official gold only; exclusive program strata",
            "metadata_only": True,
            "rows": metadata,
        },
    )
    fixed = root / "artifacts/s4-continuous-matched-pool-001"
    require(
        h.sha(fixed / "report.json")
        == "2c51e16cb88ab91dfb45dca84a2325b371ad259dce522a01fa50e042090cfa87",
        "fixed replay receipt",
    )
    fixed_rows = native.bound_rows(fixed)
    by_id = {r.id: r for r in fixed_rows}
    fixed_draws = h.read(fixed / "draw-recipe.json")["draws"]
    replay = [by_id[d["item_id"]] for d in fixed_draws if d["cohort"] != "video"]
    require(len(replay) == 600, "fixed600replay draws")
    require(
        all(canonical_sample_sha(by_id[d["item_id"]]) == d["sample_sha256"] for d in fixed_draws),
        "fixed replay canonical/order",
    )
    versions = h.cohorts(replay, inventory)
    require(
        collections.Counter(versions[s.meta.data_version] for s in replay)
        == {"crossmodal": 200, "audio": 100, "image": 100, "screenshot": 100, "text": 100},
        "replay cohorts",
    )
    draws = []
    for step in range(100):
        draws.extend(video_groups[step])
        draws.extend(replay[step * 6 : step * 6 + 6])
    arranged = inverse_order(draws)
    order = list(range(1200))
    random.Random(42).shuffle(order)
    require([arranged[i].id for i in order] == [r.id for r in draws], "actual shuffle roundtrip")
    for step in range(100):
        actual_video = [arranged[order[step * 12 + j]] for j in range(6)]
        expected = dict(zip(GROUPS, (2, 2, 1, 1) if step % 2 == 0 else (1, 1, 2, 2), strict=True))
        require(
            collections.Counter(classifications[r.id] for r in actual_video) == expected,
            "actual per-step program quotas",
        )
    prefixes = {}
    for steps in [10, 20, 100]:
        video = [draws[i] for i in range(steps * 12) if i % 12 < 6]
        prefixes[str(steps)] = {
            g: dict(
                collections.Counter(
                    r.questions[0].target.value for r in video if classifications[r.id] == g
                )
            )
            for g in GROUPS
        }
        require(
            all(sum(prefixes[str(steps)][g].values()) == steps * 6 // 4 for g in GROUPS),
            "balanced actual program prefix",
        )
    versions[VERSION] = "video"
    require(len(draws) == len({s.id for s in draws}) == 1200, "unique120draws")
    for step in range(100):
        group = draws[step * 12 : step * 12 + 12]
        require(
            collections.Counter(versions[s.meta.data_version] for s in group)
            == {"video": 6, "crossmodal": 2, "audio": 1, "image": 1, "screenshot": 1, "text": 1},
            "per-step replay layout",
        )
        require(
            collections.Counter(s.questions[0].target.value for s in group[:6])
            == {"yes": 3, "no": 3},
            "per-step video polarity",
        )
    replay_ledger = h.exact_ledger(replay, h.rows(fixed / "allocation-ledger.jsonl"))
    ledger = h.exact_ledger(arranged, generated_ledger + replay_ledger)
    h.write(out / "train.jsonl", data_bytes(arranged).decode())
    h.write(
        out / "allocation-ledger.jsonl",
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ledger),
    )
    verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    config = h.read(original / "config-ce.json")
    config.update(sample_count=1200, steps=100, checkpoint_every=5, midway_step=10)
    require(
        config["sample_count"] == 1200 and config["steps"] == 100 and config["accumulation"] == 12,
        "fixed CE loop",
    )
    require(
        not any(
            config.get(k, 0) for k in ["pair_loss_weight", "reference_kl_weight", "brier_weight"]
        ),
        "pure CE",
    )
    require(
        config["seed"] == 42
        and config["video_num_frames"] == 16
        and config["image_max_soft_tokens"] == 1120,
        "fixed media/seed budget",
    )
    require(
        config["rank"] == 16 and config["alpha"] == 32 and config["learning_rate"] == 2e-5,
        "fixed parent CE optimization",
    )
    parent = root / "artifacts/s4-temporal-coverage-train-001/adapter"
    require(
        h.sha(parent / "adapter_model.safetensors")
        == "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
        and h.sha(parent / "adapter_config.json")
        == "a4fc244afa7a308262478aad466e250592fc15176d447ea12390d8bf321539ad",
        "proposed frozen f2db identity",
    )
    config["expected_data_versions"] = sorted(versions)
    smoke_samples(
        LoopConfig.model_validate(config),
        out / "train.jsonl",
        expected_data_versions=config["expected_data_versions"],
    )
    h.write(out / "config-ce.json", config)
    h.write(
        out / "draw-recipe.json",
        {
            "seed": 42,
            "global_steps": 100,
            "accumulation": 12,
            "per_step": {"video": 6, "crossmodal": 2, "old": 4},
            "draws": [
                {
                    "draw": i,
                    "step": i // 12,
                    "offset": i % 12,
                    "item_id": s.id,
                    "sample_sha256": canonical_sample_sha(s),
                    "cohort": versions[s.meta.data_version],
                    "program_stratum": classifications.get(s.id),
                    "target": s.questions[0].target.value,
                }
                for i, s in enumerate(draws)
            ],
        },
    )
    combined = inventory.model_dump(mode="json", exclude_none=True)
    replay_ids = {s.id for s in draws}
    for i, artifact in enumerate(inventory.samples):
        if (
            artifact.expected_pool == "train"
            and artifact.exposure == "current"
            and any(json.loads(line)["id"] in replay_ids for line in bound_lines(artifact))
        ):
            combined["samples"][i]["exposure"] = "historical"
    combined["samples"].append(
        {
            "path": str(out / "train.jsonl"),
            "sha256": h.sha(out / "train.jsonl"),
            "expected_pool": "train",
            "exposure": "current",
            "cohorts_by_data_version": versions,
        }
    )
    # Reuse the original exact cross component ledger without inventing new samples.
    require(
        h.sha(fixed / "components.jsonl")
        == h.read(fixed / "report.json")["output_sha256"]["components.jsonl"],
        "fixed component chain",
    )
    h.write(out / "components.jsonl", (fixed / "components.jsonl").read_text())
    existing_components = {row["sha256"] for row in combined.get("component_ledgers", [])}
    if h.sha(out / "components.jsonl") not in existing_components:
        combined.setdefault("component_ledgers", []).append(
            {"path": str(out / "components.jsonl"), "sha256": h.sha(out / "components.jsonl")}
        )
    h.write(out / "inventory-manifest.json", combined)
    check_no_updates()
    audit = audit_manifest(out / "inventory-manifest.json")
    h.write(out / "inventory-audit.json", audit.model_dump(mode="json"))
    check_no_updates()
    require(
        h.sha(exposure / "report.json") == expected_exposure_sha
        and h.sha(exposure / "inventory-manifest.json") == inventory_sha,
        "inventory base changed",
    )
    require(audit.outcome == "isolated", "full inventory isolation")
    summary.update(
        status="data_ready_only",
        exit_code=0,
        inventory_isolation=audit.outcome,
        source_admission="existing ADR0005 CC0 full8299 TRAIN scope",
        source_sha256={
            "train": inspection.POOL_SHA,
            "questions": inspection.QUESTIONS_SHA,
            "annotations": inspection.ANNOTATIONS_SHA,
            "readme": inspection.README_SHA,
        },
        exposure_base={
            "path": str(exposure),
            "report_sha256": expected_exposure_sha,
            "manifest_sha256": inventory_sha,
        },
        replay_report_sha256=h.sha(fixed / "report.json"),
        replay_draws=600,
        replay_unique_samples=h.read(fixed / "report.json")["replay_unique_samples"],
        invocation_steps_initial=10,
        automatic_full_run_authorized=False,
        global_steps=100,
        root_observation_window_seconds=1200,
        root_usual_max_steps_before_reconsideration=40,
        per_step={"video": 6, "crossmodal": 2, "audio": 1, "image": 1, "screenshot": 1, "text": 1},
        draw_order_canonical_sha256=h.digest(data_bytes(draws)),
        video_rows=600,
        media_verified_unique=media_verified,
        program_group_priority=list(GROUPS),
        actual_prefix_strata=prefixes,
        selected_before_after_collision_count=sum(
            m["program_stratum"] == "before_after" and m["has_collision_operator"] for m in metadata
        ),
        objective="original CE only; no pairs/variants/new labels",
        checkpoint_io_boundary="checkpoint_every5 versus matched control10; objective unchanged",
        proposed_init_adapter={
            "path": str(root / "artifacts/s4-temporal-coverage-train-001/adapter"),
            "weights_sha256": "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
            "rank": 16,
            "alpha": 32,
        },
        duration_seconds=time.monotonic() - started,
        helper_sha256=h.sha(Path(__file__)),
        reused_helper_sha256={
            name: h.sha(Path(__file__).with_name(name + ".py"))
            for name in ["inspect-temporal-order-contrast", "prepare-claim-pair-pilot"]
        },
        output_sha256={p.name: h.sha(p) for p in out.iterdir() if p.is_file()},
    )
    h.write(out / "report.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--exposure-dir", type=Path, required=True)
    ap.add_argument("--exposure-report-sha256", required=True)
    a = ap.parse_args()
    already_existed = a.output_dir.exists()
    try:
        result = prepare(
            a.root.resolve(),
            a.official.resolve(),
            a.output_dir.resolve(),
            a.exposure_dir.resolve(),
            a.exposure_report_sha256,
        )
    except Exception as exc:
        result = {
            "status": "rejected",
            "exit_code": 1,
            "category": str(exc),
            "training_authorized": False,
            "gpu_used": False,
        }
        if not already_existed:
            a.output_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
            if not (a.output_dir / "failure.json").exists():
                h.write(a.output_dir / "failure.json", result)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
