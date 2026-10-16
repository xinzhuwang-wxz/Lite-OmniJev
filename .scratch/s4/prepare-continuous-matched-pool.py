"""Fixed global matched-pair TRAIN course; CPU only, no automatic100step run."""

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
NAMESPACE = "42:s4-continuous-matched-pool-001:"


def model_input_key(sample):
    prefix = sample.prefix.model_dump(mode="json")
    prefix.pop("media")
    question = sample.questions[0].model_dump(mode="json")
    question.pop("target")
    return h.digest(json.dumps({"prefix": prefix, "question": question}, sort_keys=True).encode())


def validate_pair(left, right, left_program, right_program):
    require(left.meta.pool == right.meta.pool == "train", "TRAIN pair")
    require(left.meta.family_id != right.meta.family_id, "different families")
    require(left.prefix.media[0].sha256 != right.prefix.media[0].sha256, "different media SHA")
    require(
        {left.questions[0].target.value, right.questions[0].target.value} == {"yes", "no"},
        "opposite official gold",
    )
    require(
        left.questions[0].instruction == right.questions[0].instruction
        and left_program == right_program,
        "exact original text/program",
    )
    require(model_input_key(left) == model_input_key(right), "nonmedia prefix/question equality")


def choose_pairs(groups, limit=300):
    result = []
    seen = set()
    for key in sorted(
        groups, key=lambda k: h.digest((NAMESPACE + json.dumps(k, sort_keys=True)).encode())
    ):
        rows = sorted(groups[key], key=lambda r: h.digest((NAMESPACE + r.id).encode()))
        yes = [r for r in rows if r.questions[0].target.value == "yes"]
        no = [r for r in rows if r.questions[0].target.value == "no"]
        for left in yes:
            if left.meta.family_id in seen:
                continue
            right = next(
                (
                    r
                    for r in no
                    if r.meta.family_id not in seen
                    and r.meta.family_id != left.meta.family_id
                    and r.prefix.media[0].sha256 != left.prefix.media[0].sha256
                ),
                None,
            )
            if right is None:
                continue
            pair = [left, right]
            if int(h.digest((NAMESPACE + left.id + right.id).encode()), 16) % 2:
                pair.reverse()
            result.append(pair)
            seen.update(r.meta.family_id for r in pair)
            if len(result) == limit:
                return result
    return result


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
    eligible = [r for r in source_samples if r.meta.family_id not in excluded]
    scenes = inspection.selected_scenes(
        official / "questions/train.json", {int(r.meta.family_id.split(":")[1]) for r in eligible}
    )
    groups = collections.defaultdict(list)
    source_questions = {}
    for sample in eligible:
        index = int(sample.meta.family_id.split(":")[1])
        question = next(
            q
            for q in scenes[index]["questions"]
            if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
        )
        if not native.native_program(question["program"]):
            continue
        require(
            question["question"] == sample.questions[0].instruction
            and question["answer"].casefold() == sample.questions[0].target.value,
            "original text/gold",
        )
        require(
            source_hash_for_video({"scene_index": index}, question, sample.prefix.media[0].sha256)
            == sample.meta.source_sha256,
            "official program/media source binding",
        )
        key = (question["question"], tuple(question["program"]), model_input_key(sample))
        groups[key].append(sample)
        source_questions[sample.id] = question
    pairs = choose_pairs(groups)
    selected = [r for pair in pairs for r in pair]
    counts = collections.Counter(r.questions[0].target.value for r in selected)
    out.mkdir(mode=0o700)
    summary = {
        "status": "insufficient_exact_pairs",
        "exit_code": 1,
        "pair_count": len(pairs),
        "video_families": len({r.meta.family_id for r in selected}),
        "gold_counts": dict(counts),
        "exclusions": exclusion_report,
        "training_authorized": False,
        "pending_parent_selection": True,
        "parent_model": None,
        "gpu_used": False,
        "cost": None,
    }
    if len(pairs) < 300:
        h.write(out / "report.json", summary)
        return summary
    require(len(selected) == len({r.meta.family_id for r in selected}) == 600, "unique60families")
    metadata = []
    media_verified = 0
    for i, (left, right) in enumerate(pairs):
        validate_pair(
            left, right, source_questions[left.id]["program"], source_questions[right.id]["program"]
        )
        for sample in [left, right]:
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
        program = source_questions[left.id]["program"]
        metadata.append(
            {
                "pair": i,
                "metadata_only": True,
                "model_nonmedia_question_sha256": model_input_key(left),
                "official_program_sha256": h.digest(json.dumps(program, sort_keys=True).encode()),
                "text_sha256": h.digest(left.questions[0].instruction.encode()),
                "members": [
                    {
                        "item_id": r.id,
                        "family_id": r.meta.family_id,
                        "sample_sha256": canonical_sample_sha(r),
                        "official_source_sha256": r.meta.source_sha256,
                        "media_sha256": r.prefix.media[0].sha256,
                        "original_gold": r.questions[0].target.value,
                    }
                    for r in [left, right]
                ],
            }
        )
    video_groups = [[r for pair in pairs[i : i + 3] for r in pair] for i in range(0, 300, 3)]
    generated_ledger = h.exact_ledger(
        selected, [r for r in h.rows(source / "allocation-ledger.jsonl") if r["pool"] == "train"]
    )
    h.write(
        out / "matched-pair-metadata.json",
        {
            "target_source": "original official gold only; pairs are metadata, no pairloss",
            "metadata_only": True,
            "rows": metadata,
        },
    )
    fixed = root / "artifacts/s4-continuous-main-pool-001"
    require(
        h.sha(fixed / "report.json")
        == "bdf25dbe6955b983d263d13d3436f930481dc908241ef12b56526545156b1cc9",
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
        for offset in [0, 2, 4]:
            left, right = [arranged[order[step * 12 + offset + j]] for j in [0, 1]]
            validate_pair(
                left,
                right,
                source_questions[left.id]["program"],
                source_questions[right.id]["program"],
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
    config.update(sample_count=1200, steps=100, checkpoint_every=10, midway_step=10)
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
        replay_unique_samples=len(
            {d["source_item_id"] for d in fixed_draws if d["cohort"] != "video"}
        ),
        invocation_steps_initial=10,
        automatic_full_run_authorized=False,
        global_steps=100,
        root_observation_window_seconds=1200,
        root_usual_max_steps_before_reconsideration=50,
        per_step={"video": 6, "crossmodal": 2, "audio": 1, "image": 1, "screenshot": 1, "text": 1},
        draw_order_canonical_sha256=h.digest(data_bytes(draws)),
        video_rows=600,
        media_verified_unique=media_verified,
        original_postfix="exact native in/out before/after equality; official gold unchanged",
        pair_metadata_only=True,
        nonmedia_prefix_question_equal=True,
        actual_draw_pair_offsets=[0, 2, 4],
        prepared_input_limitation="nonmedia fields verified; no tensor preparation or GPU",
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
