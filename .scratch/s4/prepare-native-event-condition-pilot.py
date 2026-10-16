"""Native official TRAIN event-condition pilot; camera proxies never change gold."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import subprocess
import time
import zipfile
from pathlib import Path

from data.schema import Sample
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines
from scripts.gold_media_bundle import media_path
from scripts.s0_media_dev import source_hash_for_video
from scripts.s1_text_pools import canonical_sample_sha
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples

VERSION = "clevrer-train-video-noul-s3-pools-v1"
NAMESPACE = "42:s4-native-event-condition-001:"
REPLAY_SHA = "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b"


def load_helper(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), Path(__file__).with_name(name + ".py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load_helper("prepare-claim-pair-pilot")
inspection = load_helper("inspect-temporal-order-contrast")


def require(ok, why):
    if not ok:
        raise ValueError(why)


proxy = load_helper("diagnose-temporal-observability")


def native_program(program):
    """Validate exact restricted native postfix grammar, not English keywords."""
    properties = {0: {"color": "red", "material": "metal", "shape": "sphere"}}
    try:
        if program[:2] != ["events", "events"]:
            return False
        _, end = proxy.object_selector(program, 2, properties)
        if program[end : end + 3] not in [
            ["unique", kind, "unique"] for kind in ["filter_in", "filter_out"]
        ]:
            return False
        if program[end + 3] not in {"filter_before", "filter_after"}:
            return False
        _, target_end = proxy.object_selector(program, end + 4, properties)
        return program[target_end:] in [[kind, "exist"] for kind in ["filter_in", "filter_out"]]
    except (IndexError, KeyError, TypeError):
        return False


def proxy_metadata(program, annotation, gold):
    require(native_program(program), "unsupported native postfix")
    result = proxy.diagnose(program, annotation, [int(i * 127 / 15) for i in range(16)], gold)
    order = result.get("critical_order_proxy")
    return {
        "role": "sampling_metadata_only_not_target_or_official_execution",
        "schema": "camera-boolean-transition-v1",
        "initial_boundary": "frame0 is not inferred to be an entry event",
        "branch": (
            "unknown"
            if order is None
            else "target_transition_absent"
            if order["target_events"] == 0
            else "target_transition_present"
        ),
        "gold_agrees_with_proxy": None
        if order is None
        else order["gold_agrees_with_transition_proxy"],
        "diagnostic": result,
        "official_target_unchanged": gold,
    }


def actual_video_families(samples, seed, steps, accumulation):
    require(type(steps) is int and steps >= 0, "actual steps")
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    return {
        samples[order[i % len(order)]].meta.family_id
        for i in range(steps * accumulation)
        if samples[order[i % len(order)]].meta.modality == ["video"]
    }


def bound_rows(directory):
    receipt = h.read(directory / "report.json")
    for name, digest in receipt["output_sha256"].items():
        require(
            Path(name).name == name and h.sha(directory / name) == digest, "bound output changed"
        )
    rows = [Sample.model_validate(r) for r in h.rows(directory / "train.jsonl")]
    require(all(s.meta.pool == "train" for s in rows), "TRAIN only")
    return rows


def exclusions(root):
    lineage_path = root / "reports/s4-ce0-video-exposure-001.json"
    require(
        h.sha(lineage_path) == "cb9574e06fc9950198f703ab9107d6d2d42824a5d2399ded2a26a146dd88cbfa",
        "lineage SHA",
    )
    lineage = h.read(lineage_path)
    for stage in lineage["stages"]:
        require(
            h.sha(Path(stage["report"])) == stage["report_sha256"]
            and h.sha(Path(stage["data"])) == stage["data_sha256"],
            "lineage binding",
        )
    first_dir = root / "artifacts/s4-temporal-coverage-pilot-001"
    require(
        h.sha(first_dir / "report.json")
        == "1e7f877e04ea311d29ebc1fe1711f31aa2cc91034f8794b666850bcf5f2fff19",
        "first SHA",
    )
    first = bound_rows(first_dir)
    old = set(lineage["lineage_unique_video_families"]) | {
        s.meta.family_id for s in first if s.meta.modality == ["video"]
    }
    require(len(old) == 408, "native candidate lineage408")
    runtime_path = root / "reports/s4-continuous-main-pool-runtime-step0020.json"
    runtime = h.read(runtime_path)
    require(
        runtime["status"] == "paused"
        and runtime["exit_code"] == 0
        and runtime["steps_completed"] == 20
        and runtime["invocation"]["next_draw"] == 240,
        "continuous actual20steps required",
    )
    continuous_dir = root / "artifacts/s4-continuous-main-pool-001"
    continuous = bound_rows(continuous_dir)
    require(
        h.sha(continuous_dir / "train.jsonl") == runtime["data_sha256"], "actual continuous data"
    )
    cf = actual_video_families(continuous, runtime["config"]["seed"], 20, 12)
    learn_dir = root / "artifacts/s4-train-learnability-001"
    learn = bound_rows(learn_dir)
    require(len(learn) == 12, "learnability12")
    lf = {s.meta.family_id for s in learn}
    return old | cf | lf, {
        "lineage408": len(old),
        "continuous_actual_draws": 240,
        "continuous_video_families": len(cf),
        "learnability_families": len(lf),
        "union": len(old | cf | lf),
        "continuous_runtime_sha256": h.sha(runtime_path),
        "continuous_data_sha256": runtime["data_sha256"],
        "learnability_report_sha256": h.sha(learn_dir / "report.json"),
    }


def check_no_updates():
    commands = subprocess.check_output(["ps", "-eo", "args"], text=True)
    active = [
        line
        for line in commands.splitlines()
        if any(
            token in line
            for token in ["-m scripts.train_", "-m scripts.train_reference", " train_reference.py "]
        )
    ]
    require(not active, "update-bearing process active; defer inventory")


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
    eligible = [
        s
        for s in source_samples
        if s.meta.family_id not in excluded
        and set(s.questions[0].instruction.lower().split()) & {"before", "after"}
    ]
    candidates = sorted(eligible, key=lambda s: h.digest((NAMESPACE + s.id).encode()))
    scenes = inspection.selected_scenes(
        official / "questions/train.json", {int(s.meta.family_id.split(":")[1]) for s in candidates}
    )
    selected, metadata, seen = [], [], set()
    rejected, counts = collections.Counter(), collections.Counter()
    media_verified = 0
    with zipfile.ZipFile(official / "annotations/train/annotation_train.zip") as zipped:
        members = {
            int(Path(n).stem.split("_")[-1]): n for n in zipped.namelist() if n.endswith(".json")
        }
        for sample in candidates:
            if sample.meta.family_id in seen:
                continue
            index = int(sample.meta.family_id.split(":")[1])
            question = next(
                q
                for q in scenes[index]["questions"]
                if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
            )
            if not native_program(question["program"]):
                rejected["not_native_in_out_relation"] += 1
                continue
            gold = sample.questions[0].target.value
            if counts[gold] >= 30:
                continue
            require(
                question["question"] == sample.questions[0].instruction
                and question["answer"].casefold() == gold,
                "original question/gold",
            )
            require(
                source_hash_for_video(
                    {"scene_index": index}, question, sample.prefix.media[0].sha256
                )
                == sample.meta.source_sha256,
                "original program binding",
            )
            info = zipped.getinfo(members[index])
            require(info.file_size < 5_000_000 and not info.flag_bits & 1, "bounded annotation")
            raw = zipped.read(info)
            annotation = json.loads(raw)
            require(annotation["scene_index"] == index, "annotation identity")
            try:
                detail = proxy_metadata(question["program"], annotation, gold)
            except (ValueError, KeyError, StopIteration, IndexError):
                detail = {
                    "branch": "unknown",
                    "gold_agrees_with_proxy": None,
                    "role": "sampling_metadata_only_not_target_or_official_execution",
                }
            require(
                h.sha(
                    media_path(
                        [Path(source_report["source"]["media_root"])], sample.prefix.media[0].uri
                    )
                )
                == sample.prefix.media[0].sha256,
                "media SHA",
            )
            selected.append(sample)
            metadata.append(
                {
                    "item_id": sample.id,
                    "family_id": sample.meta.family_id,
                    "sample_sha256": canonical_sample_sha(sample),
                    "program_sha256": h.digest(
                        json.dumps(question["program"], sort_keys=True).encode()
                    ),
                    "annotation_member": members[index],
                    "annotation_sha256": h.digest(raw),
                    "proxy": detail,
                }
            )
            seen.add(sample.meta.family_id)
            counts[gold] += 1
            media_verified += 1
            if len(selected) == 60:
                break
    out.mkdir(mode=0o700)
    summary = {
        "status": "insufficient_native_families",
        "exit_code": 1,
        "video_families": len(selected),
        "gold_counts": dict(counts),
        "rejected": dict(rejected),
        "exclusions": exclusion_report,
        "training_authorized": False,
        "pending_parent_selection": True,
        "parent_model": None,
        "gpu_used": False,
        "cost": None,
    }
    if len(selected) < 60:
        h.write(out / "report.json", summary)
        return summary
    require(counts == {"yes": 30, "no": 30}, "balanced native gold")
    # Stable hash order within polarity; each actual trainer block has3yes/3no.
    yes = [s for s in selected if s.questions[0].target.value == "yes"]
    no = [s for s in selected if s.questions[0].target.value == "no"]
    video_groups = [yes[i : i + 3] + no[i : i + 3] for i in range(0, 30, 3)]
    for i, group in enumerate(video_groups):
        random.Random(42 + i).shuffle(group)
    generated_ledger = h.exact_ledger(
        selected, [r for r in h.rows(source / "allocation-ledger.jsonl") if r["pool"] == "train"]
    )
    h.write(
        out / "native-program-metadata.json",
        {"target_source": "original official gold only", "metadata_only": True, "rows": metadata},
    )
    by_id = {s.id: s for s in [Sample.model_validate(x) for x in h.rows(original / "train.jsonl")]}
    original_draws = h.read(original / "draw-recipe.json")["draws"]
    replay = [by_id[d["item_id"]] for d in original_draws if d["cohort"] != "video"]
    require(len(replay) == 60, "original replay60")
    versions = h.cohorts(replay, inventory)
    require(
        collections.Counter(versions[s.meta.data_version] for s in replay)
        == {"crossmodal": 20, "audio": 10, "image": 10, "screenshot": 10, "text": 10},
        "replay cohorts",
    )
    draws = []
    for step in range(10):
        draws.extend(video_groups[step])
        draws.extend(replay[step * 6 : step * 6 + 6])
    arranged = h.inverse_order(draws)
    versions[VERSION] = "video"
    require(len(draws) == len({s.id for s in draws}) == 120, "unique120draws")
    for step in range(10):
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
    replay_ledger = h.exact_ledger(replay, h.rows(original / "allocation-ledger.jsonl"))
    ledger = h.exact_ledger(arranged, generated_ledger + replay_ledger)
    h.write(out / "train.jsonl", data_bytes(arranged).decode())
    h.write(
        out / "allocation-ledger.jsonl",
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ledger),
    )
    verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    config = h.read(original / "config-ce.json")
    require(
        config["sample_count"] == 120 and config["steps"] == 10 and config["accumulation"] == 12,
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
            "steps": 10,
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
        replay_report_sha256=REPLAY_SHA,
        per_step={"video": 6, "crossmodal": 2, "audio": 1, "image": 1, "screenshot": 1, "text": 1},
        draw_order_canonical_sha256=h.digest(data_bytes(draws)),
        video_rows=60,
        media_verified_unique=media_verified,
        original_postfix="native syntax validated; source-bound; no official executor available",
        original_official_gold="unchanged; proxy is never a target or admission/quality gate",
        strict_official_absence_status=(
            "not_proven; cached README lacks executor boundary semantics"
        ),
        proxy_branch_counts=dict(collections.Counter(m["proxy"]["branch"] for m in metadata)),
        proxy_gold_agreement_counts=dict(
            collections.Counter(str(m["proxy"]["gold_agrees_with_proxy"]) for m in metadata)
        ),
        observability="camera Boolean is not occlusion/pixel/model visibility proof",
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
