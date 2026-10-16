"""One bounded CPU TRAIN collision-order pilot; no parent choice or GPU authorization."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
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

VERSION = "clevrer-train-collision-order-v1"
NAMESPACE = "42:s4-order-pilot-001:"
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


def safe_pairs(annotation):
    """Only unambiguous, single-occurrence visible collisions at least16frames apart."""
    inspection.inspect_annotation(annotation)  # Shared structural/visibility sanity checks.
    props = {
        o["object_id"]: (o["color"], o["material"], o["shape"])
        for o in annotation["object_property"]
    }
    descriptors = collections.Counter(props.values())
    events = collections.defaultdict(list)
    for event in annotation["collision"]:
        events[tuple(sorted(event["object_ids"]))].append(event["frame_id"])
    # Refuse the entire scene rather than silently disambiguating ambiguous/repeated events.
    if any(n != 1 for n in descriptors.values()) or any(len(v) != 1 for v in events.values()):
        return []
    frames = {
        f["frame_id"]: {o["object_id"]: o["inside_camera_view"] for o in f["objects"]}
        for f in annotation["motion_trajectory"]
    }
    visible = {
        pair: values[0]
        for pair, values in events.items()
        if all(frames[values[0]].get(o) is True for o in pair)
    }
    result = []
    for left, right in itertools.combinations(sorted(visible), 2):
        a, b = visible[left], visible[right]
        if abs(a - b) < 16:
            continue
        result.append(
            {
                "events": [
                    {
                        "object_ids": list(pair),
                        "descriptors": [list(props[o]) for o in pair],
                        "frame": visible[pair],
                    }
                    for pair in [left, right]
                ],
                "schema_version": "clevrer-collision-order-program-v1",
            }
        )
    return result


def event_text(event):
    return "the collision between the " + " and the ".join(
        " ".join(descriptor) for descriptor in event["descriptors"]
    )


def make_pair(source, selected, annotation_sha, member_name, source_program_sha):
    """Evaluate both orders independently; internal object/frame identities never enter prompt."""
    require(source.meta.pool == "train", "TRAIN source required")
    events = selected["events"]
    require(abs(events[0]["frame"] - events[1]["frame"]) >= 16, "event separation")
    group = h.digest((source.meta.family_id + json.dumps(selected, sort_keys=True)).encode())
    variants = []
    for order in [(0, 1), (1, 0)]:
        left, right = [events[i] for i in order]
        truth = "yes" if left["frame"] < right["frame"] else "no"
        program = {
            "schema_version": selected["schema_version"],
            "operation": "less_frame",
            "events": [left, right],
            "result": truth,
            "requirements": "unique collision each; unique descriptors; visible; gap>=16",
        }
        provenance = {
            "official_questions_sha256": inspection.QUESTIONS_SHA,
            "annotations_zip_sha256": inspection.ANNOTATIONS_SHA,
            "annotation_member": member_name,
            "annotation_sha256": annotation_sha,
            "source_sample_sha256": canonical_sample_sha(source),
            "original_postfix_program_sha256": source_program_sha,
            "original_postfix_execution": "not_executed_source_binding_only",
            "program": program,
            "program_sha256": h.digest(json.dumps(program, sort_keys=True).encode()),
        }
        value = source.model_dump(mode="json")
        value["id"] = f"clevrer-order-{source.meta.family_id.split(':')[1]}-{group[:12]}-{order[0]}"
        value["questions"] = [
            {
                "qid": "noul",
                "type": "noul",
                "instruction": f"Does {event_text(left)} occur before {event_text(right)}?",
                "target": {
                    "kind": "hard",
                    "value": truth,
                    "source": "clevrer_collision_annotation_less_frame_v1",
                },
            }
        ]
        value["meta"].update(
            data_version=VERSION,
            source_sha256=h.digest(json.dumps(provenance, sort_keys=True).encode()),
        )
        sample = Sample.model_validate(value)
        ledger = {
            "item_id": sample.id,
            "pool": "train",
            "family_id": source.meta.family_id,
            "family_hash": h.digest(source.meta.family_id.encode()),
            "source_dataset": source.meta.source_dataset,
            "source_revision": source.meta.source_revision,
            "license": "CC0-1.0",
            "original_sample_sha256": canonical_sample_sha(source),
            "derived_sample_sha256": canonical_sample_sha(sample),
            "media_sha256": source.prefix.media[0].sha256,
            "allocation_key_sha256": h.digest((group + str(order)).encode()),
            "group_id": group,
            "program_provenance": provenance,
        }
        variants.append((sample, ledger))
    require({s.questions[0].target.value for s, _ in variants} == {"yes", "no"}, "gold pair")
    if int(h.digest((NAMESPACE + "position:" + source.meta.family_id).encode()), 16) % 2:
        variants.reverse()
    return variants


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
    eligible = [
        s
        for s in source_samples
        if set(s.questions[0].instruction.lower().split()) & {"before", "after", "when", "while"}
    ]
    candidates = []
    seen = set()
    for sample in sorted(eligible, key=lambda s: h.digest((NAMESPACE + s.meta.family_id).encode())):
        if sample.meta.family_id not in seen:
            candidates.append(sample)
            seen.add(sample.meta.family_id)
        if len(candidates) == 64:
            break
    scenes = inspection.selected_scenes(
        official / "questions/train.json", {int(s.meta.family_id.split(":")[1]) for s in candidates}
    )
    original_ledger = h.exact_ledger(
        candidates, [r for r in h.rows(source / "allocation-ledger.jsonl") if r["pool"] == "train"]
    )
    require(len(original_ledger) == len(candidates), "source family allocations")
    video_pairs, generated_ledger, rejected = [], [], collections.Counter()
    gaps, checked, media_verified = [], 0, 0
    with zipfile.ZipFile(official / "annotations/train/annotation_train.zip") as zipped:
        for sample in candidates:
            checked += 1
            index = int(sample.meta.family_id.split(":")[1])
            scene = scenes[index]
            qid = int(sample.id.rsplit("-", 1)[-1])
            question = next(q for q in scene["questions"] if q["question_id"] == qid)
            require(
                question["question"] == sample.questions[0].instruction
                and question["answer"].casefold() == sample.questions[0].target.value,
                "original source gold binding",
            )
            require(
                source_hash_for_video(
                    {"scene_index": index}, question, sample.prefix.media[0].sha256
                )
                == sample.meta.source_sha256,
                "original source program binding",
            )
            members = [n for n in zipped.namelist() if n.endswith(f"/annotation_{index:05d}.json")]
            require(len(members) == 1, "annotation member identity")
            info = zipped.getinfo(members[0])
            require(info.file_size < 5_000_000 and not info.flag_bits & 1, "bounded annotation")
            raw = zipped.read(info)
            annotation = json.loads(raw)
            require(
                annotation["scene_index"] == index
                and annotation["video_filename"] == scene["video_filename"],
                "annotation scene",
            )
            choices = safe_pairs(annotation)
            if not choices:
                rejected["unsafe_or_no_eligible_event_pair"] += 1
                continue
            selected = min(
                choices,
                key=lambda c: h.digest(
                    (NAMESPACE + sample.meta.family_id + json.dumps(c, sort_keys=True)).encode()
                ),
            )
            pair = make_pair(
                sample,
                selected,
                h.digest(raw),
                members[0],
                h.digest(json.dumps(question["program"], sort_keys=True).encode()),
            )
            require(
                h.sha(
                    media_path(
                        [Path(source_report["source"]["media_root"])], sample.prefix.media[0].uri
                    )
                )
                == sample.prefix.media[0].sha256,
                "media SHA",
            )
            media_verified += 1
            video_pairs.append([s for s, _ in pair])
            generated_ledger.extend(row for _, row in pair)
            gaps.append(abs(selected["events"][0]["frame"] - selected["events"][1]["frame"]))
            if len(video_pairs) == 30:
                break
    out.mkdir(mode=0o700)
    summary = {
        "status": "insufficient_safe_families",
        "exit_code": 1,
        "checked_family_count": checked,
        "max_checked": 64,
        "safe_family_count": len(video_pairs),
        "rejected": dict(rejected),
        "training_authorized": False,
        "pending_parent_selection": True,
        "parent_model": None,
        "gpu_used": False,
        "cost": None,
    }
    if len(video_pairs) < 30:
        h.write(out / "report.json", summary)
        return summary
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
        draws.extend(s for pair in video_pairs[step * 3 : step * 3 + 3] for s in pair)
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
    replay_ids = {s.id for s in replay}
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
        video_families=30,
        independently_evaluated_programs=60,
        media_verified_unique=media_verified,
        first_position_yes=sum(p[0].questions[0].target.value == "yes" for p in video_pairs),
        first_position_no=sum(p[0].questions[0].target.value == "no" for p in video_pairs),
        gap_frames={"min": min(gaps), "max": max(gaps)},
        original_postfix="source binding only; not re-executed",
        observability="event visibility at annotated frame and gap>=16 verified; actual uniform16 "
        "event capture/occlusion not verified; no guarantee invented",
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
