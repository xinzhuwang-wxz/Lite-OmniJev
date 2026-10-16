"""Read-only public DEV50 time-coverage proxies; never claim annotation means pixel visibility."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import time
import zipfile
from pathlib import Path

from data.schema import Sample, answer_tokens
from scripts.s0_development_baselines import sha256_json
from scripts.s0_media_dev import source_hash_for_video
from scripts.s1_text_pools import canonical_sample_sha

PREP_SHA = "38981eb5da7fdc1ef7658e1c6c37c051e169457859e07a398dfb65df7354f0eb"
RECORD_SHA = "7e14478e05f62beb85127e40a6e9e13ba38f5b3c1e18ffa3b21c52a8282e0ff3"
CANDIDATE_RECEIPT_SHA = "4a21afc14d86c31e49aa1004f51047b00901a6695206a6a5e0c321fb1fbf9df6"
DATA_SHA = "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2"


def inspector():
    path = Path(__file__).with_name("inspect-temporal-order-contrast.py")
    spec = importlib.util.spec_from_file_location("order_inspect", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def object_selector(program, start, properties):
    """Only literal attribute filters; reject unknown syntax rather than execute full postfix."""
    if program[start] != "objects":
        raise ValueError("selector must begin with objects")
    chosen = set(properties)
    index = start + 1
    while index + 1 < len(program) and program[index + 1] in {
        "filter_color",
        "filter_material",
        "filter_shape",
    }:
        attribute = program[index + 1].removeprefix("filter_")
        chosen = {o for o in chosen if properties[o][attribute] == program[index]}
        index += 2
    return chosen, index


def transitions(annotation, object_id, kind):
    """Camera Boolean transition proxy, not an official in/out executor."""
    frames = annotation["motion_trajectory"]
    if [f["frame_id"] for f in frames] != list(range(128)):
        raise ValueError("requires contiguous128annotation frames")
    states = [
        next(o["inside_camera_view"] for o in f["objects"] if o["object_id"] == object_id)
        for f in frames
    ]
    if any(type(state) is not bool for state in states):
        raise ValueError("camera state must be Boolean")
    if kind == "filter_in":
        return [i for i in range(1, 128) if not states[i - 1] and states[i]]
    if kind == "filter_out":
        return [i for i in range(1, 128) if states[i - 1] and not states[i]]
    raise ValueError("unknown transition type")


def frame_coverage(frame, indices):
    if type(frame) is not int or not 0 <= frame < 128:
        raise ValueError("invalid frame")
    if (
        len(indices) != 16
        or indices != sorted(set(indices))
        or indices[0] != 0
        or indices[-1] != 127
    ):
        raise ValueError("requires complete uniform16indices")
    nearest = min(indices, key=lambda i: (abs(i - frame), i))
    left = max(i for i in indices if i <= frame)
    right = min(i for i in indices if i >= frame)
    return {
        "exact_sample": frame in indices,
        "nearest_distance": abs(nearest - frame),
        "nearest_index": nearest,
        "bracket_width": right - left,
    }


def diagnose(program, annotation, indices, gold):
    properties = {o["object_id"]: o for o in annotation["object_property"]}
    object_starts = [i for i, t in enumerate(program) if t == "objects"]
    relation = next((t for t in program if t in {"filter_before", "filter_after"}), None)
    category = (
        "event_relation"
        if relation
        else ("event_conditioned_motion" if "events" in program else "endpoint_motion")
    )
    result = {"category": category, "anchor_resolved": False, "critical_order_proxy": None}
    if not relation and "get_frame" not in program:
        return result
    if category == "endpoint_motion":
        token = program[program.index("get_frame") - 1]
        if token not in {"start", "end"}:
            return result
        frame = 0 if token == "start" else 127
        target_objects, _ = object_selector(program, object_starts[0], properties)
    else:
        anchor_start = object_starts[0] if relation else object_starts[1]
        objects, end = object_selector(program, anchor_start, properties)
        if len(objects) != 1 or program[end : end + 2] not in [
            ["unique", "filter_in"],
            ["unique", "filter_out"],
        ]:
            return result
        anchor_kind = program[end + 1]
        candidates = transitions(annotation, next(iter(objects)), anchor_kind)
        if len(candidates) != 1:
            result["unresolved_reason"] = "camera transition absent or repeated"
            return result
        frame = candidates[0]
        target_start = object_starts[1] if relation else object_starts[0]
        target_objects, target_end = object_selector(program, target_start, properties)
        if relation:
            kind = program[target_end]
            result["target_event_operator"] = kind
            if kind in {"filter_in", "filter_out"}:
                times = [t for o in target_objects for t in transitions(annotation, o, kind)]
            elif kind == "filter_collision" and target_objects == set(properties):
                times = [e["frame_id"] for e in annotation["collision"]]
            else:
                result["unresolved_reason"] = "unsupported target event syntax"
                return result
            witnesses = [
                t for t in times if (t < frame if relation == "filter_before" else t > frame)
            ]
            proxy_gold = "yes" if witnesses else "no"
            order = {
                "gold_agrees_with_transition_proxy": proxy_gold == gold,
                "target_events": len(times),
                "ties": sum(t == frame for t in times),
            }
            if times:
                # For existential yes, any far witness suffices; no requires the nearest boundary.
                relevant = (
                    max(witnesses, key=lambda t: abs(t - frame))
                    if witnesses
                    else min(times, key=lambda t: abs(t - frame))
                )
                other = frame_coverage(relevant, indices)
                order.update(
                    gap_frames=abs(relevant - frame),
                    same_nearest_sample=other["nearest_index"]
                    == frame_coverage(frame, indices)["nearest_index"],
                    both_events_exactly_sampled=relevant in indices and frame in indices,
                )
            result["critical_order_proxy"] = order
    coverage = frame_coverage(frame, indices)
    camera = {
        f["frame_id"]: {o["object_id"]: o["inside_camera_view"] for o in f["objects"]}
        for f in annotation["motion_trajectory"]
    }
    coverage["target_objects"] = len(target_objects)
    coverage["target_inside_anchor"] = sum(camera[frame][o] for o in target_objects)
    coverage["target_inside_nearest_sample"] = sum(
        camera[coverage["nearest_index"]][o] for o in target_objects
    )
    result.update(anchor_resolved=True, anchor_coverage=coverage)
    return result


def aggregate(rows):
    result = {}
    groups = {"all": rows}
    for category in ["event_relation", "event_conditioned_motion", "endpoint_motion"]:
        groups[category] = [r for r in rows if r["category"] == category]
    for label, predicate in {
        "anchor_exact_sample": lambda r: (
            r["anchor_resolved"] and r["anchor_coverage"]["exact_sample"]
        ),
        "anchor_between_samples": lambda r: (
            r["anchor_resolved"] and not r["anchor_coverage"]["exact_sample"]
        ),
        "no_target_inside_nearest": lambda r: (
            r["anchor_resolved"] and r["anchor_coverage"]["target_inside_nearest_sample"] == 0
        ),
        "order_same_sample_proxy": lambda r: (
            r["critical_order_proxy"]
            and r["critical_order_proxy"].get("same_nearest_sample", False)
        ),
    }.items():
        groups[label] = [r for r in rows if predicate(r)]
    for category in ["event_relation", "event_conditioned_motion"]:
        for exact in [True, False]:
            label = category + ("_anchor_exact" if exact else "_anchor_between")
            groups[label] = [
                r
                for r in rows
                if r["category"] == category
                and r["anchor_resolved"]
                and r["anchor_coverage"]["exact_sample"] == exact
            ]
    for name, predicate in {
        "order_no_target_event": lambda r: (
            r["critical_order_proxy"] is not None
            and r["critical_order_proxy"]["target_events"] == 0
        ),
        "order_gap_at_least16": lambda r: (
            (r["critical_order_proxy"] or {}).get("gap_frames", 0) >= 16
        ),
        "order_gap_below16": lambda r: (
            0 < (r["critical_order_proxy"] or {}).get("gap_frames", 0) < 16
        ),
        "targets_absent_by_attributes": lambda r: (
            r["anchor_resolved"] and r["anchor_coverage"]["target_objects"] == 0
        ),
        "resolved_targets_all_outside_nearest": lambda r: (
            r["anchor_resolved"]
            and r["anchor_coverage"]["target_objects"] > 0
            and r["anchor_coverage"]["target_inside_nearest_sample"] == 0
        ),
    }.items():
        groups[name] = [r for r in rows if predicate(r)]
    for name, selected in groups.items():
        result[name] = {
            "count": len(selected),
            "baseline_correct": sum(r["baseline_correct"] for r in selected),
            "candidate_correct": sum(r["candidate_correct"] for r in selected),
            "gold_yes": sum(r.get("gold") == "yes" for r in selected),
            "baseline_yes": sum(r.get("baseline_prediction") == "yes" for r in selected),
            "candidate_yes": sum(r.get("candidate_prediction") == "yes" for r in selected),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    started = time.monotonic()
    m = inspector()
    motion = a.root / "artifacts/s4-motion-coverage-pilot-001"
    candidate = a.root / "artifacts/s4-temporal-coverage-dev50-003/uniform"

    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    require(m.sha(motion / "preparation.json") == PREP_SHA, "fixed50preparation SHA")
    prepared = json.loads((motion / "preparation.json").read_bytes())
    source = json.loads((a.root / "reports/s3-dev-eval-video-stable-001.json").read_bytes())
    data = Path(source["command"][source["command"].index("--data") + 1])
    require(m.sha(data) == DATA_SHA, "public dev500 SHA")
    all_samples = [
        Sample.model_validate_json(x) for x in data.read_text().splitlines() if x.strip()
    ]
    require(
        len(all_samples) == 500 and all(s.meta.pool == "dev" for s in all_samples),
        "public dev only",
    )
    by_id = {s.id: s for s in all_samples}
    samples = [by_id[r["id"]] for r in prepared["selected"]]
    require(len(samples) == 50, "fixed50count")
    for sample, row in zip(samples, prepared["selected"], strict=True):
        require(canonical_sample_sha(sample) == row["sample_sha256"], "fixed sample SHA")
    records = []
    for path, expected in [
        (motion / "uniform/records.jsonl", RECORD_SHA),
        (candidate / "receipt.json", CANDIDATE_RECEIPT_SHA),
    ]:
        require(m.sha(path) == expected, "saved readout identity")
    receipt = json.loads((candidate / "receipt.json").read_bytes())
    require(receipt["status"] == "completed" and receipt["exit_code"] == 0, "candidate incomplete")
    require(
        m.sha(candidate / "records.jsonl") == receipt["records_sha256"], "candidate records SHA"
    )
    arms = [
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for path in [motion / "uniform/records.jsonl", candidate / "records.jsonl"]
    ]
    require(all(len(rows) == 50 for rows in arms), "readout count50")
    for path, expected in [
        (a.official / "questions/train.json", m.QUESTIONS_SHA),
        (a.official / "annotations/train/annotation_train.zip", m.ANNOTATIONS_SHA),
    ]:
        require(m.sha(path) == expected, "official source SHA")
    scenes = m.selected_scenes(
        a.official / "questions/train.json", {int(s.meta.family_id.split(":")[1]) for s in samples}
    )
    manifests = {}
    for video, binding in prepared["manifests"].items():
        require(m.sha(binding["path"]) == binding["sha256"], "frame manifest SHA")
        manifests[video] = json.loads(Path(binding["path"]).read_bytes())
    program_counts = collections.Counter()
    with zipfile.ZipFile(a.official / "annotations/train/annotation_train.zip") as zipped:
        for sample, left, right in zip(samples, *arms, strict=True):
            require(left["id"] == right["id"] == sample.id, "paired IDs/order")
            require(left["input_sha256"] == right["input_sha256"], "paired prepared input hash")
            require(
                sha256_json(left["input_summary"]) == left["input_summary_sha256"],
                "input summary SHA",
            )
            for row in [left, right]:
                require(
                    sha256_json(row["input_summary"]) == row["input_summary_sha256"],
                    "each input summary SHA",
                )
                require(
                    answer_tokens(sample.questions[0])[row["target_index"]]
                    == sample.questions[0].target.value,
                    "source gold",
                )
            index = int(sample.meta.family_id.split(":")[1])
            scene = scenes[index]
            question = next(
                q
                for q in scene["questions"]
                if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
            )
            require(
                source_hash_for_video(
                    {"scene_index": index}, question, sample.prefix.media[0].sha256
                )
                == sample.meta.source_sha256,
                "question/program/media binding",
            )
            require(
                question["answer"].casefold() == sample.questions[0].target.value, "official answer"
            )
            manifest = manifests[sample.prefix.media[0].sha256]
            indices = manifest["uniform_indices"]
            require(
                manifest["total_num_frames"] == 128
                and left["frame_indices"] == right["frame_indices"] == indices,
                "actual uniform indices",
            )
            require(indices == [round(i * 127 / 15) for i in range(16)], "uniform policy")
            names = [n for n in zipped.namelist() if n.endswith(f"/annotation_{index:05d}.json")]
            require(
                len(names) == 1 and zipped.getinfo(names[0]).file_size < 5_000_000,
                "bounded annotation",
            )
            annotation = json.loads(zipped.read(names[0]))
            require(annotation["scene_index"] == index, "annotation family")
            program = question["program"]
            program_counts.update(t for t in program if t.startswith("filter_") or t == "get_frame")
            observed = diagnose(program, annotation, indices, question["answer"].casefold())
            gold = sample.questions[0].target.value
            observed.update(
                baseline_correct=left["prediction"] == gold,
                candidate_correct=right["prediction"] == gold,
                gold=gold,
                baseline_prediction=left["prediction"],
                candidate_prediction=right["prediction"],
            )
            records.append(observed)
    anchors = [r["anchor_coverage"] for r in records if r["anchor_resolved"]]
    orders = [r["critical_order_proxy"] for r in records if r["critical_order_proxy"]]
    gaps = [r["gap_frames"] for r in orders if "gap_frames" in r]
    result = {
        "status": "completed",
        "exit_code": 0,
        "cost": None,
        "gpu_used": False,
        "groups": aggregate(records),
        "program_operator_counts": dict(program_counts),
        "video_timing": {
            "fps_values": sorted({v["fps"] for v in manifests.values()}),
            "duration_values": sorted({v["duration"] for v in manifests.values()}),
            "uniform16_indices": [round(i * 127 / 15) for i in range(16)],
            "adjacent_sample_gap_frames": [8, 9],
            "max_sample_gap_seconds": max(9 / v["fps"] for v in manifests.values()),
        },
        "anchor_resolved": len(anchors),
        "unresolved": len(records) - len(anchors),
        "anchor_distance_frames": {
            "max": max(r["nearest_distance"] for r in anchors),
            "mean": sum(r["nearest_distance"] for r in anchors) / len(anchors),
        },
        "relation_proxy": {
            "count": len(orders),
            "official_gold_agreement": sum(r["gold_agrees_with_transition_proxy"] for r in orders),
            "without_target_event": sum(r["target_events"] == 0 for r in orders),
            "with_same_frame_target": sum(r["ties"] > 0 for r in orders),
            "gap_frames": {"min": min(gaps), "max": max(gaps)},
            "gap_below_uniform_stride9": sum(g < 9 for g in gaps),
            "gap_below16": sum(g < 16 for g in gaps),
        },
        "boundaries": {
            "camera_transition_not_official_executor": True,
            "actual_occlusion_verified": False,
            "annotation_inside_not_pixel_visibility": True,
            "thresholds_fitted": False,
            "no_new_decoding_or_model_execution": True,
        },
        "bindings": {
            "preparation_sha256": PREP_SHA,
            "baseline_records_sha256": RECORD_SHA,
            "candidate_receipt_sha256": CANDIDATE_RECEIPT_SHA,
            "candidate_records_sha256": receipt["records_sha256"],
            "public_data_sha256": DATA_SHA,
            "questions_sha256": m.QUESTIONS_SHA,
            "annotations_sha256": m.ANNOTATIONS_SHA,
            "helper_sha256": m.sha(Path(__file__)),
        },
        "duration_seconds": time.monotonic() - started,
    }
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
