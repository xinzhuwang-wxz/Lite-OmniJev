"""Read-only bounded TRAIN feasibility; emits aggregates, never question rows or training data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import time
import zipfile
from pathlib import Path

POOL_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"
QUESTIONS_SHA = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
ANNOTATIONS_SHA = "e7f9185ad3dde8139b839b00f0f75483374dd0325a3cbec2dac882078497784c"
README_SHA = "a4838cc68b79ba049b54f95def1714cdbbd1380193d7d2a5d1a85440ac862e7b"
NAMESPACE = "42:s4-order-feasibility:"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_scenes(path, indices):
    """Structurally skip unselected official TRAIN scenes without JSON-decoding their content."""
    depth, quoted, escape, buf = 0, False, False, []
    result = {}
    with path.open() as stream:
        while chunk := stream.read(1024 * 1024):
            for char in chunk:
                if depth:
                    buf.append(char)
                if quoted:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        quoted = False
                    continue
                if char == '"':
                    quoted = True
                elif char == "{":
                    if not depth:
                        buf = [char]
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if not depth:
                        raw = "".join(buf)
                        match = re.search(r'"scene_index"\s*:\s*(\d+)', raw)
                        if match and int(match[1]) in indices:
                            value = json.loads(raw)
                            assert value["scene_index"] not in result
                            result[value["scene_index"]] = value
                        buf = []
    assert depth == 0 and not quoted and set(result) == indices
    return result


def inspect_annotation(annotation):
    properties = {
        o["object_id"]: (o["color"], o["material"], o["shape"])
        for o in annotation["object_property"]
    }
    assert len(properties) == len(annotation["object_property"])
    descriptor_count = collections.Counter(properties.values())
    trajectory = {
        f["frame_id"]: {o["object_id"]: o["inside_camera_view"] for o in f["objects"]}
        for f in annotation["motion_trajectory"]
    }
    assert set(trajectory) == set(range(128))
    events = collections.defaultdict(list)
    for event in annotation["collision"]:
        objects = event["object_ids"]
        frame = event["frame_id"]
        assert len(objects) == 2 and len(set(objects)) == 2
        assert type(frame) is int and 0 <= frame < 128
        assert all(o in properties for o in objects)
        events[tuple(sorted(objects))].append(frame)
    unique = {
        pair: frames[0]
        for pair, frames in events.items()
        if len(frames) == 1 and all(descriptor_count[properties[o]] == 1 for o in pair)
    }
    visible = {
        pair: frame
        for pair, frame in unique.items()
        if all(trajectory[frame].get(o) is True for o in pair)
    }
    pairs = list(itertools.combinations(visible, 2))
    nontied = [(a, b) for a, b in pairs if visible[a] != visible[b]]
    valid = [(a, b) for a, b in nontied if abs(visible[a] - visible[b]) >= 16]
    uniform = [round(i * 127 / 15) for i in range(16)]

    def nearest(frame):
        return min(uniform, key=lambda selected: (abs(selected - frame), selected))

    return {
        "same_nearest_uniform16_bucket_pairs": sum(
            nearest(visible[a]) == nearest(visible[b]) for a, b in valid
        ),
        "gap_below9frame_pairs": sum(abs(visible[a] - visible[b]) < 9 for a, b in valid),
        "collision_events": len(annotation["collision"]),
        "unique_collision_keys": len(unique),
        "visible_unique_collision_keys": len(visible),
        "repeated_collision_key": any(len(v) != 1 for v in events.values()),
        "ambiguous_descriptor": any(n != 1 for n in descriptor_count.values()),
        "fewer_than_two_collision_keys": len(events) < 2,
        "ties_rejected": len(pairs) - len(nontied),
        "gap_below16_rejected": len(nontied) - len(valid),
        "eligible_pairs": len(valid),
        "gaps": [abs(visible[a] - visible[b]) for a, b in valid],
        "both_orders_independently_evaluable": bool(valid)
        and all((visible[a] < visible[b]) != (visible[b] < visible[a]) for a, b in valid),
    }


def self_test():
    base = {
        "object_property": [
            {"object_id": i, "color": str(i), "material": "metal", "shape": "sphere"}
            for i in range(3)
        ],
        "motion_trajectory": [
            {
                "frame_id": f,
                "objects": [{"object_id": i, "inside_camera_view": True} for i in range(3)],
            }
            for f in range(128)
        ],
        "collision": [
            {"object_ids": [0, 1], "frame_id": 10},
            {"object_ids": [1, 2], "frame_id": 30},
        ],
    }
    assert inspect_annotation(base)["eligible_pairs"] == 1

    def modified():
        return json.loads(json.dumps(base))

    value = modified()
    value["collision"] = value["collision"][:1]
    assert inspect_annotation(value)["eligible_pairs"] == 0
    value = modified()
    value["collision"].append(value["collision"][0])
    assert inspect_annotation(value)["eligible_pairs"] == 0
    value = modified()
    value["collision"][1]["frame_id"] = 10
    assert inspect_annotation(value)["ties_rejected"] == 1
    value = modified()
    value["motion_trajectory"][10]["objects"][0]["inside_camera_view"] = False
    assert inspect_annotation(value)["eligible_pairs"] == 0
    value = modified()
    value["object_property"][1]["color"] = "0"
    assert inspect_annotation(value)["eligible_pairs"] == 0
    value = modified()
    value["collision"][1]["frame_id"] = 15
    assert inspect_annotation(value)["gap_below16_rejected"] == 1
    print("7 synthetic annotation rejection/ordering checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/s1-text-lora-001"))
    ap.add_argument(
        "--official",
        type=Path,
        default=Path("/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files"),
    )
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    assert args.output is not None and not args.output.exists()
    started = time.monotonic()
    pool = args.root / "artifacts/s3-pools-video-001/run-a/train.jsonl"
    questions = args.official / "questions/train.json"
    archive = args.official / "annotations/train/annotation_train.zip"
    for path, expected in [
        (pool, POOL_SHA),
        (questions, QUESTIONS_SHA),
        (archive, ANNOTATIONS_SHA),
        (args.official / "README.txt", README_SHA),
    ]:
        assert sha(path) == expected, "source SHA changed"
    rows = [json.loads(x) for x in pool.read_text().splitlines() if x.strip()]
    assert len(rows) == 8299 and all(r["meta"]["pool"] == "train" for r in rows)
    candidates = [
        s
        for s in rows
        if set(s["questions"][0]["instruction"].lower().split())
        & {"before", "after", "when", "while"}
    ]
    chosen, families = [], set()
    for target in ["yes", "no"]:
        for sample in sorted(
            candidates, key=lambda s: hashlib.sha256((NAMESPACE + s["id"]).encode()).hexdigest()
        ):
            if (
                sample["questions"][0]["target"]["value"] == target
                and sample["meta"]["family_id"] not in families
            ):
                chosen.append(sample)
                families.add(sample["meta"]["family_id"])
                if sum(s["questions"][0]["target"]["value"] == target for s in chosen) == 12:
                    break
    assert len(chosen) == len(families) == 24
    indices = {int(f.split(":")[1]) for f in families}
    scenes = selected_scenes(questions, indices)
    counts, program_counts, hashes, gaps = collections.Counter(), collections.Counter(), [], []
    from scripts.s0_media_dev import source_hash_for_video

    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        for sample in chosen:
            index = int(sample["meta"]["family_id"].split(":")[1])
            scene = scenes[index]
            qid = int(sample["id"].rsplit("-", 1)[-1])
            question = next(q for q in scene["questions"] if q["question_id"] == qid)
            assert question["question"] == sample["questions"][0]["instruction"]
            assert question["answer"].casefold() == sample["questions"][0]["target"]["value"]
            assert (
                source_hash_for_video(
                    {"scene_index": index}, question, sample["prefix"]["media"][0]["sha256"]
                )
                == sample["meta"]["source_sha256"]
            )
            program = question["program"]
            program_counts["before_after"] += bool(set(program) & {"filter_before", "filter_after"})
            program_counts["collision"] += "filter_collision" in program
            program_counts["in"] += "filter_in" in program
            program_counts["out"] += "filter_out" in program
            matched = [n for n in names if n.endswith(f"/annotation_{index:05d}.json")]
            assert len(matched) == 1
            info = zipped.getinfo(matched[0])
            assert info.file_size < 5_000_000 and not info.flag_bits & 1
            payload = zipped.read(info)
            annotation = json.loads(payload)
            assert annotation["scene_index"] == index
            assert annotation["video_filename"] == scene["video_filename"]
            observed = inspect_annotation(annotation)
            counts["families"] += 1
            counts["strict_eligible_families"] += bool(observed["eligible_pairs"])
            for key, value in observed.items():
                if key == "gaps":
                    gaps.extend(value)
                else:
                    counts[key] += value
            hashes.append(hashlib.sha256(payload).hexdigest())
    result = {
        "status": "feasibility_observed_not_training_data",
        "exit_code": 0,
        "source_license": "CC0-1.0",
        "source_admission": "existing ADR0005 full8299 TRAIN scope",
        "counts": dict(counts),
        "selected_source_program_counts": dict(program_counts),
        "source_sha256": {
            "train_pool": POOL_SHA,
            "official_train_questions": QUESTIONS_SHA,
            "train_annotations_zip": ANNOTATIONS_SHA,
            "official_readme": README_SHA,
        },
        "selection_namespace": NAMESPACE,
        "selected_family_count": 24,
        "selection_family_set_sha256": hashlib.sha256(
            json.dumps(sorted(families)).encode()
        ).hexdigest(),
        "selected_annotation_sha256_set_hash": hashlib.sha256(
            json.dumps(sorted(hashes)).encode()
        ).hexdigest(),
        "source_question_and_program_bindings_verified": 24,
        "gap_frames": {"min": min(gaps), "max": max(gaps), "mean": sum(gaps) / len(gaps)},
        "strict_rule": "two different collision object-pairs, each occurring exactly once; unique "
        "attribute descriptors; both participants in camera at their event; unequal "
        "integer event frames separated by at least16frames. Compute both orders separately.",
        "original_program_execution": "not implemented; no original before/after gold guessed",
        "source_only_scope": "only selected admitted TRAIN scenes decoded; other official source "
        "records structurally skipped; no private artifact opened",
        "sampling_observability": "not verified under 16-frame processor; no video decode or GPU",
        "training_data_created": False,
        "gpu_used": False,
        "cost": None,
        "duration_seconds": time.monotonic() - started,
        "helper_sha256": sha(Path(__file__)),
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
