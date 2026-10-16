"""Oracle scenes: annotation-truth scenes in the caption schema (validation split).

Runs on the GPU box CPU. Produces oracle-scenes.jsonl for the dev150 families
using the same JSON shape the executor consumes, so the identical executor
measures the pipeline's upper bound. Motion is still absent (schema parity
with the VLM arm keeps the comparison honest).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location(
    "b1grid", "/root/autodl-tmp/s1-text-lora-001/artifacts/s4-ensemble-oracle-001/build-b1-consistency-grid.py"
)
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

PHASES = ("start", "early", "quarter", "middle", "three_quarters", "late", "very_late", "end")


def phase_of(frame: int, total: int) -> str:
    frac = frame / max(1, total - 1)
    if frac <= 0.04:
        return "start"
    if frac <= 0.18:
        return "early"
    if frac <= 0.36:
        return "quarter"
    if frac <= 0.52:
        return "middle"
    if frac <= 0.68:
        return "three_quarters"
    if frac <= 0.84:
        return "late"
    if frac < 0.96:
        return "very_late"
    return "end"


def scene_from_truth(raw: dict) -> dict:
    parsed = b1.scene_truth(raw)
    objects, truth = parsed["objects"], parsed["truth"]
    final = parsed["final_frame"]
    inside_flags = {}
    for frame in raw["motion_trajectory"]:
        present = {i["object_id"]: bool(i["inside_camera_view"]) for i in frame["objects"]}
        for oid in objects:
            inside_flags.setdefault(oid, []).append(present.get(oid, False))
    exits = {}
    for oid, flags in inside_flags.items():
        last_exit = None
        for i in range(1, len(flags)):
            if flags[i - 1] and not flags[i]:
                last_exit = i
        exits[oid] = last_exit
    entries = []
    names = {}
    for oid, spec_ in objects.items():
        t = truth.get(oid)
        if not t:
            continue
        name = f"{spec_['color']} {spec_['material']} {spec_['shape']}"
        names[oid] = name
        speeds = []
        for frame in raw["motion_trajectory"]:
            for item in frame["objects"]:
                if item["object_id"] == oid:
                    vx, vz = item["velocity"][0], item["velocity"][2]
                    speeds.append((vx * vx + vz * vz) ** 0.5)
        moving = bool(speeds and (max(speeds) > 0.1 or sum(speeds) / len(speeds) > 0.05))
        entries.append(
            {
                "name": name,
                "first_seen": (
                    "start"
                    if t["entry_frame"] is None
                    else phase_of(t["entry_frame"], final + 1)
                ),
                "last_seen": (
                    phase_of(exits[oid], final + 1)
                    if exits.get(oid) is not None
                    else "end"
                ),
                "moving": moving,
            }
        )
    order = sorted(
        (oid for oid in names if truth[oid]["entry_frame"] is not None),
        key=lambda oid: truth[oid]["entry_frame"],
    )
    edges = [f"{names[a]} before {names[b]}" for a, b in zip(order, order[1:])]
    collisions = [
        {
            "between": [names[i] for i in hit["object_ids"]],
            "when": phase_of(hit["frame_id"], final + 1),
        }
        for hit in parsed["collisions"]
        if all(i in names for i in hit["object_ids"])
    ]
    return {
        "objects": entries,
        "entry_order": edges,
        "collisions": collisions,
        "present_at_start": [
            names[oid] for oid in names if truth[oid]["present_at_begin"]
        ],
        "present_at_end": [names[oid] for oid in names if truth[oid]["present_at_end"]],
    }


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: build_oracle_scenes.py QUESTIONS VAL_ANNOTATION_ZIP OUTPUT")
    questions_path, annotation_path, out_path = (Path(a) for a in sys.argv[1:])
    families = {
        json.loads(line)["family"]
        for line in questions_path.read_text().splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(annotation_path) as archive:
        names = set(archive.namelist())
        written = 0
        with out_path.open("w", encoding="utf-8") as stream:
            for line in questions_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                scene_number = int(row["family"].split(":")[1])
                raw = json.loads(archive.read(b1.annotation_member(scene_number, names)))
                stream.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "summary_valid": True,
                            "summary": scene_from_truth(raw),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
    print(json.dumps({"scenes": written, "families": len(families)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
