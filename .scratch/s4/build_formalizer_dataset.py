"""Formalizer dataset: SFT pairs + offline-decidable reward set from cached truth.

Inputs (already on the GPU box): the 600-item TRAIN slice and the official
train annotation zip. For every video we emit:
- the truth scene JSON (8-phase schema, same as the oracle scenes) — the SFT
  target for "video -> scene JSON";
- a set of program-decidable probe questions with golds derived from the same
  truth (executor coverage/correctness is computable offline, no new labels);
- the truth scene itself for per-predicate F1 reward terms.
Split: first 500 videos train, last 100 probe (deterministic, same ranking).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.getcwd())
B1_PATH = "/root/autodl-tmp/s1-text-lora-001/artifacts/s4-ensemble-oracle-001/build-b1-consistency-grid.py"
spec = importlib.util.spec_from_file_location("b1grid", B1_PATH)
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)
ORACLE_PATH = "/root/autodl-tmp/s1-text-lora-001/artifacts/s4-ensemble-oracle-001/build_oracle_scenes.py"
spec2 = importlib.util.spec_from_file_location("oracle_builder", ORACLE_PATH)
oracle_builder = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(oracle_builder)

PLURAL = {"cube": "cubes", "cylinder": "cylinders", "sphere": "spheres"}
TRAIN_COUNT = 500
PROBE_COUNT = 100


def probe_questions(scene_number: int, raw: dict) -> list[dict]:
    """Program-decidable questions with golds from the annotation truth."""
    parsed = b1.scene_truth(raw)
    objects, truth = parsed["objects"], parsed["truth"]
    live = {oid: spec_ for oid, spec_ in objects.items() if truth.get(oid)}
    if not live:
        return []

    def describe(oid):
        s = objects[oid]
        return f"{s['color']} {s['material']} {s['shape']}"

    questions = []
    entering = sorted(
        (oid for oid in live if truth[oid]["entry_frame"] is not None),
        key=lambda o: truth[o]["entry_frame"],
    )

    def cls(oid):
        s = objects[oid]
        return f"{s['color']} {PLURAL[s['shape']]}"

    for i, left in enumerate(entering[:3]):
        for right in entering[i + 1 : i + 4]:
            before = truth[left]["entry_frame"] < truth[right]["entry_frame"]
            questions.append(
                {
                    "instruction": f"Are there any {cls(left)} that enter the scene before the {describe(right)} enters the scene?",
                    "gold": "yes" if before else "no",
                    "predicate": "enterers",
                }
            )
    leaving = [
        oid
        for oid in live
        if not truth[oid]["present_at_end"] and truth[oid]["last_inside"] < parsed["final_frame"]
    ]
    for oid in leaving[:2]:
        questions.append(
            {
                "instruction": f"Are there any {cls(oid)} that exit the scene?",
                "gold": "yes",
                "predicate": "exiters",
            }
        )
    for oid in sorted(live)[:2]:
        questions.append(
            {
                "instruction": f"Are there any {cls(oid)} when the video ends?",
                "gold": "yes" if truth[oid]["present_at_end"] else "no",
                "predicate": "exiters",
            }
        )
    speeds = {}
    for frame in raw["motion_trajectory"]:
        for item in frame["objects"]:
            if item["object_id"] in live:
                vx, vz = item["velocity"][0], item["velocity"][2]
                speeds.setdefault(item["object_id"], []).append((vx * vx + vz * vz) ** 0.5)
    for oid in sorted(speeds)[:2]:
        vals = speeds[oid]
        moving = bool(vals and (max(vals) > 0.1 or sum(vals) / len(vals) > 0.05))
        questions.append(
            {
                "instruction": f"Are there any {'moving' if moving else 'stationary'} {cls(oid).replace(PLURAL[objects[oid]['shape']], PLURAL[objects[oid]['shape']])}?",
                "gold": "yes" if moving else "no",
                "predicate": "moving",
            }
        )
    colors = {objects[oid]["color"] for oid in live}
    first = sorted(live)[0]
    c = objects[first]["color"]
    questions.append(
        {
            "instruction": f"Are there any {c} objects when the video begins?",
            "gold": "yes" if c in colors else "no",
            "predicate": "colors",
        }
    )
    return questions


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build_formalizer_dataset.py SLICE_600 ANNOTATION_ZIP OUT_DIR"
        )
    slice_path, annotation_path, out_dir = (Path(a) for a in sys.argv[1:])
    import hashlib

    rows = [json.loads(l) for l in slice_path.read_text().splitlines() if l.strip()]
    if len(rows) != 600:
        raise SystemExit(f"expected 600 rows, got {len(rows)}")
    records = []
    with zipfile.ZipFile(annotation_path) as archive:
        names = set(archive.namelist())
        for row in rows:
            scene_number = int(row["meta"]["family_id"].split(":")[1])
            raw = json.loads(archive.read(b1.annotation_member(scene_number, names)))
            truth_scene = oracle_builder.scene_from_truth(raw)
            questions = probe_questions(scene_number, raw)
            records.append(
                {
                    "video_id": row["id"],
                    "scene_number": scene_number,
                    "media": row["prefix"]["media"][0],
                    "truth_scene": truth_scene,
                    "probe_questions": questions,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    train, probe = records[:TRAIN_COUNT], records[TRAIN_COUNT:]
    summary = {"train": 0, "probe": 0, "questions": 0, "by_predicate": {}}
    for name, subset in (("train", train), ("probe", probe)):
        path = out_dir / f"formalizer-{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset),
            encoding="utf-8",
        )
        summary[name] = len(subset)
        summary["questions"] += sum(len(r["probe_questions"]) for r in subset)
        for record in subset:
            for question in record["probe_questions"]:
                key = question["predicate"]
                summary["by_predicate"][key] = summary["by_predicate"].get(key, 0) + 1
    summary["sha"] = {
        name: hashlib.sha256((out_dir / f"formalizer-{name}.jsonl").read_bytes()).hexdigest()
        for name in ("train", "probe")
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
