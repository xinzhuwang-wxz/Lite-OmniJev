"""II-5 video SAS: program-facet auxiliary rows for TRAIN videos (structured supervision).

Reuses the B1 annotation parsing (scene truth: objects, entry/exit frames,
collisions) and extends it with count/color/motion-direction/speed facets.
Every gold is program-decidable from the official annotation truth; no
undecidable-U rows (ADR0024). Runs on the GPU box CPU; outputs the facet bank,
a deterministic pilot slice, the mains rows for the same videos, and ledgers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "b1grid", Path(__file__).with_name("build-b1-consistency-grid.py")
)
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

TARGET_MAINS = 400
FACETS_PER_MAIN = 1  # arm2 facet fraction ~0.5
W03_FACET_FRACTION = 0.3
COUNT_OPTIONS = [str(i) for i in range(0, 9)]


def describe(obj: dict) -> str:
    return f"{obj['color']} {obj['material']} {obj['shape']}"


def mean_velocity(scene: dict, oid: int) -> tuple[float, float]:
    xs, zs = [], []
    for frame in scene["motion_trajectory"]:
        for item in frame["objects"]:
            if item["object_id"] == oid:
                xs.append(item["velocity"][0])
                zs.append(item["velocity"][2])
    return (sum(xs) / max(1, len(xs)), sum(zs) / max(1, len(zs)))


def facet_rows(scene: int, pool_row: dict, parsed: dict, out_prefix: str):
    objects, truth = parsed["objects"], parsed["truth"]
    live = {oid: spec_ for oid, spec_ in objects.items() if truth[oid] is not None}
    final = parsed["final_frame"]
    rows = []

    def add(cell: str, instruction: str, *, options=None, qtype="noul", gold="yes"):
        rows.append(
            {
                "id": f"{out_prefix}:{cell}:{len(rows):03d}",
                "prefix": {
                    "state": (
                        "Answer the yes/no question using the attached video."
                        if qtype == "noul"
                        else "Answer the multiple-choice question using the attached video."
                    ),
                    "media": pool_row["prefix"]["media"],
                },
                "questions": [
                    {
                        "qid": qtype,
                        "type": qtype,
                        "instruction": instruction,
                        "options": options,
                        "score_levels": None,
                        "score_descriptions": None,
                        "target": {
                            "kind": "hard",
                            "value": gold,
                            "source": f"ii5v_facet_{cell}",
                        },
                        "pair_of": None,
                    }
                ],
                "meta": {
                    **pool_row["meta"],
                },
            }
        )

    begin = [oid for oid in live if truth[oid]["present_at_begin"]]
    end = [oid for oid in live if truth[oid]["present_at_end"]]
    # count facet (choice over 0..8; the answer token is the letter)
    if len(begin) <= 8:
        add(
            "count_begin",
            "How many objects are in the scene when the video begins?",
            options=COUNT_OPTIONS,
            qtype="choice",
            gold=chr(65 + len(begin)),
        )
    # color existence facets (capped)
    colors_present = {objects[oid]["color"] for oid in begin}
    for color in sorted({s["color"] for s in live.values()})[:2]:
        add(
            "color_begin",
            f"Is there a {color} object in the scene when the video begins?",
            gold="yes" if color in colors_present else "no",
        )
    # relation (enter/exit order) facets
    entering = sorted(
        oid for oid in live if truth[oid]["entry_frame"] is not None
    )
    for index, left in enumerate(entering[:2]):
        for right in entering[index + 1 : index + 3]:
            if truth[left]["entry_frame"] == truth[right]["entry_frame"]:
                continue
            before = truth[left]["entry_frame"] < truth[right]["entry_frame"]
            add(
                "enter_order",
                f"Does the {describe(objects[left])} enter the scene before the "
                f"{describe(objects[right])} enters the scene?",
                gold="yes" if before else "no",
            )
    # start-end facets
    for oid in entering[:2]:
        add(
            "start_end_entry",
            f"Does the {describe(objects[oid])} enter the scene after the video begins?",
            gold="yes",
        )
    leaving = sorted(
        oid
        for oid in live
        if not truth[oid]["present_at_end"] and truth[oid]["last_inside"] < final
    )
    for oid in leaving[:2]:
        add(
            "start_end_exit",
            f"Does the {describe(objects[oid])} leave the scene before the video ends?",
            gold="yes",
        )
    for oid in list(live)[:2]:
        add(
            "start_end_stay",
            f"Is the {describe(objects[oid])} in the scene when the video ends?",
            gold="yes" if truth[oid]["present_at_end"] else "no",
        )
    # collision existence facets (one true when available, one negative)
    collisions = parsed.get("collisions", [])
    if collisions:
        hit = collisions[0]
        left, right = hit["object_ids"]
        add(
            "collision_exist",
            f"Do the {describe(objects[left])} and the {describe(objects[right])} "
            "collide during the video?",
            gold="yes",
        )
    non_colliding = None
    if collisions and len(live) >= 3:
        collided = {tuple(sorted(h["object_ids"])) for h in collisions}
        ids = sorted(live)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if (a, b) not in collided:
                    non_colliding = (a, b)
                    break
            if non_colliding:
                break
    if non_colliding:
        a, b = non_colliding
        add(
            "collision_absent",
            f"Do the {describe(objects[a])} and the {describe(objects[b])} "
            "collide during the video?",
            gold="no",
        )
    # motion direction + speed facets
    for oid in list(live)[:3]:
        vx, _ = mean_velocity(parsed["raw_scene"], oid)
        if abs(vx) < 1e-3:
            continue
        add(
            "motion_direction",
            f"Does the {describe(objects[oid])} move to the left during the video?",
            gold="yes" if vx < 0 else "no",
        )
    speeds = {}
    for oid in list(live)[:4]:
        vx, vz = mean_velocity(parsed["raw_scene"], oid)
        speeds[oid] = (vx * vx + vz * vz) ** 0.5
    ranked = sorted(speeds, key=lambda o: speeds[o])
    if len(ranked) >= 2 and speeds[ranked[-1]] > speeds[ranked[0]] * 1.2:
        add(
            "motion_speed",
            f"Does the {describe(objects[ranked[-1]])} move faster than the "
            f"{describe(objects[ranked[0]])}?",
            gold="yes",
        )
    return rows


def main() -> int:
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "usage: build_ii5_video_sas.py ANNOTATION_ZIP TRAIN_JSONL OUT_DIR [FULL_FACET_TARGET]"
        )
    full_target = int(sys.argv[4]) if len(sys.argv) == 5 else None
    annotations_path, train_path, out_dir = (Path(a) for a in sys.argv[1:4])
    if b1.sha256_file(annotations_path) != b1.ANNOTATIONS_SHA:
        raise SystemExit("annotation sha mismatch")
    if b1.sha256_file(train_path) != b1.TRAIN_SHA:
        raise SystemExit("train pool sha mismatch")
    pool_rows = [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    video_rows = {
        int(r["meta"]["family_id"].split(":")[1]): r
        for r in pool_rows
        if r["meta"]["modality"] == ["video"]
    }
    import hashlib

    ranked_families = sorted(
        video_rows, key=lambda scene: hashlib.sha256(f"ii5v:{scene}".encode()).hexdigest()
    )
    mains, bank = [], []
    with zipfile.ZipFile(annotations_path) as archive:
        names = set(archive.namelist())
        for scene in ranked_families:
            if len(mains) >= (TARGET_MAINS if full_target is None else 10**9):
                break
            if full_target is not None and len(bank) >= full_target:
                break
            raw = json.loads(archive.read(b1.annotation_member(scene, names)))
            parsed = b1.scene_truth(raw)
            parsed["raw_scene"] = raw
            row = json.loads(json.dumps(video_rows[scene]))
            # unify data_version so single-version trainer validation passes
            row["meta"]["data_version"] = (
                row["meta"]["data_version"] + "+ii5v-v1"
            )
            mains.append(row)
            video_facets = facet_rows(
                scene, row, parsed, out_prefix=f"ii5v-facet-{scene}"
            )
            if full_target is not None:
                bank.extend(video_facets)
            elif video_facets:
                pick = sorted(
                    video_facets,
                    key=lambda r: hashlib.sha256(("ii5v:pick:" + r["id"]).encode()).hexdigest(),
                )[0]
                bank.append(pick)
    facets = bank
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        (out_dir / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        (out_dir / name.replace(".jsonl", "-ledger.jsonl")).write_text(
            "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in rows),
            encoding="utf-8",
        )
        import hashlib as h

        return h.sha256((out_dir / name).read_bytes()).hexdigest()

    if full_target is not None:
        if len(facets) < full_target:
            raise SystemExit(f"full facet target unreachable: {len(facets)} < {full_target}")
        shas = {
            "facets": dump("ii5v-full-facets.jsonl", facets),
            "mains": dump("ii5v-full-mains.jsonl", mains),
            "mixed": dump("ii5v-full-mixed.jsonl", [*mains, *facets]),
        }
        print(
            json.dumps(
                {
                    "mode": "full",
                    "families": len(mains),
                    "facets": len(facets),
                    "mixed_rows": len(mains) + len(facets),
                    "sha": shas,
                },
                indent=2,
            )
        )
        return 0
    mixed = mains + facets
    arm3_facets = [
        r
        for r in facets
        if int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 10 < 3
    ]
    arm3 = [*mains, *arm3_facets]

    summary = {
        "mains": len(mains),
        "facets_selected_one_per_video": len(facets),
        "mixed_rows": len(mixed),
        "arm3_rows": len(arm3),
        "arm3_facet_fraction": round(len(arm3_facets) / max(1, len(arm3)), 3),
        "facets_by_cell": {},
        "sha": {
            "mains": dump("ii5v-mains.jsonl", mains),
            "mixed": dump("ii5v-mixed.jsonl", mixed),
            "arm3": dump("ii5v-arm3.jsonl", arm3),
        },
    }
    for row in facets:
        cell = row["id"].split(":")[1]
        summary["facets_by_cell"][cell] = summary["facets_by_cell"].get(cell, 0) + 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
