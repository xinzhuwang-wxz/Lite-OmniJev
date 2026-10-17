"""Build per-frame CLEVRER perception atoms (presence / count / column / row) on CPU.

Each atom is one product Sample with one Question about one extracted video frame, with
ground truth read from the official CLEVRER annotation (never from a model). Presence is
noul; count and position are lettered choice. Position bins split world x (column) and
world y (row) at quantile edges over every in-view location of the selected videos; the
edges go to the receipt and can be reused with --edges-from so eval bins match training.
Atom kinds and raw answers go to OUT.atoms.jsonl because the sample contract forbids extras.

Usage: --annotations DIR --media-root DIR --frames-out DIR --families FILE --pool train
       --annotation-zip-sha SHA --output OUT.jsonl [--frames 16] [--bins 6]
       [--neg-per-frame 2] [--max-videos N] [--seed S] [--edges-from RECEIPT]
--annotations is searched recursively for annotation_*.json; --families has one
"<index> <pool>" per line and only indices whose pool equals --pool are used, so the
clevrer:<index> family never crosses pools.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from data.schema import Sample

COLORS = ("gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow")
MATERIALS = ("metal", "rubber")
SHAPES = ("sphere", "cube", "cylinder")
PLURAL = {"sphere": "spheres", "cube": "cubes", "cylinder": "cylinders"}
COUNT_OPTIONS = tuple(str(i) for i in range(7))
STATE = "Answer the question about the attached image."
TARGET_SOURCE = "clevrer_annotation"
DATA_VERSION = "ccd-atoms-v1"
# Column/row orientation in the image is assumed from CLEVRER's fixed camera; answers are
# defined by world x/y bins, so the labels are names for consistent bins.
AXES = {"column": (0, "column", "leftmost", "rightmost"), "row": (1, "row", "nearest", "farthest")}

Extractor = Callable[[Path, int, Path], None]
Description = tuple[str, str, str]


def ffmpeg_extract(video: Path, frame: int, out: Path) -> None:
    """Write decoded frame `frame` (0-based) of `video` as a PNG."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no frame {frame} from {video}")


def default_frames(n: int) -> list[int]:
    if n < 2:
        raise ValueError("--frames must be at least 2")
    return sorted({round(i * 127 / (n - 1)) for i in range(n)})


def bin_edges(values: list[float], bins: int) -> list[float]:
    """K-1 inner cut points at the j/K quantiles; bin = bisect_right(edges, value)."""
    if bins < 2 or len(values) < 2:
        raise ValueError("position bins need --bins >= 2 and at least two locations")
    return [round(edge, 6) for edge in statistics.quantiles(values, n=bins, method="inclusive")]


def position_options(axis: str, bins: int) -> list[str]:
    _, word, first, last = AXES[axis]
    labels = [f"{word} {i + 1}" for i in range(bins)]
    labels[0] += f" ({first})"
    labels[-1] += f" ({last})"
    return labels


def phrase(desc: Description) -> str:
    return " ".join(desc)


def _question(
    kind: str, text: str, options: list[str] | None, raw: str
) -> tuple[dict[str, Any], str]:
    if options is None:
        instruction, value = text, raw
    else:
        lettered = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
        instruction, value = f"{text}\nOptions:\n{lettered}", chr(65 + options.index(raw))
    question = {
        "qid": kind,
        "type": "noul" if options is None else "choice",
        "instruction": instruction,
        "options": options,
        "score_levels": None,
        "score_descriptions": None,
        "target": {"kind": "hard", "source": TARGET_SOURCE, "value": value},
        "pair_of": None,
    }
    return question, raw


def frame_atoms(
    annotation: dict[str, Any],
    frame: int,
    edges: dict[str, list[float]],
    neg_per_frame: int,
    seed: int,
) -> list[tuple[str, dict[str, Any], int | None, str]]:
    """Return (kind, question, object_id, raw answer) atoms for one frame of one video."""
    index = annotation["scene_index"]
    rng = random.Random(f"{seed}:{index}:{frame}")
    objects = {
        o["object_id"]: (o["color"], o["material"], o["shape"])
        for o in annotation["object_property"]
    }
    step = next(s for s in annotation["motion_trajectory"] if s["frame_id"] == frame)
    visible = {o["object_id"]: o for o in step["objects"] if o["inside_camera_view"]}
    seen = [objects[i] for i in sorted(visible)]
    atoms: list[tuple[str, dict[str, Any], int | None, str]] = []

    def presence(text: str, answer: bool, object_id: int | None) -> None:
        question, raw = _question("presence", text, None, "yes" if answer else "no")
        atoms.append(("presence", question, object_id, raw))

    asked: set[Description] = set()
    for object_id in sorted(objects):
        desc = objects[object_id]
        if desc in asked:
            continue
        asked.add(desc)
        hit = desc in seen
        presence(f"Is there a {phrase(desc)} in this image?", hit, object_id)
    perturbed: set[Description] = set()
    for c, m, s in objects.values():
        perturbed.update((alt, m, s) for alt in COLORS)
        perturbed.update((c, alt, s) for alt in MATERIALS)
        perturbed.update((c, m, alt) for alt in SHAPES)
    absent = sorted(perturbed - set(objects.values()))
    for desc in rng.sample(absent, min(neg_per_frame, len(absent))):
        presence(f"Is there a {phrase(desc)} in this image?", False, None)
    color = rng.choice(COLORS)
    presence(f"Is there a {color} object in this image?", any(d[0] == color for d in seen), None)
    material, shape = rng.choice(MATERIALS), rng.choice(SHAPES)
    presence(
        f"Is there a {material} {shape} in this image?",
        any(d[1:] == (material, shape) for d in seen),
        None,
    )

    count_rows = [
        ("How many objects are in this image?", len(seen)),
        (
            f"How many {(c := rng.choice(COLORS))} objects are in this image?",
            sum(d[0] == c for d in seen),
        ),
        (
            f"How many {(m := rng.choice(MATERIALS))} objects are in this image?",
            sum(d[1] == m for d in seen),
        ),
        (
            f"How many {PLURAL[(s := rng.choice(SHAPES))]} are in this image?",
            sum(d[2] == s for d in seen),
        ),
    ]
    for text, count in count_rows:
        if count < len(COUNT_OPTIONS):
            question, raw = _question("count", text, list(COUNT_OPTIONS), str(count))
            atoms.append(("count", question, None, raw))

    descriptions = list(objects.values())
    for object_id in sorted(visible):
        desc = objects[object_id]
        if descriptions.count(desc) != 1:
            continue
        location = visible[object_id]["location"]
        for axis, (coord, word, _, _) in AXES.items():
            options = position_options(axis, len(edges[axis]) + 1)
            answer = options[bisect.bisect_right(edges[axis], location[coord])]
            text = f"Which {word} of the floor is the {phrase(desc)} in?"
            question, raw = _question(axis, text, options, answer)
            atoms.append((axis, question, object_id, raw))
    return atoms


def load_annotations(root: Path) -> dict[int, tuple[dict[str, Any], str]]:
    found: dict[int, tuple[dict[str, Any], str]] = {}
    for path in sorted(root.rglob("annotation_*.json")):
        raw = path.read_bytes()
        annotation = json.loads(raw)
        found[int(annotation["scene_index"])] = (annotation, hashlib.sha256(raw).hexdigest())
    return found


def main(argv: list[str] | None = None, extract: Extractor = ffmpeg_extract) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--frames-out", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--pool", choices=("train", "dev"), required=True)
    parser.add_argument("--annotation-zip-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--bins", type=int, default=6)
    parser.add_argument("--neg-per-frame", type=int, default=2)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--edges-from", type=Path, default=None)
    args = parser.parse_args(argv)

    frames = default_frames(args.frames)
    annotations = load_annotations(args.annotations)
    pooled = sorted(
        int(parts[0])
        for parts in (line.split() for line in args.families.read_text().splitlines())
        if len(parts) >= 2 and parts[1] == args.pool
    )
    eligible = [index for index in pooled if index in annotations]
    random.Random(args.seed).shuffle(eligible)
    chosen = sorted(eligible[: args.max_videos] if args.max_videos is not None else eligible)
    if not chosen:
        raise SystemExit("no annotated videos in the requested pool")

    if args.edges_from is not None:
        edges = json.loads(args.edges_from.read_text())["bin_edges"]
    else:
        located = [
            o["location"]
            for index in chosen
            for step in annotations[index][0]["motion_trajectory"]
            for o in step["objects"]
            if o["inside_camera_view"]
        ]
        edges = {
            axis: bin_edges([loc[coord] for loc in located], args.bins)
            for axis, (coord, *_) in AXES.items()
        }

    counts: dict[str, dict[str, int]] = {}
    skipped: list[int] = []
    sidecar = args.output.with_name(args.output.name + ".atoms.jsonl")
    ids: set[str] = set()
    with (
        args.output.open("x", encoding="utf-8") as stream,
        sidecar.open("x", encoding="utf-8") as atoms_out,
    ):
        for index in chosen:
            video = args.media_root / "video" / f"video_{index:05d}.mp4"
            if not video.is_file():
                skipped.append(index)
                continue
            annotation = annotations[index][0]
            for frame in frames:
                uri = f"video_{index:05d}/f{frame:03d}.png"
                png = args.frames_out / uri
                extract(video, frame, png)
                png_sha = hashlib.sha256(png.read_bytes()).hexdigest()
                serial: dict[str, int] = {}
                for kind, question, object_id, raw in frame_atoms(
                    annotation, frame, edges, args.neg_per_frame, args.seed
                ):
                    n = serial[kind] = serial.get(kind, -1) + 1
                    sample = Sample.model_validate(
                        {
                            "id": f"atom-{index:05d}-f{frame:03d}-{kind}-{n}",
                            "prefix": {
                                "state": STATE,
                                "media": [{"type": "image", "uri": uri, "sha256": png_sha}],
                            },
                            "questions": [question],
                            "meta": {
                                "modality": ["image"],
                                "source_dataset": "CLEVRER",
                                "source_revision": args.annotation_zip_sha,
                                "source_sha256": args.annotation_zip_sha,
                                "license": "CC0-1.0",
                                "usage": "commercial",
                                "pool": args.pool,
                                "family_id": f"clevrer:{index}",
                                "data_version": DATA_VERSION,
                            },
                        }
                    )
                    if sample.id in ids:
                        raise ValueError(f"duplicate atom id {sample.id}")
                    ids.add(sample.id)
                    stream.write(
                        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n"
                    )
                    atoms_out.write(
                        json.dumps(
                            {
                                "id": sample.id,
                                "kind": kind,
                                "video": index,
                                "frame": frame,
                                "object_id": object_id,
                                "answer_raw": raw,
                            }
                        )
                        + "\n"
                    )
                    by_answer = counts.setdefault(kind, {})
                    by_answer[raw] = by_answer.get(raw, 0) + 1

    receipt = {
        "data_version": DATA_VERSION,
        "pool": args.pool,
        "seed": args.seed,
        "frames": frames,
        "bins": len(edges["column"]) + 1,
        "bin_edges": edges,
        "bin_edges_from": str(args.edges_from) if args.edges_from else None,
        "videos": [i for i in chosen if i not in skipped],
        "skipped_missing_media": skipped,
        "samples": len(ids),
        "counts_by_kind_and_answer": counts,
        "inputs": {
            "annotation_zip_sha256": args.annotation_zip_sha,
            "annotations": str(args.annotations),
            "annotation_file_sha256": {str(i): annotations[i][1] for i in chosen},
            "families": str(args.families),
            "families_sha256": hashlib.sha256(args.families.read_bytes()).hexdigest(),
            "media_root": str(args.media_root),
            "frames_root": str(args.frames_out),
            "neg_per_frame": args.neg_per_frame,
            "max_videos": args.max_videos,
        },
        "outputs": {"samples": str(args.output), "atoms": str(sidecar)},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    receipt_path = args.output.with_name(args.output.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"samples": len(ids), "videos": len(receipt["videos"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
