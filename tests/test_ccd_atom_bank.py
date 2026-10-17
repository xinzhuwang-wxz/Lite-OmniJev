"""Synthetic two-object CLEVRER annotation; frame extraction is stubbed (no ffmpeg)."""

import json
from pathlib import Path

from data.schema import Sample
from scripts.ccd_atom_bank import main

ZIP_SHA = "c" * 64


def annotation(index: int) -> dict[str, object]:
    # Object 0 (red metal cube) is always in view at (3, 3); object 1 (blue rubber
    # sphere) sits at (-3, -3) and leaves the view at the last frame.
    steps = [
        {
            "frame_id": f,
            "objects": [
                {"object_id": 0, "location": [3.0, 3.0, 0.2], "inside_camera_view": True},
                {"object_id": 1, "location": [-3.0, -3.0, 0.2], "inside_camera_view": f < 127},
            ],
        }
        for f in range(128)
    ]
    return {
        "scene_index": index,
        "video_filename": f"video_{index:05d}.mp4",
        "object_property": [
            {"object_id": 0, "color": "red", "material": "metal", "shape": "cube"},
            {"object_id": 1, "color": "blue", "material": "rubber", "shape": "sphere"},
        ],
        "motion_trajectory": steps,
        "collision": [],
    }


def build(tmp_path: Path, *extra: str) -> tuple[list[Sample], list[dict[str, object]], dict]:
    annotations = tmp_path / "annotations" / "annotation_00000-01000"
    annotations.mkdir(parents=True, exist_ok=True)
    (tmp_path / "media" / "video").mkdir(parents=True, exist_ok=True)
    for index in (7, 8):
        name = f"annotation_{index:05d}.json"
        (annotations / name).write_text(json.dumps(annotation(index)))
        (tmp_path / "media" / "video" / f"video_{index:05d}.mp4").write_bytes(b"mp4")
    families = tmp_path / "families.txt"
    families.write_text("7 train\n8 dev\n")
    extracted: list[tuple[str, int]] = []

    def stub(video: Path, frame: int, out: Path) -> None:
        extracted.append((video.name, frame))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f"png-{video.name}-{frame}".encode())

    output = tmp_path / f"atoms{len(extra)}.jsonl"
    argv = [
        "--annotations",
        str(tmp_path / "annotations"),
        "--media-root",
        str(tmp_path / "media"),
        "--frames-out",
        str(tmp_path / "frames"),
        "--families",
        str(families),
        "--pool",
        "train",
        "--annotation-zip-sha",
        ZIP_SHA,
        "--output",
        str(output),
        "--frames",
        "2",
        "--bins",
        "2",
        *extra,
    ]
    assert main(argv, extract=stub) == 0
    assert extracted == [("video_00007.mp4", 0), ("video_00007.mp4", 127)]
    samples = [Sample.model_validate_json(line) for line in output.read_text().splitlines()]
    atoms = [json.loads(line) for line in Path(f"{output}.atoms.jsonl").read_text().splitlines()]
    receipt = json.loads(Path(f"{output}.receipt.json").read_text())
    return samples, atoms, receipt


def by_text(samples: list[Sample], frame: int, prefix: str) -> dict[str, object]:
    found = {}
    for s in samples:
        q = s.questions[0]
        if s.id.startswith(f"atom-00007-f{frame:03d}-") and q.instruction.startswith(prefix):
            assert q.target is not None and q.target.kind == "hard"
            found[q.instruction.split("\n")[0]] = q.target.value
    return found


def test_atoms_follow_annotation_truth_and_contract(tmp_path: Path) -> None:
    samples, atoms, receipt = build(tmp_path)

    assert len({s.id for s in samples}) == len(samples) == len(atoms)
    assert [a["id"] for a in atoms] == [s.id for s in samples]
    for s in samples:
        assert s.meta.family_id == "clevrer:7" and s.meta.pool == "train"
        assert s.meta.source_sha256 == ZIP_SHA and s.meta.modality == ["image"]
        assert s.prefix.media[0].type == "image"
    assert {a["video"] for a in atoms} == {7} and {a["frame"] for a in atoms} == {0, 127}

    presence0 = by_text(samples, 0, "Is there a")
    presence127 = by_text(samples, 127, "Is there a")
    assert presence0["Is there a red metal cube in this image?"] == "yes"
    assert presence0["Is there a blue rubber sphere in this image?"] == "yes"
    assert presence127["Is there a red metal cube in this image?"] == "yes"
    assert presence127["Is there a blue rubber sphere in this image?"] == "no"
    perturbed = [
        a for a in atoms if a["kind"] == "presence" and a["object_id"] is None and a["frame"] == 0
    ]
    assert perturbed and all(a["answer_raw"] in {"yes", "no"} for a in perturbed)
    full_negatives = [
        text
        for text, value in presence0.items()
        if len(text.split()) == 9
        and text
        not in {
            "Is there a red metal cube in this image?",
            "Is there a blue rubber sphere in this image?",
        }
    ]
    assert len(full_negatives) == 2
    assert all(presence0[text] == "no" for text in full_negatives)
    if "Is there a red object in this image?" in presence0:
        assert presence0["Is there a red object in this image?"] == "yes"

    assert by_text(samples, 0, "How many objects")["How many objects are in this image?"] == "C"
    assert by_text(samples, 127, "How many objects")["How many objects are in this image?"] == "B"

    assert receipt["bin_edges"] == {"column": [3.0], "row": [3.0]}
    columns0 = by_text(samples, 0, "Which column")
    assert columns0["Which column of the floor is the red metal cube in?"] == "B"
    assert columns0["Which column of the floor is the blue rubber sphere in?"] == "A"
    rows127 = by_text(samples, 127, "Which row")
    assert rows127 == {"Which row of the floor is the red metal cube in?": "B"}
    raw = {(a["kind"], a["frame"], a["object_id"]): a["answer_raw"] for a in atoms}
    assert raw[("column", 0, 1)] == "column 1 (leftmost)"
    assert raw[("row", 0, 0)] == "row 2 (nearest)"

    assert receipt["frames"] == [0, 127] and receipt["videos"] == [7]
    counted = receipt["counts_by_kind_and_answer"]
    assert sum(counted["presence"].values()) == len([a for a in atoms if a["kind"] == "presence"])


def test_receipt_edges_are_reused(tmp_path: Path) -> None:
    edges = tmp_path / "old.receipt.json"
    edges.write_text(json.dumps({"bin_edges": {"column": [-5.0, 0.0], "row": [5.0, 6.0]}}))
    samples, _, receipt = build(tmp_path, "--edges-from", str(edges))

    assert receipt["bin_edges"] == {"column": [-5.0, 0.0], "row": [5.0, 6.0]}
    columns = by_text(samples, 0, "Which column")
    assert columns["Which column of the floor is the red metal cube in?"] == "C"
    assert by_text(samples, 0, "Which row")["Which row of the floor is the red metal cube in?"] == (
        "A"
    )
