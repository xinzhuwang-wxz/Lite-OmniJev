import binascii
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

from scripts import s0_gui_dev as gui_dev


def chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data))
    )


def png(width: int, height: int) -> bytes:
    rows = b"".join(b"\x00" + (b"\xff\xff\xff" * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(
    root: Path,
    *,
    episode_count: int = 8,
    bad_episode: str | None = None,
    tiny_episode: str | None = None,
) -> dict[str, Path]:
    media = root / "media"
    screenshots = media / "screenshots"
    annotations = root / "annotations"
    screenshots.mkdir(parents=True)
    annotations.mkdir()
    selected = []
    for index in range(episode_count):
        episode_id = f"{index + 1:016d}"
        selected.append(
            {
                "episode_id": episode_id,
                "episode_file": f"{episode_id}.json",
                "split": "app_split",
                "partition": "train",
                "screenshot_count": 2,
                "compressed_bytes": 1,
                "uncompressed_bytes": 1,
                "archive_member_count": 2,
            }
        )
        width = 60 if tiny_episode == episode_id else 200
        height = 60 if tiny_episode == episode_id else 200
        for step in range(2):
            (screenshots / f"{episode_id}_{step}.png").write_bytes(png(width, height))
        instruction = "" if bad_episode == episode_id else f"Tap target {episode_id}."
        annotation = {
            "episode_id": episode_id,
            "device_info": {"w": width, "h": height},
            "task_info": {"task": "fixture task"},
            "step_length": 2,
            "steps": [
                {
                    "step": 0,
                    "screenshot": f"{episode_id}_0.png",
                    "action": "CLICK",
                    "info": [[25, 25], [25, 25]],
                    "low_level_instruction": instruction,
                    "description": "excluded input field",
                    "intention": "excluded input field",
                    "sam2_bbox": [10, 10, 50, 50],
                },
                {
                    "step": 1,
                    "screenshot": f"{episode_id}_1.png",
                    "action": "TEXT",
                    "info": "abc",
                    "low_level_instruction": "Enter text.",
                    "sam2_bbox": [60, 60, 90, 90],
                },
            ],
        }
        (annotations / f"{episode_id}.json").write_text(json.dumps(annotation), encoding="utf-8")
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "split_name": "app_split",
                "partition": "train",
                "selected_episodes": selected,
            }
        ),
        encoding="utf-8",
    )
    return {"plan": plan, "media": media, "annotations": annotations}


def run_builder(
    root: Path, fx: dict[str, Path], *, dev_count: int = 3, limit: int = 6
) -> tuple[int, dict[str, Any]]:
    out = root / "out"
    code = gui_dev.main(
        [
            "--plan",
            str(fx["plan"]),
            "--annotation-root",
            str(fx["annotations"]),
            "--media-root",
            str(fx["media"]),
            "--dev-out",
            str(out / "dev.jsonl"),
            "--dev-ledger-out",
            str(out / "dev-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(out / "exclusion.jsonl"),
            "--media-manifest-out",
            str(out / "media.jsonl"),
            "--provenance-out",
            str(out / "provenance.jsonl"),
            "--receipts-out",
            str(out / "receipts.json"),
            "--report",
            str(out / "report.json"),
            "--dev-count",
            str(dev_count),
            "--max-annotation-downloads",
            str(limit),
        ]
    )
    return code, json.loads((out / "report.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def install_fixture_decoder(monkeypatch: Any) -> None:
    def decode(payload: bytes) -> tuple[int, int, str]:
        width, height = tmp_payload_dimensions(payload)
        return width, height, hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(gui_dev, "decode_png_rgb", decode)


def tmp_payload_dimensions(payload: bytes) -> tuple[int, int]:
    path = Path("/tmp/nonexistent")
    del path
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise gui_dev.GuiDevError("PNG signature required")
    return struct.unpack_from(">II", payload, 16)


def test_builds_gui_choice4_dev_records_without_leaking_gold_in_inputs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    install_fixture_decoder(monkeypatch)
    fx = write_fixture(tmp_path)
    code, report = run_builder(tmp_path, fx)
    assert code == 0
    assert report["status"] == "passed"
    assert report["gpu_used"] is False
    assert report["holdout_accessed"] is False
    assert report["training_admitted"] is False
    assert report["primitive_scope"]["choice4_target_localization_only"] is True
    assert report["split"]["dev_items"] == 3
    assert sum(report["split"]["target_token_counts"].values()) == 3
    assert len(report["split"]["target_token_counts"]) >= 2

    out = tmp_path / "out"
    dev = read_jsonl(out / "dev.jsonl")
    ledger = read_jsonl(out / "dev-ledger.jsonl")
    media = read_jsonl(out / "media.jsonl")
    provenance = read_jsonl(out / "provenance.jsonl")
    exclusion = read_jsonl(out / "exclusion.jsonl")
    assert len(dev) == len(ledger) == len(media) == len(provenance) == 3
    assert all(row["meta"]["modality"] == ["image"] for row in dev)
    assert all(row["questions"][0]["type"] == "choice" for row in dev)
    assert all(len(row["questions"][0]["options"]) == 4 for row in dev)
    assert all(row["modality"] == "screenshot" for row in ledger)
    assert {row["label_origin"] for row in ledger} == {"program_truth"}
    assert all(row["width"] == 200 and row["height"] == 200 for row in media)
    assert all("decoded_rgb_sha256" in row for row in media)
    assert "gold_bbox" not in json.dumps(dev)
    assert "click_point" not in json.dumps(dev)
    assert "excluded input field" not in json.dumps(dev)
    selected_families = {row["meta"]["family_id"] for row in dev}
    excluded_families = {row["family_id"] for row in exclusion if row["pool"] == "dev_excluded"}
    assert selected_families == excluded_families
    for row in provenance:
        boxes = [tuple(box) for box in row["option_bboxes"]]
        assert tuple(row["gold_bbox"]) in boxes
        assert row["target_token"] in {"A", "B", "C", "D"}

    second = tmp_path / "second"
    second.mkdir()
    code2, report2 = run_builder(second, fx)
    assert code2 == 0
    assert report2["outputs"] == report["outputs"]
    for name in [
        "dev.jsonl",
        "dev-ledger.jsonl",
        "exclusion.jsonl",
        "media.jsonl",
        "provenance.jsonl",
        "receipts.json",
    ]:
        assert (second / "out" / name).read_bytes() == (out / name).read_bytes()


def test_filters_bad_instruction_and_impossible_negatives_before_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    install_fixture_decoder(monkeypatch)
    ordered = [f"{index + 1:016d}" for index in range(8)]
    by_hash = sorted(ordered, key=lambda item: (gui_dev.allocation_key(item), item))
    fx = write_fixture(tmp_path, bad_episode=by_hash[0], tiny_episode=by_hash[1])
    code, report = run_builder(tmp_path, fx, dev_count=3, limit=6)
    assert code == 0
    excluded = report["split"]["excluded_counts"]
    assert excluded["empty_low_level_instruction"] == 1
    assert excluded["insufficient_negative_boxes"] == 1


def test_reports_blocker_when_not_enough_eligible_gui_items(
    tmp_path: Path, monkeypatch: Any
) -> None:
    install_fixture_decoder(monkeypatch)
    ordered = [f"{index + 1:016d}" for index in range(5)]
    by_hash = sorted(ordered, key=lambda item: (gui_dev.allocation_key(item), item))
    fx = write_fixture(tmp_path, episode_count=5, bad_episode=by_hash[0], tiny_episode=by_hash[1])
    code, report = run_builder(tmp_path, fx, dev_count=3, limit=4)
    assert code == 1
    assert report["status"] == "blocked"
    assert report["blocker"].startswith("eligible GUI dev items")


def test_negative_box_generation_stops_at_three_for_tiny_boxes() -> None:
    boxes = gui_dev.candidate_negative_boxes((50, 50, 51, 51), (50, 50), width=200, height=200)
    assert boxes == [(0, 0, 1, 1), (1, 0, 2, 1), (2, 0, 3, 1)]
