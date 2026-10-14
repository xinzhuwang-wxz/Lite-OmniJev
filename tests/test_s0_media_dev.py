import base64
import hashlib
import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from scripts import s0_media_dev as media_dev


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(item) for item in value), set())
    return set()


def jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    # Valid 1x1 JPEG; the fixture color only keeps callers deterministic without Pillow.
    _ = color
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
        "////////////////////////////////////////////2wBDAf////////////////////////////////////////////////////////////"
        "//////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAA"
        "AAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAEP/aAA"
        "gBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAEP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFB"
        "AAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8hf//aAAwDAQACAAMAAAAQ/wD/xAAUEQEAAAAAAAAAAAAAAAAAAAAQ/9oACAEDAQE/EH//"
        "xAAUEQEAAAAAAAAAAAAAAAAAAAAQ/9oACAECAQE/EH//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EH//2Q=="
    )


def write_aok_fixture(tmp_path: Path, *, bad_license: bool = False) -> dict[str, Any]:
    root = tmp_path / "aok"
    image_root = root / "coco-ccby-publicdomain-images"
    (image_root / "train").mkdir(parents=True)
    train_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index in range(6):
        qid = f"qid-{index}"
        image_id = 1000 + index
        file_name = f"{image_id:012d}.jpg"
        payload = jpeg_bytes((index * 20, 30, 40))
        image_path = image_root / "train" / file_name
        image_path.write_bytes(payload)
        train_rows.append(
            {
                "split": "train",
                "question_id": qid,
                "image_id": image_id,
                "question": f"What object is visible {index}?",
                "choices": ["cat", "dog", "bus", "cup"],
                "correct_choice_idx": index % 4,
            }
        )
        manifest_rows.append(
            {
                "split": "train",
                "coco_split": "train",
                "question_id": qid,
                "image_id": image_id,
                "file_name": file_name,
                "license_id": 7 if bad_license and index == 0 else 4,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "coco_url": f"http://images.cocodataset.org/train2017/{file_name}",
                "flickr_url": f"https://example.test/{file_name}",
            }
        )
    train_rows.append(
        {
            "split": "train",
            "question_id": "unlicensed-qid",
            "image_id": 9999,
            "question": "This family remains unavailable.",
            "choices": ["a", "b", "c", "d"],
            "correct_choice_idx": 0,
        }
    )
    tar_path = root / "aokvqa_v1p0.tar.gz"
    encoded = json.dumps(train_rows).encode()
    with tarfile.open(tar_path, "w:gz") as archive:
        info = tarfile.TarInfo(media_dev.AOK_TRAIN_MEMBER)
        info.size = len(encoded)
        archive.addfile(info, BytesIO(encoded))
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8")
    return {
        "tar": tar_path,
        "manifest": manifest,
        "image_root": image_root,
        "tar_sha": sha(tar_path),
        "manifest_sha": sha(manifest),
    }


def run_image(
    tmp_path: Path, fx: dict[str, Any], *, dev_count: int = 3
) -> tuple[int, dict[str, Any]]:
    code = media_dev.main(
        [
            "--mode",
            "image",
            "--aok-tar",
            str(fx["tar"]),
            "--aok-manifest",
            str(fx["manifest"]),
            "--image-root",
            str(fx["image_root"]),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "media.jsonl"),
            "--media-out-dir",
            str(tmp_path / "media-out"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-aok-tar-sha256",
            fx["tar_sha"],
            "--expected-aok-manifest-sha256",
            fx["manifest_sha"],
            "--dev-count",
            str(dev_count),
        ]
    )
    return code, json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def write_clevrer_fixture(tmp_path: Path) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    for scene_index in range(10_000):
        answer = "yes" if scene_index % 2 == 0 else "no"
        scenes.append(
            {
                "scene_index": scene_index,
                "questions": [
                    {
                        "question_id": 5,
                        "question_type": "descriptive",
                        "question_subtype": "exist",
                        "question": f"Are there objects in scene {scene_index}?",
                        "program": ["objects", "exist"],
                        "answer": answer,
                    },
                    {
                        "question_id": 6,
                        "question_type": "descriptive",
                        "question_subtype": "query_color",
                        "question": "What color?",
                        "program": ["query_color"],
                        "answer": "blue",
                    },
                ],
            }
        )
    questions = tmp_path / "train.json"
    questions.write_text(json.dumps(scenes), encoding="utf-8")
    video_zip = tmp_path / "video_train.zip"
    with zipfile.ZipFile(video_zip, "w") as archive:
        for scene_index in range(10_000):
            archive.writestr(
                media_dev.clevrer_video_member(scene_index), f"video-{scene_index}".encode()
            )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": "videos/train/video_train.zip",
                        "local_sha256": media_dev.CLEVRER_VIDEO_ZIP_SHA256,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "questions": questions,
        "questions_sha": sha(questions),
        "video_zip": video_zip,
        "receipt": receipt,
    }


def run_video(
    tmp_path: Path, fx: dict[str, Any], *, dev_count: int = 3
) -> tuple[int, dict[str, Any]]:
    code = media_dev.main(
        [
            "--mode",
            "video",
            "--clevrer-questions",
            str(fx["questions"]),
            "--video-zip",
            str(fx["video_zip"]),
            "--clevrer-receipt",
            str(fx["receipt"]),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "media.jsonl"),
            "--media-out-dir",
            str(tmp_path / "media-out"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-clevrer-questions-sha256",
            fx["questions_sha"],
            "--dev-count",
            str(dev_count),
        ]
    )
    return code, json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def test_builds_image_dev_with_redacted_ledgers_and_full_train_family_exclusion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        media_dev,
        "decode_image",
        lambda path: {
            "width": 4,
            "height": 3,
            "format": "JPEG",
            "rgb_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        },
    )
    fx = write_aok_fixture(tmp_path)
    code, report = run_image(tmp_path, fx)
    assert code == 0
    assert report["status"] == "passed"
    assert report["training_admitted"] is False
    assert report["holdout_accessed"] is False
    assert report["primitive_scope"] == {
        "image_choice_only": True,
        "video_noul_only": False,
        "score_covered": False,
        "complete_distribution_or_quality": False,
    }
    assert report["split"]["dev_items"] == 3
    assert report["split"]["candidate_families"] == 6
    assert report["split"]["all_train_families"] == 7

    dev = read_jsonl(tmp_path / "dev.jsonl")
    ledger = read_jsonl(tmp_path / "dev-ledger.jsonl")
    exclusion = read_jsonl(tmp_path / "exclusion.jsonl")
    media = read_jsonl(tmp_path / "media.jsonl")
    assert len(dev) == len(ledger) == len(media) == 3
    assert all(row["meta"]["modality"] == ["image"] for row in dev)
    assert all(row["questions"][0]["type"] == "choice" for row in dev)
    assert all(row["questions"][0]["options"] == ["cat", "dog", "bus", "cup"] for row in dev)
    assert all(
        row["questions"][0]["target"]["source"] == "aokvqa_correct_choice_idx" for row in dev
    )
    assert {row["pool"] for row in exclusion} == {
        "dev_excluded",
        "train_available",
        "train_unavailable",
    }
    assert "unlicensed-qid" not in json.dumps({"ledger": ledger, "exclusion": exclusion})
    assert all_keys({"ledger": ledger, "exclusion": exclusion}).isdisjoint(
        {"question", "target", "options", "media", "uri"}
    )
    assert all("attribution" in row and "flickr_url" in row for row in media)

    second = tmp_path / "second"
    second.mkdir()
    code2, report2 = run_image(second, fx)
    assert code2 == 0
    assert report2["outputs"] == report["outputs"]
    for name in ["dev.jsonl", "dev-ledger.jsonl", "exclusion.jsonl", "media.jsonl"]:
        assert (second / name).read_bytes() == (tmp_path / name).read_bytes()


def test_image_rejects_manifest_license_mismatch(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(media_dev, "decode_image", lambda path: {})
    fx = write_aok_fixture(tmp_path, bad_license=True)
    code, report = run_image(tmp_path, fx, dev_count=2)
    assert code == 1
    assert report["status"] == "blocked"
    assert report["blocker"] == "AOKVQA allowed manifest contains non-CC-BY license"
    assert not (tmp_path / "dev.jsonl").exists()


def test_refuses_to_overwrite_existing_report(tmp_path: Path) -> None:
    fx = write_aok_fixture(tmp_path)
    report = tmp_path / "report.json"
    report.write_text("keep\n", encoding="utf-8")
    code = media_dev.main(
        [
            "--mode",
            "image",
            "--aok-tar",
            str(fx["tar"]),
            "--aok-manifest",
            str(fx["manifest"]),
            "--image-root",
            str(fx["image_root"]),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "media.jsonl"),
            "--media-out-dir",
            str(tmp_path / "media-out"),
            "--report",
            str(report),
            "--expected-aok-tar-sha256",
            fx["tar_sha"],
            "--expected-aok-manifest-sha256",
            fx["manifest_sha"],
            "--dev-count",
            "2",
        ]
    )
    assert code == 1
    assert report.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / "dev.jsonl").exists()


def test_builds_video_dev_extracts_selected_members_and_marks_all_train_scenes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        media_dev,
        "decode_video",
        lambda path: {
            "duration_seconds": 3.0,
            "fps": 24.0,
            "total_num_frames": 72,
            "sampled_frame_indices": list(range(16)),
            "decoded_frames": 16,
            "decoded_width": 64,
            "decoded_height": 64,
            "decoded_frame_rgb_sha256": [digest(f"{path}-{i}") for i in range(16)],
        },
    )
    fx = write_clevrer_fixture(tmp_path)
    code, report = run_video(tmp_path, fx)
    assert code == 0
    assert report["status"] == "passed"
    assert (
        report["source"]["video_zip_sha256_reused_from_receipt"]
        == media_dev.CLEVRER_VIDEO_ZIP_SHA256
    )
    assert report["source"]["video_zip_full_hash_recomputed"] is False
    assert report["split"]["dev_items"] == 3
    assert report["split"]["candidate_families"] == 10_000
    assert report["split"]["all_train_families"] == 10_000
    assert report["split"]["fixed_video_frames"] == 16

    dev = read_jsonl(tmp_path / "dev.jsonl")
    ledger = read_jsonl(tmp_path / "dev-ledger.jsonl")
    exclusion = read_jsonl(tmp_path / "exclusion.jsonl")
    media = read_jsonl(tmp_path / "media.jsonl")
    assert len(dev) == len(ledger) == len(media) == 3
    assert len(exclusion) == 10_000
    assert all(row["meta"]["modality"] == ["video"] for row in dev)
    assert all(row["questions"][0]["type"] == "noul" for row in dev)
    assert all(
        row["questions"][0]["target"]["source"] == "clevrer_descriptive_exist_answer" for row in dev
    )
    assert all(
        (tmp_path / "media-out" / row["uri"]).read_bytes().startswith(b"video-") for row in media
    )
    assert all_keys({"ledger": ledger, "exclusion": exclusion}).isdisjoint(
        {"question", "target", "answer", "media", "uri"}
    )


def test_video_rejects_receipt_mismatch_before_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(media_dev, "decode_video", lambda path: {})
    fx = write_clevrer_fixture(tmp_path)
    fx["receipt"].write_text(
        json.dumps(
            {"files": [{"relative_path": "videos/train/video_train.zip", "local_sha256": "0" * 64}]}
        ),
        encoding="utf-8",
    )
    code, report = run_video(tmp_path, fx, dev_count=2)
    assert code == 1
    assert report["blocker"] == "CLEVRER receipt video zip sha256 mismatch"
    assert not (tmp_path / "dev.jsonl").exists()
