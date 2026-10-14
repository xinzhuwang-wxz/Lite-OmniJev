import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import s0_crossmodal_dev as crossmodal


def digest(value: bytes | str) -> str:
    payload = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_passed_report(path: Path, dev_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "exit_code": 0,
                "outputs": {"dev_jsonl_sha256": crossmodal.sha256_file(dev_path)},
            }
        ),
        encoding="utf-8",
    )


def sample(
    *,
    sample_id: str,
    modality: str,
    family: str,
    media_uri: str,
    media_sha: str,
    question_type: str,
    answer: str,
    pool: str = "dev",
) -> dict[str, Any]:
    question: dict[str, Any]
    if question_type == "choice":
        question = {
            "qid": "choice",
            "type": "choice",
            "instruction": f"What is shown in {sample_id}?",
            "options": ["cat", "dog", "bus", "cup"],
            "target": {"kind": "hard", "value": answer, "source": "fixture"},
        }
    else:
        question = {
            "qid": "noul",
            "type": "noul",
            "instruction": f"Is sound {sample_id} present?",
            "target": {"kind": "hard", "value": answer, "source": "fixture"},
        }
    return {
        "id": sample_id,
        "prefix": {
            "state": f"Use the attached {modality}.",
            "media": [{"type": modality, "uri": media_uri, "sha256": media_sha}],
        },
        "questions": [question],
        "meta": {
            "modality": [modality],
            "source_dataset": f"{modality}-fixture",
            "source_revision": "fixture-rev",
            "source_sha256": digest(f"{sample_id}:source"),
            "license": "CC0-1.0",
            "usage": "commercial",
            "pool": pool,
            "family_id": family,
            "data_version": f"{modality}-fixture-v1",
        },
    }


def write_fixture(tmp_path: Path, *, image_pool: str = "dev") -> dict[str, Path]:
    image_root = tmp_path / "image-root"
    audio_root = tmp_path / "audio-root"
    (image_root / "image").mkdir(parents=True)
    audio_root.mkdir(parents=True)
    image_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    for index in range(8):
        image_payload = f"image-{index}".encode()
        audio_payload = f"audio-{index}".encode()
        image_name = f"img-{index}.jpg"
        audio_name = f"aud-{index}.wav"
        (image_root / "image" / image_name).write_bytes(image_payload)
        (audio_root / audio_name).write_bytes(audio_payload)
        image_rows.append(
            sample(
                sample_id=f"image-{index}",
                modality="image",
                family=f"image-family-{index}",
                media_uri=f"image/{image_name}",
                media_sha=digest(image_payload),
                question_type="choice",
                answer=["A", "B", "C", "D"][index % 4],
                pool=image_pool if index == 0 else "dev",
            )
        )
        audio_rows.append(
            sample(
                sample_id=f"audio-{index}",
                modality="audio",
                family=f"audio-family-{index}",
                media_uri=audio_name,
                media_sha=digest(audio_payload),
                question_type="noul",
                answer="yes" if index % 2 == 0 else "no",
            )
        )
    image_dev = tmp_path / "image.jsonl"
    audio_dev = tmp_path / "audio.jsonl"
    image_dev.write_text("\n".join(json.dumps(row) for row in image_rows) + "\n", encoding="utf-8")
    audio_dev.write_text("\n".join(json.dumps(row) for row in audio_rows) + "\n", encoding="utf-8")
    image_report = tmp_path / "image-report.json"
    audio_report = tmp_path / "audio-report.json"
    write_passed_report(image_report, image_dev)
    write_passed_report(audio_report, audio_dev)
    return {
        "image_root": image_root,
        "audio_root": audio_root,
        "image_dev": image_dev,
        "audio_dev": audio_dev,
        "image_report": image_report,
        "audio_report": audio_report,
    }


def run_builder(
    tmp_path: Path, fx: dict[str, Path], *, dev_count: int = 4
) -> tuple[int, dict[str, Any]]:
    code = crossmodal.main(
        [
            "--image-dev",
            str(fx["image_dev"]),
            "--audio-dev",
            str(fx["audio_dev"]),
            "--image-media-root",
            str(fx["image_root"]),
            "--audio-media-root",
            str(fx["audio_root"]),
            "--image-source-report",
            str(fx["image_report"]),
            "--audio-source-report",
            str(fx["audio_report"]),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--component-ledger-out",
            str(tmp_path / "component-ledger.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "media.jsonl"),
            "--media-out-dir",
            str(tmp_path / "media-out"),
            "--report",
            str(tmp_path / "report.json"),
            "--dev-count",
            str(dev_count),
        ]
    )
    return code, json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def test_builds_crossmodal_dev_with_and_truth_redaction_and_determinism(tmp_path: Path) -> None:
    fx = write_fixture(tmp_path)
    code, report = run_builder(tmp_path, fx)
    assert code == 0
    assert report["status"] == "passed"
    assert report["training_admitted"] is False
    assert report["holdout_accessed"] is False
    assert report["not_native_crossmodal_annotation"] is True
    assert report["split"]["yes"] == 2
    assert report["split"]["no"] == 2
    assert report["split"]["proposal_rules"] == {
        "both_true": 2,
        "image_false_audio_true": 1,
        "image_true_audio_false": 1,
    }

    dev = read_jsonl(tmp_path / "dev.jsonl")
    ledger = read_jsonl(tmp_path / "dev-ledger.jsonl")
    components = read_jsonl(tmp_path / "component-ledger.jsonl")
    media = read_jsonl(tmp_path / "media.jsonl")
    assert len(dev) == len(ledger) == len(components) == len(media) == 4
    assert all(row["meta"]["modality"] == ["image", "audio"] for row in dev)
    assert all(row["questions"][0]["type"] == "noul" for row in dev)
    assert all(len(row["prefix"]["media"]) == 2 for row in dev)
    assert all((tmp_path / "media-out" / row["image_uri"]).is_file() for row in media)
    assert all((tmp_path / "media-out" / row["audio_uri"]).is_file() for row in media)
    assert {row["label_origin"] for row in ledger} == {"program_truth"}
    assert "target" not in json.dumps(ledger)
    assert "gold" not in json.dumps(dev).casefold()
    assert {row["program_truth"] for row in components if row["proposal_rule"] == "both_true"} == {
        "yes"
    }
    assert {row["program_truth"] for row in components if row["proposal_rule"] != "both_true"} == {
        "no"
    }
    assert all(row["component_family_separate_ledger"] is True for row in components)

    second = tmp_path / "second"
    second.mkdir()
    code2, report2 = run_builder(second, fx)
    assert code2 == 0
    assert report2["outputs"] == report["outputs"]
    for name in ["dev.jsonl", "dev-ledger.jsonl", "component-ledger.jsonl", "media.jsonl"]:
        assert (second / name).read_bytes() == (tmp_path / name).read_bytes()


def test_rejects_duplicate_family_bad_media_hash_and_different_pool(tmp_path: Path) -> None:
    fx = write_fixture(tmp_path / "dup")
    rows = read_jsonl(fx["image_dev"])
    rows[1]["meta"]["family_id"] = rows[0]["meta"]["family_id"]
    fx["image_dev"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    write_passed_report(fx["image_report"], fx["image_dev"])
    code, report = run_builder(tmp_path / "dup", fx)
    assert code == 1
    assert report["blocker"] == "duplicate image family"

    fx = write_fixture(tmp_path / "bad-sha")
    rows = read_jsonl(fx["audio_dev"])
    rows[0]["prefix"]["media"][0]["sha256"] = "0" * 64
    fx["audio_dev"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    write_passed_report(fx["audio_report"], fx["audio_dev"])
    code, report = run_builder(tmp_path / "bad-sha", fx)
    assert code == 1
    assert report["blocker"] == "audio media sha256 mismatch"

    fx = write_fixture(tmp_path / "bad-pool", image_pool="train")
    code, report = run_builder(tmp_path / "bad-pool", fx)
    assert code == 1
    assert report["blocker"] == "image sample is not dev pool"


def test_rejects_failed_source_report_and_output_overwrite(tmp_path: Path) -> None:
    fx = write_fixture(tmp_path / "failed-report")
    fx["image_report"].write_text(
        json.dumps(
            {
                "status": "blocked",
                "exit_code": 1,
                "outputs": {"dev_jsonl_sha256": crossmodal.sha256_file(fx["image_dev"])},
            }
        ),
        encoding="utf-8",
    )
    code, report = run_builder(tmp_path / "failed-report", fx)
    assert code == 1
    assert report["blocker"] == "image source report did not pass"

    fx = write_fixture(tmp_path / "mismatch-report")
    fx["audio_report"].write_text(
        json.dumps({"status": "passed", "exit_code": 0, "outputs": {"dev_jsonl_sha256": "0" * 64}}),
        encoding="utf-8",
    )
    code, report = run_builder(tmp_path / "mismatch-report", fx)
    assert code == 1
    assert report["blocker"] == "audio source report dev hash mismatch"

    fx = write_fixture(tmp_path / "overwrite")
    report_path = tmp_path / "overwrite" / "report.json"
    report_path.write_text("keep\n", encoding="utf-8")
    code = crossmodal.main(
        [
            "--image-dev",
            str(fx["image_dev"]),
            "--audio-dev",
            str(fx["audio_dev"]),
            "--image-media-root",
            str(fx["image_root"]),
            "--audio-media-root",
            str(fx["audio_root"]),
            "--dev-out",
            str(tmp_path / "overwrite" / "dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "overwrite" / "dev-ledger.jsonl"),
            "--component-ledger-out",
            str(tmp_path / "overwrite" / "component-ledger.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "overwrite" / "media.jsonl"),
            "--media-out-dir",
            str(tmp_path / "overwrite" / "media-out"),
            "--report",
            str(report_path),
            "--dev-count",
            "4",
        ]
    )
    assert code == 1
    assert report_path.read_text(encoding="utf-8") == "keep\n"
