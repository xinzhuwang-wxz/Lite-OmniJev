import csv
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

from scripts import s0_audio_dev as audio_dev


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: Path, *, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 8000
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x00\x00" * frames)


def write_metadata(path: Path, names: list[str], *, bad_license: bool = False) -> str:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "",
                "file_name",
                "keywords",
                "sound_link",
                "start_end_samples",
                "manufacturer",
                "license",
            ],
        )
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "": str(index),
                    "file_name": name,
                    "keywords": "fixture",
                    "sound_link": f"https://freesound.org/people/user/sounds/{1000 + index}",
                    "start_end_samples": "",
                    "manufacturer": f"creator-{index}",
                    "license": "http://creativecommons.org/licenses/by-nc/3.0/"
                    if bad_license and index == 0
                    else "http://creativecommons.org/publicdomain/zero/1.0/",
                }
            )
    return sha(path)


def write_train(path: Path, names: list[str], *, six_row_name: str | None = None) -> str:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["file_name", "QuestionText", "answer", "confidence"]
        )
        writer.writeheader()
        for index, name in enumerate(names):
            repeats = 6 if name == six_row_name else 3
            for _ in range(repeats):
                writer.writerow(
                    {
                        "file_name": name,
                        "QuestionText": f"Is fixture sound {index} present?",
                        "answer": "yes" if index % 2 == 0 else "no",
                        "confidence": "yes",
                    }
                )
    return sha(path)


def write_reports(tmp_path: Path, names: list[str], metadata_sha: str) -> tuple[Path, Path]:
    license_plan = tmp_path / "license-plan.json"
    audio_report = tmp_path / "audio-report.json"
    approved = [
        {
            "file_name": name,
            "license": "CC0-1.0",
            "license_url": "http://creativecommons.org/publicdomain/zero/1.0/",
            "original_url": f"https://freesound.org/people/user/sounds/{1000 + index}",
            "attribution": f"creator-{index}",
        }
        for index, name in enumerate(names)
    ]
    license_plan.write_text(
        json.dumps({"metadata_sha256": metadata_sha, "approved_audio": approved}),
        encoding="utf-8",
    )
    audio_report.write_text(
        json.dumps(
            {
                "status": "approved_audio_extracted_source_audit_only",
                "metadata_sha256": metadata_sha,
            }
        ),
        encoding="utf-8",
    )
    return license_plan, audio_report


def fixture(tmp_path: Path, *, seconds: dict[str, float] | None = None) -> dict[str, Any]:
    names = ["a.wav", "b.wav", "c.wav", "d.wav", "six.wav"]
    audio_root = tmp_path / "audio"
    for name in names:
        write_wav(audio_root / name, seconds=(seconds or {}).get(name, 1.0))
    metadata = tmp_path / "metadata.csv"
    metadata_sha = write_metadata(metadata, names)
    train = tmp_path / "train.csv"
    train_sha = write_train(train, names, six_row_name="six.wav")
    license_plan, audio_report = write_reports(tmp_path, names, metadata_sha)
    return {
        "names": names,
        "audio_root": audio_root,
        "metadata": metadata,
        "metadata_sha": metadata_sha,
        "train": train,
        "train_sha": train_sha,
        "license_plan": license_plan,
        "audio_report": audio_report,
    }


def run_builder(
    tmp_path: Path, fx: dict[str, Any], *, dev_count: int = 2
) -> tuple[int, dict[str, Any]]:
    code = audio_dev.main(
        [
            "--train-csv",
            str(fx["train"]),
            "--metadata-csv",
            str(fx["metadata"]),
            "--audio-root",
            str(fx["audio_root"]),
            "--license-plan",
            str(fx["license_plan"]),
            "--audio-preparation-report",
            str(fx["audio_report"]),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "media.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-train-sha256",
            str(fx["train_sha"]),
            "--expected-metadata-sha256",
            str(fx["metadata_sha"]),
            "--dev-count",
            str(dev_count),
        ]
    )
    return code, json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_builds_audio_dev_ledgers_and_media_manifest(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    code, report = run_builder(tmp_path, fx)
    assert code == 0
    assert report["status"] == "passed"
    assert report["training_admitted"] is False
    assert report["quality_gate"] == "not_run"
    assert report["gpu_used"] is False
    assert report["holdout_accessed"] is False
    assert report["real_human_kappa_verified"] is False
    assert report["primitive_scope"] == {
        "noul_only": True,
        "choice_covered": False,
        "score_covered": False,
    }
    assert report["split"]["dev_items"] == 2
    assert report["split"]["excluded_group_counts"] == {"six_row_group_excluded": 1}

    dev = read_jsonl(tmp_path / "dev.jsonl")
    ledger = read_jsonl(tmp_path / "dev-ledger.jsonl")
    media = read_jsonl(tmp_path / "media.jsonl")
    exclusion = read_jsonl(tmp_path / "exclusion.jsonl")
    assert len(dev) == len(ledger) == len(media) == 2
    assert all(row["meta"]["modality"] == ["audio"] for row in dev)
    assert all(row["questions"][0]["type"] == "noul" for row in dev)
    assert all(
        row["questions"][0]["target"]["source"] == "clotho_original_unanimous_three_answers"
        for row in dev
    )
    assert "answer" not in json.dumps({"ledger": ledger, "media": media})
    assert {row["label_origin"] for row in ledger} == {"hard_gold"}
    assert {row["license"] for row in media} == {"CC0-1.0"}
    selected_families = {row["meta"]["family_id"] for row in dev}
    excluded_families = {row["family_id"] for row in exclusion if row["pool"] == "dev_excluded"}
    assert selected_families == excluded_families

    second = tmp_path / "second"
    second.mkdir()
    code2, report2 = run_builder(second, fx)
    assert code2 == 0
    assert report2["outputs"] == report["outputs"]
    for name in ["dev.jsonl", "dev-ledger.jsonl", "exclusion.jsonl", "media.jsonl"]:
        assert (second / name).read_bytes() == (tmp_path / name).read_bytes()


def test_duration_filter_happens_before_selection(tmp_path: Path) -> None:
    fx = fixture(tmp_path, seconds={"a.wav": 31.0})
    code, report = run_builder(tmp_path, fx, dev_count=2)
    assert code == 0
    assert report["split"]["excluded_group_counts"] == {
        "duration_over_30s": 1,
        "six_row_group_excluded": 1,
    }
    selected_files = {row["original_file_name"] for row in read_jsonl(tmp_path / "media.jsonl")}
    assert "a.wav" not in selected_files


def test_rejects_missing_media_and_unapproved_license(tmp_path: Path) -> None:
    fx = fixture(tmp_path / "missing")
    (fx["audio_root"] / "a.wav").unlink()
    code, report = run_builder(tmp_path / "missing", fx)
    assert code == 1
    assert report["blocker"] == "approved media file missing"

    bad = tmp_path / "bad-license"
    bad.mkdir()
    names = ["a.wav", "b.wav", "c.wav"]
    audio_root = bad / "audio"
    for name in names:
        write_wav(audio_root / name)
    metadata = bad / "metadata.csv"
    metadata_sha = write_metadata(metadata, names, bad_license=True)
    train = bad / "train.csv"
    train_sha = write_train(train, names)
    license_plan, audio_report = write_reports(bad, names, metadata_sha)
    code, report = run_builder(
        bad,
        {
            "train": train,
            "train_sha": train_sha,
            "metadata": metadata,
            "metadata_sha": metadata_sha,
            "audio_root": audio_root,
            "license_plan": license_plan,
            "audio_report": audio_report,
        },
        dev_count=1,
    )
    assert code == 1
    assert report["blocker"] == "license plan disagrees with metadata"


def test_rejects_output_alias_existing_report_symlink_and_bad_dev_count(tmp_path: Path) -> None:
    fx = fixture(tmp_path)
    same = tmp_path / "same.jsonl"
    code = audio_dev.main(
        [
            "--train-csv",
            str(fx["train"]),
            "--metadata-csv",
            str(fx["metadata"]),
            "--audio-root",
            str(fx["audio_root"]),
            "--license-plan",
            str(fx["license_plan"]),
            "--audio-preparation-report",
            str(fx["audio_report"]),
            "--dev-out",
            str(same),
            "--dev-ledger-out",
            str(same),
            "--exclusion-ledger-out",
            str(tmp_path / "exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "media.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-train-sha256",
            fx["train_sha"],
            "--expected-metadata-sha256",
            fx["metadata_sha"],
            "--dev-count",
            "2",
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["blocker"] == "output paths must be unique"
    assert not same.exists()

    existing = tmp_path / "existing-report.json"
    existing.write_text("old evidence", encoding="utf-8")
    code = audio_dev.main(
        [
            "--train-csv",
            str(fx["train"]),
            "--metadata-csv",
            str(fx["metadata"]),
            "--audio-root",
            str(fx["audio_root"]),
            "--license-plan",
            str(fx["license_plan"]),
            "--audio-preparation-report",
            str(fx["audio_report"]),
            "--dev-out",
            str(tmp_path / "x-dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "x-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "x-exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "x-media.jsonl"),
            "--report",
            str(existing),
            "--expected-train-sha256",
            fx["train_sha"],
            "--expected-metadata-sha256",
            fx["metadata_sha"],
            "--dev-count",
            "2",
        ]
    )
    assert code == 1
    assert existing.read_text(encoding="utf-8") == "old evidence"

    dangling = tmp_path / "dangling.jsonl"
    dangling.symlink_to(tmp_path / "missing.jsonl")
    code = audio_dev.main(
        [
            "--train-csv",
            str(fx["train"]),
            "--metadata-csv",
            str(fx["metadata"]),
            "--audio-root",
            str(fx["audio_root"]),
            "--license-plan",
            str(fx["license_plan"]),
            "--audio-preparation-report",
            str(fx["audio_report"]),
            "--dev-out",
            str(dangling),
            "--dev-ledger-out",
            str(tmp_path / "s-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "s-exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "s-media.jsonl"),
            "--report",
            str(tmp_path / "symlink-report.json"),
            "--expected-train-sha256",
            fx["train_sha"],
            "--expected-metadata-sha256",
            fx["metadata_sha"],
            "--dev-count",
            "2",
        ]
    )
    report = json.loads((tmp_path / "symlink-report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["blocker"] == "refusing to overwrite existing outputs"
    assert dangling.is_symlink()

    bad = tmp_path / "bad-count"
    bad.mkdir()
    code, report = run_builder(bad, fx, dev_count=4)
    assert code == 1
    assert report["blocker"] == "dev_count must be between 1 and expected_count - 1"


def test_non_train_approved_media_is_not_read(tmp_path: Path) -> None:
    names = ["a.wav", "b.wav", "c.wav", "unused.wav"]
    audio_root = tmp_path / "audio"
    for name in names[:-1]:
        write_wav(audio_root / name)
    metadata = tmp_path / "metadata.csv"
    metadata_sha = write_metadata(metadata, names)
    train = tmp_path / "train.csv"
    train_sha = write_train(train, names[:-1])
    license_plan, audio_report = write_reports(tmp_path, names, metadata_sha)
    code, report = run_builder(
        tmp_path,
        {
            "train": train,
            "train_sha": train_sha,
            "metadata": metadata,
            "metadata_sha": metadata_sha,
            "audio_root": audio_root,
            "license_plan": license_plan,
            "audio_report": audio_report,
        },
        dev_count=1,
    )
    assert code == 0
    assert report["source"]["approved_audio_count"] == 4
    assert report["source"]["header_checked_train_audio_count"] == 3


def test_rejects_media_symlink_and_symlink_root(tmp_path: Path) -> None:
    fx = fixture(tmp_path / "file-link")
    target = fx["audio_root"] / "real.wav"
    (fx["audio_root"] / "a.wav").rename(target)
    (fx["audio_root"] / "a.wav").symlink_to(target)
    code, report = run_builder(tmp_path / "file-link", fx)
    assert code == 1
    assert report["blocker"] == "media symlinks are forbidden"

    root_case = tmp_path / "root-link"
    fx = fixture(root_case)
    real_root = fx["audio_root"]
    link_root = root_case / "audio-link"
    link_root.symlink_to(real_root, target_is_directory=True)
    fx["audio_root"] = link_root
    code, report = run_builder(root_case, fx)
    assert code == 1
    assert report["blocker"] == "asset root must be a real directory"


def test_rejects_short_tail_wav_decode(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    write_wav(path, seconds=1.0)
    path.write_bytes(path.read_bytes()[:-2])
    try:
        audio_dev.decode_wav_full(path)
    except audio_dev.AudioDevError as exc:
        assert str(exc) == "wav decode ended early"
    else:  # pragma: no cover - regression guard
        raise AssertionError("short WAV tail was accepted")


def test_rejects_metadata_url_and_attribution_conflict(tmp_path: Path) -> None:
    fx = fixture(tmp_path / "url")
    payload = json.loads(fx["license_plan"].read_text(encoding="utf-8"))
    payload["approved_audio"][0]["original_url"] = "https://freesound.org/people/user/sounds/9999"
    fx["license_plan"].write_text(json.dumps(payload), encoding="utf-8")
    code, report = run_builder(tmp_path / "url", fx)
    assert code == 1
    assert report["blocker"] == "license plan disagrees with metadata sound URL"

    fx = fixture(tmp_path / "attr")
    payload = json.loads(fx["license_plan"].read_text(encoding="utf-8"))
    payload["approved_audio"][0]["attribution"] = "other"
    fx["license_plan"].write_text(json.dumps(payload), encoding="utf-8")
    code, report = run_builder(tmp_path / "attr", fx)
    assert code == 1
    assert report["blocker"] == "license plan disagrees with metadata attribution"


def test_fixed_alias_can_map_media_filename_without_fuzzy_matching(tmp_path: Path) -> None:
    canonical = "souffle_me_tallique.wav"
    archive_name = "souffle_me.tallique.wav"
    names = [canonical, "b.wav", "c.wav"]
    audio_root = tmp_path / "audio"
    write_wav(audio_root / archive_name)
    write_wav(audio_root / "b.wav")
    write_wav(audio_root / "c.wav")
    metadata = tmp_path / "metadata.csv"
    metadata_sha = write_metadata(metadata, names)
    train = tmp_path / "train.csv"
    train_sha = write_train(train, names)
    license_plan, audio_report = write_reports(tmp_path, names, metadata_sha)
    payload = json.loads(audio_report.read_text(encoding="utf-8"))
    payload["archive_name_aliases"] = [
        {
            "metadata_file_name": canonical,
            "archive_member_name": archive_name,
            "mode": "fixed_revision_alias",
        }
    ]
    audio_report.write_text(json.dumps(payload), encoding="utf-8")
    code, report = run_builder(
        tmp_path,
        {
            "train": train,
            "train_sha": train_sha,
            "metadata": metadata,
            "metadata_sha": metadata_sha,
            "audio_root": audio_root,
            "license_plan": license_plan,
            "audio_report": audio_report,
        },
        dev_count=1,
    )
    assert code == 0
    media = read_jsonl(tmp_path / "media.jsonl")
    alias_rows = [row for row in media if row["original_file_name"] == canonical]
    if alias_rows:
        assert alias_rows[0]["media_file_name"] == archive_name
    assert report["source"]["fixed_alias_count"] == 1

    bad_report = tmp_path / "bad-audio-report.json"
    payload["archive_name_aliases"] = [
        {"metadata_file_name": "a.wav", "archive_member_name": "near-match.wav"}
    ]
    bad_report.write_text(json.dumps(payload), encoding="utf-8")
    code = audio_dev.main(
        [
            "--train-csv",
            str(train),
            "--metadata-csv",
            str(metadata),
            "--audio-root",
            str(audio_root),
            "--license-plan",
            str(license_plan),
            "--audio-preparation-report",
            str(bad_report),
            "--dev-out",
            str(tmp_path / "bad-dev.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "bad-ledger.jsonl"),
            "--exclusion-ledger-out",
            str(tmp_path / "bad-exclusion.jsonl"),
            "--media-manifest-out",
            str(tmp_path / "bad-media.jsonl"),
            "--report",
            str(tmp_path / "bad-report.json"),
            "--expected-train-sha256",
            train_sha,
            "--expected-metadata-sha256",
            metadata_sha,
            "--dev-count",
            "1",
        ]
    )
    bad = json.loads((tmp_path / "bad-report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert bad["blocker"] == "unexpected audio filename alias"
