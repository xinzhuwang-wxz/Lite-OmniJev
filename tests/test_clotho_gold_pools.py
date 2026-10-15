"""Synthetic private-job behavior; no real sealed inputs."""

import csv
import io
import json
import stat
import wave
import zipfile
from pathlib import Path

import pytest

from scripts import clotho_gold_pools as gold
from scripts.clevrer_gold_pools import BuildError
from scripts.gold_inventory import audit_manifest
from scripts.s0_audio_dev import sample_for


def fixture(tmp_path: Path) -> tuple[Path, dict]:
    def artifact(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        return {"path": str(path), "sha256": gold.digest(content)}

    def csv_bytes(columns, rows):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode()

    audio = tmp_path / "audio"
    audio.mkdir()
    metadata, approved, questions = [], [], []
    for i in range(7):
        name = f"sound{i}.wav"
        with wave.open(str(audio / name), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes([i, 0]) * 160)
        url = f"https://freesound.org/people/example/sounds/{i}/"
        metadata.append(
            dict(
                file_name=name,
                sound_link=url,
                license="https://creativecommons.org/publicdomain/zero/1.0/",
                manufacturer="author",
            )
        )
        approved.append(
            dict(
                file_name=name,
                original_url=url,
                license_url=metadata[-1]["license"],
                attribution="author",
                license="CC0-1.0",
            )
        )
        for q in range(2):
            for _ in range(3):
                questions.append(
                    dict(
                        file_name=name,
                        QuestionText=f"Secret question {i} {q}?",
                        answer="yes",
                        confidence="high",
                    )
                )
    meta = artifact("metadata.csv", csv_bytes(list(metadata[0]), metadata))
    plan = artifact(
        "plan.json", gold.encode({"metadata_sha256": meta["sha256"], "approved_audio": approved})
    )
    source = artifact("train.csv", csv_bytes(list(questions[0]), questions))
    empty = csv_bytes(list(questions[0]), [])
    archive_path = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in audio.iterdir():
            archive.write(path, f"audio_files/{path.name}")
    manifest = {
        "schema_version": "clotho-gold-build-v1",
        "train": source,
        "val": artifact("val.csv", empty),
        "test": artifact("test.csv", empty),
        "metadata": meta,
        "license_plan": plan,
        "archive": {"path": str(archive_path), "sha256": gold.file_digest(archive_path)},
        "audio_root": str(audio),
        "quarantine": artifact("quarantine.json", gold.encode({"families": [], "media": []})),
    }
    # collect requires the complete schema, supplied after synthetic reserves exist.
    from scripts.s0_audio_dev import (
        candidate_rows,
        load_license_plan,
        load_metadata,
        load_train_groups,
    )

    candidates, _, _ = candidate_rows(
        groups=load_train_groups(Path(source["path"]), train_sha256=source["sha256"]),
        metadata=load_metadata(
            Path(meta["path"]),
            metadata_sha256=meta["sha256"],
            approved=load_license_plan(Path(plan["path"]), metadata_sha256=meta["sha256"]),
        ),
        audio_root=audio,
        aliases={},
    )
    artifacts = []
    reserves = []
    for i, pool in enumerate(("calibration", "holdout", "dev")):
        sample = sample_for(next(r for r in candidates if r["sound_id"] == str(i)))
        raw = sample.model_dump(mode="json")
        raw["meta"]["pool"] = pool
        raw["prefix"]["media"][0]["uri"] = (audio / f"sound{i}.wav").as_uri()
        a = artifact(f"old-{pool}.jsonl", gold.encode(raw))
        artifacts.append({**a, "cohort": "audio", "exposure": "current", "expected_pool": pool})
        if pool != "dev":
            reserves.append(a)
    manifest["current_reserves"] = reserves
    manifest["inventory"] = artifact(
        "inventory.json",
        gold.encode(
            {
                "schema_version": "gold-inventory-manifest-v1",
                "samples": artifacts,
                "component_ledgers": [],
            }
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(gold.encode(manifest))
    return path, manifest


def run(tmp_path, manifest_path, *, canary=True, cal=5, hold=4):
    args = [
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(tmp_path / "output"),
        "--report",
        str(tmp_path / "receipt.json"),
        "--calibration-count",
        str(cal),
        "--holdout-count",
        str(hold),
    ]
    if canary:
        args.append("--canary")
    code = gold.main(args)
    return code, json.loads((tmp_path / "receipt.json").read_bytes())


def test_preserves_existing_and_isolates_all_history(tmp_path, capsys):
    path, manifest = fixture(tmp_path)
    code, receipt = run(tmp_path, path)
    assert code == 0
    assert receipt["counts"]["calibration"] == 5
    assert receipt["counts"]["holdout"] == 4
    assert receipt["counts"]["preserved_items"] == 2
    assert not receipt["real_source_verified"]
    assert "Secret question" not in capsys.readouterr().out
    for pool, old in zip(("calibration", "holdout"), manifest["current_reserves"], strict=True):
        original = gold.Sample.model_validate_json(Path(old["path"]).read_bytes()).model_dump()
        output = [
            gold.Sample.model_validate_json(line).model_dump()
            for line in (tmp_path / "output" / f"{pool}.jsonl").read_text().splitlines()
        ]
        assert original in output
    assert audit_manifest(tmp_path / "output/inventory-manifest.json").outcome == "isolated"
    assert stat.S_IMODE((tmp_path / "output").stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(p.stat().st_mode) == 0o600
        for p in (tmp_path / "output").rglob("*")
        if p.is_file()
    )
    old_bytes = (tmp_path / "output/calibration.jsonl").read_bytes()
    assert (
        gold.main(
            [
                "--manifest",
                str(path),
                "--output-dir",
                str(tmp_path / "output"),
                "--report",
                str(tmp_path / "other-report.json"),
                "--canary",
            ]
        )
        == 1
    )
    assert (tmp_path / "output/calibration.jsonl").read_bytes() == old_bytes


def test_wrong_current_gold_rejected(tmp_path):
    path, manifest = fixture(tmp_path)
    old = manifest["current_reserves"][0]
    raw = json.loads(Path(old["path"]).read_bytes())
    raw["questions"][0]["target"]["value"] = "no"
    Path(old["path"]).write_bytes(gold.encode(raw))
    old["sha256"] = gold.file_digest(Path(old["path"]))
    inv = json.loads(Path(manifest["inventory"]["path"]).read_bytes())
    inv["samples"][0]["sha256"] = old["sha256"]
    Path(manifest["inventory"]["path"]).write_bytes(gold.encode(inv))
    manifest["inventory"]["sha256"] = gold.file_digest(Path(manifest["inventory"]["path"]))
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(tmp_path, path)
    assert code == 1 and receipt["category"] == "invalid_existing_gold"


@pytest.mark.parametrize("cal,hold", [(100, 4), (5, 3)])
def test_missing_capacity(tmp_path, cal, hold):
    path, _ = fixture(tmp_path)
    code, receipt = run(tmp_path, path, cal=cal, hold=hold)
    assert code == 1 and receipt["category"] == "insufficient_capacity"


def test_exact_deterministic_subset_sum():
    sizes = {"a": 4, "b": 2, "c": 3, "d": 1}
    chosen = gold.subset_sum(sizes, 5)
    assert sum(sizes[k] for k in chosen) == 5
    assert gold.subset_sum(dict(reversed(list(sizes.items()))), 5) == chosen
    with pytest.raises(BuildError, match="insufficient_capacity"):
        gold.subset_sum({"a": 2}, 1)


@pytest.mark.parametrize("attack", ["pin", "unsafe_zip", "media_changed", "family_overlap"])
def test_fail_closed(tmp_path, attack, capsys):
    path, manifest = fixture(tmp_path)
    if attack == "unsafe_zip":
        with zipfile.ZipFile(manifest["archive"]["path"], "a") as archive:
            archive.writestr("../private-secret.wav", b"x")
        manifest["archive"]["sha256"] = gold.file_digest(Path(manifest["archive"]["path"]))
    elif attack == "media_changed":
        media = Path(manifest["audio_root"]) / "sound6.wav"
        media.write_bytes((Path(manifest["audio_root"]) / "sound5.wav").read_bytes())
    elif attack == "family_overlap":
        invpath = Path(manifest["inventory"]["path"])
        inv = json.loads(invpath.read_bytes())
        original = json.loads(Path(manifest["current_reserves"][0]["path"]).read_bytes())
        original["meta"]["pool"] = "train"
        conflict = tmp_path / "conflict.jsonl"
        conflict.write_bytes(gold.encode(original))
        inv["samples"].append(
            {
                "path": str(conflict),
                "sha256": gold.file_digest(conflict),
                "cohort": "audio",
                "exposure": "historical",
                "expected_pool": "train",
            }
        )
        invpath.write_bytes(gold.encode(inv))
        manifest["inventory"]["sha256"] = gold.file_digest(invpath)
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(
        tmp_path,
        path,
        canary=attack != "pin",
        cal=1400 if attack == "pin" else 5,
        hold=500 if attack == "pin" else 4,
    )
    assert code == 1
    assert receipt["category"] in {
        "source_not_pinned",
        "unsafe_archive",
        "media_source_mismatch",
        "ambiguous_media_alias",
        "historical_pool_conflict",
    }
    assert "private-secret" not in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_duplicate_archive_basename_rejected(tmp_path):
    path, manifest = fixture(tmp_path)
    with zipfile.ZipFile(manifest["archive"]["path"], "a") as archive:
        archive.writestr("another/sound0.wav", b"different")
    manifest["archive"]["sha256"] = gold.file_digest(Path(manifest["archive"]["path"]))
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(tmp_path, path)
    assert code == 1 and receipt["category"] == "duplicate_archive_member"


def test_quarantined_existing_cannot_be_silently_replaced(tmp_path):
    path, manifest = fixture(tmp_path)
    quarantine = Path(manifest["quarantine"]["path"])
    quarantine.write_bytes(
        gold.encode({"families": [gold.canonical_family_hash("0")], "media": []})
    )
    manifest["quarantine"]["sha256"] = gold.file_digest(quarantine)
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(tmp_path, path)
    assert code == 1 and receipt["category"] == "invalid_existing_gold"


def test_original_validation_split_is_bound_and_named_correctly(tmp_path):
    path, manifest = fixture(tmp_path)
    train_path = Path(manifest["train"]["path"])
    rows = list(csv.DictReader(io.StringIO(train_path.read_text())))
    chosen = [
        r
        for r in rows
        if r["file_name"] in {"sound3.wav", "sound4.wav", "sound5.wav", "sound6.wav"}
    ]
    for split, selected in (("train", [r for r in rows if r not in chosen]), ("val", chosen)):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
        source = Path(manifest[split]["path"])
        source.write_text(stream.getvalue())
        manifest[split]["sha256"] = gold.file_digest(source)
    path.write_bytes(gold.encode(manifest))
    code, _ = run(tmp_path, path)
    assert code == 0
    records = [
        json.loads(line) for line in (tmp_path / "output/allocation.jsonl").read_text().splitlines()
    ]
    assert any(r["source_splits"] == ["val"] for r in records)
    assert all(
        r["item_id"].startswith("clotho-aqa-val-gold-")
        for r in records
        if r["source_splits"] == ["val"]
    )
    provenance = json.loads((tmp_path / "output/provenance.json").read_bytes())
    assert provenance["inputs"]["val"]["sha256"] == manifest["val"]["sha256"]
    assert provenance["cost_usd"] is None
