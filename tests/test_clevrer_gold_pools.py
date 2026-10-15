"""Public synthetic fixtures only: no sealed data is opened by these tests."""

import json
import stat
import zipfile
from pathlib import Path

import pytest

from scripts.clevrer_gold_pools import file_digest, main
from scripts.gold_inventory import audit_manifest
from scripts.s0_media_dev import clevrer_video_member
from scripts.s1_text_pools import canonical_family_hash


def fixture(tmp_path: Path):
    def bind(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": file_digest(path)}

    history = {
        "id": "old",
        "prefix": {"state": "public", "media": []},
        "questions": [
            {
                "qid": "q",
                "type": "noul",
                "instruction": "public question",
                "target": {"kind": "hard", "value": "yes", "source": "fixture"},
            }
        ],
        "meta": {
            "modality": ["text"],
            "source_dataset": "fixture",
            "source_revision": "v1",
            "source_sha256": "a" * 64,
            "license": "CC0-1.0",
            "usage": "research",
            "pool": "dev",
            "family_id": "clevrer:10000",
            "data_version": "v1",
        },
    }
    old = bind("old.jsonl", history)
    inventory = bind(
        "inventory.json",
        {
            "schema_version": "gold-inventory-manifest-v1",
            "samples": [
                {**old, "cohort": "text", "exposure": "historical", "expected_pool": "dev"}
            ],
            "component_ledgers": [],
        },
    )
    scenes = [
        {
            "scene_index": i,
            "questions": [
                {
                    "question_id": 0,
                    "question_type": "descriptive",
                    "question_subtype": "exist",
                    "question": f"Public synthetic question {i}",
                    "answer": "yes",
                },
                {
                    "question_id": 1,
                    "question_type": "descriptive",
                    "question_subtype": "exist",
                    "question": f"Other public question {i}",
                    "answer": "no",
                },
            ],
        }
        for i in range(10000, 10008)
    ]
    questions = bind("questions.json", scenes)
    video = tmp_path / "video.zip"
    with zipfile.ZipFile(video, "w") as archive:
        for scene in scenes:
            index = scene["scene_index"]
            archive.writestr(clevrer_video_member(index), f"public synthetic media {index}")
    manifest = {
        "schema_version": "clevrer-gold-build-v1",
        "questions": questions,
        "videos": {"path": str(video), "sha256": file_digest(video)},
        "readme": bind("readme.json", "public fixture"),
        "inventory": inventory,
        "quarantine": bind(
            "quarantine.json", {"families": [canonical_family_hash("clevrer:10001")], "media": []}
        ),
    }
    path = tmp_path / "build.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def run(tmp_path, path, capsys, name="out", canary=True):
    args = [
        "--manifest",
        str(path),
        "--output-dir",
        str(tmp_path / name),
        "--report",
        str(tmp_path / (name + ".report.json")),
        "--calibration-count",
        "3",
        "--holdout-count",
        "1",
    ]
    if canary:
        args.append("--canary")
    rc = main(args)
    output = capsys.readouterr()
    assert output.err == ""
    assert str(tmp_path) not in output.out
    assert "Public synthetic" not in output.out
    assert "clevrer:100" not in output.out
    return rc, json.loads(output.out)


def test_build_native_questions_deterministic_and_inventory_compatible(tmp_path, capsys):
    path, _ = fixture(tmp_path)
    rc, report = run(tmp_path, path, capsys)
    assert rc == 0
    assert report["real_source_verified"] is False
    assert report["counts"]["calibration"] == 3
    assert report["counts"]["holdout"] == 1
    assert report["counts"]["excluded_family"] == 2
    out = tmp_path / "out"
    assert audit_manifest(out / "inventory-manifest.json").outcome == "isolated"
    rows = [json.loads(line) for line in (out / "allocation.jsonl").read_text().splitlines()]
    assert len({row["family_id"] for row in rows}) == 4
    for file in out.rglob("*"):
        assert stat.S_IMODE(file.stat().st_mode) == (0o700 if file.is_dir() else 0o600)
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    rc, _ = run(tmp_path, path, capsys, "second")
    assert rc == 0
    assert (out / "allocation.jsonl").read_bytes() == (
        tmp_path / "second/allocation.jsonl"
    ).read_bytes()
    before = (out / "calibration.jsonl").read_bytes()
    assert run(tmp_path, path, capsys)[0] == 1
    assert before == (out / "calibration.jsonl").read_bytes()


@pytest.mark.parametrize(
    "case",
    [
        "traversal",
        "symlink",
        "duplicate",
        "hash",
        "gold",
        "identity",
        "media_duplicate",
        "history_hash",
    ],
)
def test_rejects_unsafe_inputs_without_disclosure(tmp_path, capsys, case):
    path, manifest = fixture(tmp_path)
    if case in ("traversal", "symlink", "duplicate"):
        with zipfile.ZipFile(manifest["videos"]["path"], "a") as archive:
            if case == "traversal":
                archive.writestr("../SECRET", "x")
            elif case == "symlink":
                info = zipfile.ZipInfo("link")
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "SECRET")
            else:
                archive.writestr("elsewhere/video_10000.mp4", "x")
        manifest["videos"]["sha256"] = file_digest(Path(manifest["videos"]["path"]))
    elif case == "hash":
        manifest["questions"]["sha256"] = "0" * 64
    elif case == "history_hash":
        (tmp_path / "old.jsonl").write_text("SECRET")
    elif case in ("gold", "identity"):
        p = Path(manifest["questions"]["path"])
        scenes = json.loads(p.read_text())
        if case == "gold":
            for scene in scenes:
                for q in scene["questions"]:
                    q["answer"] = "SECRET"
        else:
            scenes[0]["scene_index"] = 0
        p.write_text(json.dumps(scenes))
        manifest["questions"]["sha256"] = file_digest(p)
    else:
        p = Path(manifest["videos"]["path"])
        with zipfile.ZipFile(p, "w") as archive:
            for i in range(10000, 10008):
                archive.writestr(clevrer_video_member(i), "same")
        manifest["videos"]["sha256"] = file_digest(p)
    path.write_text(json.dumps(manifest))
    rc, report = run(tmp_path, path, capsys)
    assert rc == 1
    assert report["outcome"] == "rejected"
    assert "SECRET" not in json.dumps(report)
    assert not (tmp_path / "out").exists()


def test_synthetic_cannot_enter_real_mode(tmp_path, capsys):
    path, _ = fixture(tmp_path)
    rc, report = run(tmp_path, path, capsys, canary=False)
    assert rc == 1
    assert report["real_source_verified"] is False


def test_crc_corruption_is_rejected(tmp_path, capsys):
    path, manifest = fixture(tmp_path)
    archive_path = Path(manifest["videos"]["path"])
    payload = archive_path.read_bytes()
    assert b"public synthetic media 10002" in payload
    archive_path.write_bytes(
        payload.replace(b"public synthetic media 10002", b"broken synthetic media 10002")
    )
    manifest["videos"]["sha256"] = file_digest(archive_path)
    path.write_text(json.dumps(manifest))
    rc, report = run(tmp_path, path, capsys)
    assert rc == 1
    assert report["category"] == "invalid_archive"


def test_real_mode_rejects_unpinned_source_at_real_quota(tmp_path, capsys):
    path, _ = fixture(tmp_path)
    assert (
        main(
            [
                "--manifest",
                str(path),
                "--output-dir",
                str(tmp_path / "out"),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["category"] == "source_not_pinned"


def test_union_historical_versions_and_quarantined_media(tmp_path, capsys):
    path, manifest = fixture(tmp_path)
    old = json.loads((tmp_path / "old.jsonl").read_text())
    old["meta"]["family_id"] = "clevrer:10002"
    old["meta"]["data_version"] = "v2"
    historical = tmp_path / "older.jsonl"
    historical.write_text(json.dumps(old))
    inventory_path = Path(manifest["inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["samples"].append(
        {**inventory["samples"][0], "path": str(historical), "sha256": file_digest(historical)}
    )
    inventory_path.write_text(json.dumps(inventory))
    manifest["inventory"]["sha256"] = file_digest(inventory_path)
    quarantine_path = Path(manifest["quarantine"]["path"])
    quarantine = json.loads(quarantine_path.read_text())
    import hashlib

    quarantine["media"] = [hashlib.sha256(b"public synthetic media 10003").hexdigest()]
    quarantine_path.write_text(json.dumps(quarantine))
    manifest["quarantine"]["sha256"] = file_digest(quarantine_path)
    path.write_text(json.dumps(manifest))
    rc, report = run(tmp_path, path, capsys)
    assert rc == 0
    assert report["counts"]["excluded_family"] == 3
    assert report["counts"]["excluded_media_duplicate"] == 1
