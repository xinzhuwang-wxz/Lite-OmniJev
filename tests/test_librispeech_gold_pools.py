"""Only generated public canaries; no private source or sealed payload reads."""

import io
import json
import stat
import tarfile
import wave
from pathlib import Path

import pytest

from scripts import librispeech_gold_pools as builder
from scripts.gold_inventory import audit_manifest
from scripts.s1_text_pools import canonical_family_hash


def fake_flac(identity, duration=1):
    packed = (16000 << 44) | (15 << 36) | (16000 * duration)
    return (
        b"fLaC\x80\x00\x00\x22"
        + b"\x00" * 10
        + packed.to_bytes(8, "big")
        + b"\x00" * 16
        + identity.encode()
    )


def fake_decode(payload):
    content = io.BytesIO()
    with wave.open(content, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes.fromhex(builder.digest(payload)) * 1000)
    return content.getvalue()


def archive(path, split, offset, extra=None):
    data = {
        "LibriSpeech/CHAPTERS.TXT": (
            "; header\n"
            + "\n".join(
                f"{n} | {n} | 1.0 | {split} | 100 | {n} | book" for n in range(offset, offset + 5)
            )
        ).encode()
    }
    for n in range(offset, offset + 5):
        base = f"LibriSpeech/{split}/{n}/{n}"
        data[f"{base}/{n}-{n}.trans.txt"] = "\n".join(
            f"{n}-{n}-{u} THIS IS PUBLIC SYNTHETIC TRANSCRIPT NUMBER {n} UTTERANCE {u}"
            for u in range(4)
        ).encode()
        for u in range(4):
            identity = f"{n}-{n}-{u}"
            data[f"{base}/{identity}.flac"] = fake_flac(identity)
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in data.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        if extra:
            extra(tar)


def fixture(tmp_path):
    def bind(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": builder.file_digest(path)}

    history = {
        "id": "public-old",
        "prefix": {"state": "public", "media": []},
        "questions": [
            {
                "qid": "q",
                "type": "noul",
                "instruction": "public?",
                "target": {"kind": "hard", "value": "yes", "source": "synthetic"},
            }
        ],
        "meta": {
            "modality": ["text"],
            "source_dataset": "synthetic",
            "source_revision": "v1",
            "source_sha256": "a" * 64,
            "license": "CC0-1.0",
            "usage": "research",
            "pool": "dev",
            "family_id": "unrelated-public",
            "data_version": "v1",
        },
    }
    old = bind("history.jsonl", history)
    manifest = {
        "schema_version": "librispeech-gold-build-v1",
        "inventory": bind(
            "inventory.json",
            {
                "schema_version": "gold-inventory-manifest-v1",
                "samples": [
                    {**old, "cohort": "text", "exposure": "historical", "expected_pool": "dev"}
                ],
                "component_ledgers": [],
            },
        ),
        "quarantine": bind("quarantine.json", {"families": [], "media": []}),
    }
    for name, offset in (("dev_clean", 10), ("test_clean", 20)):
        path = tmp_path / (name + ".tar.gz")
        archive(path, name.replace("_", "-"), offset)
        manifest[name] = {"path": str(path), "sha256": builder.file_digest(path)}
    return manifest


def run(tmp_path, manifest, capsys, monkeypatch, name="out", canary=True, counts=(4, 4, 4)):
    monkeypatch.setattr(builder, "decode", fake_decode)
    path = tmp_path / "build.json"
    path.write_text(json.dumps(manifest))
    args = [
        "--manifest",
        str(path),
        "--output-dir",
        str(tmp_path / name),
        "--report",
        str(tmp_path / (name + ".receipt.json")),
    ]
    if canary:
        args.append("--canary")
    for pool, count in zip(builder.POOLS, counts, strict=True):
        args.extend([f"--{pool}-count", str(count)])
    code = builder.main(args)
    output = capsys.readouterr()
    assert output.err == ""
    assert "TRANSCRIPT" not in output.out and str(tmp_path) not in output.out
    receipt = json.loads(output.out)
    assert receipt["exit_code"] == code
    return receipt


def test_private_canary_complete_and_deterministic(tmp_path, capsys, monkeypatch):
    manifest = fixture(tmp_path)
    receipt = run(tmp_path, manifest, capsys, monkeypatch)
    assert receipt["outcome"] == "built"
    assert receipt["real_source_verified"] is False
    assert receipt["full_gold_gate"] == receipt["inventory_completeness"] == "not_validated"
    assert receipt["elapsed_seconds"] > 0
    assert receipt["cost_usd"] is None
    assert audit_manifest(tmp_path / "out/inventory-manifest.json").outcome == "isolated"
    run(tmp_path, manifest, capsys, monkeypatch, name="out2")
    groups = [
        json.loads(line) for line in (tmp_path / "out/allocation.jsonl").read_text().splitlines()
    ]
    pools = {}
    for group in groups:
        for alias in group["aliases"]:
            assert alias not in pools or pools[alias] == group["pool"]
            pools[alias] = group["pool"]
    for pool in builder.POOLS:
        rows = [
            json.loads(line) for line in (tmp_path / f"out/{pool}.jsonl").read_text().splitlines()
        ]
        other = [
            json.loads(line) for line in (tmp_path / f"out2/{pool}.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 4
        assert [r["questions"] for r in rows] == [r["questions"] for r in other]
        texts = {
            r["questions"][0]["options"][ord(r["questions"][0]["target"]["value"]) - 65]
            for r in rows
        }
        for row in rows:
            q = row["questions"][0]
            assert len(set(q["options"])) == 4
            assert set(q["options"]) == texts
            assert "canary-not-real" in row["meta"]["data_version"]
    for path in (tmp_path / "out").rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


@pytest.mark.parametrize(
    "counts,canary,category",
    [
        ((3, 4, 4), True, "invalid_quota"),
        ((40, 4, 4), True, "insufficient_capacity"),
        ((4, 4, 4), False, "invalid_quota"),
        ((1600, 500, 500), False, "source_not_pinned"),
    ],
)
def test_quota_and_source_pins(tmp_path, capsys, monkeypatch, counts, canary, category):
    receipt = run(tmp_path, fixture(tmp_path), capsys, monkeypatch, counts=counts, canary=canary)
    assert receipt["category"] == category
    assert receipt["outcome"] == "rejected"
    assert (tmp_path / "out.receipt.json").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "hardlink", "duplicate"])
def test_unsafe_archive_rejected(tmp_path, capsys, monkeypatch, kind):
    manifest = fixture(tmp_path)

    def extra(tar):
        name = "../PRIVATE_SENTINEL" if kind == "traversal" else "LibriSpeech/CHAPTERS.TXT"
        info = tarfile.TarInfo(name)
        if kind in ("symlink", "hardlink"):
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = "/PRIVATE_SENTINEL"
        tar.addfile(info, io.BytesIO())

    path = Path(manifest["dev_clean"]["path"])
    archive(path, "dev-clean", 10, extra)
    manifest["dev_clean"]["sha256"] = builder.file_digest(path)
    receipt = run(tmp_path, manifest, capsys, monkeypatch)
    assert receipt["category"] in ("unsafe_archive", "duplicate_archive_member")
    assert "PRIVATE_SENTINEL" not in json.dumps(receipt)


def test_component_bridges_retained_before_eligibility():
    def row(i, aliases, eligible=True):
        return builder.Utterance(
            str(i), "dev-clean", "public", "public", "a" * 64, "b" * 64, set(aliases), eligible
        )

    rows = [
        row(1, ["speaker:1", "book:1"]),
        row(2, ["book:1", "text:1"], False),
        row(3, ["text:1", "speaker:2"]),
        row(4, ["speaker:2", "media:1"]),
        row(5, ["media:1", "book:3"]),
    ]
    groups = builder.components(rows)
    assert len(groups) == 1
    assert len(groups[0][2]) == 5
    assert groups[0][0] == builder.components(list(reversed(rows)))[0][0]


@pytest.mark.parametrize("kind", ["families", "media"])
def test_quarantine_excludes_whole_component(tmp_path, capsys, monkeypatch, kind):
    manifest = fixture(tmp_path)
    rows = builder.source_rows(Path(manifest["dev_clean"]["path"]), "dev-clean")
    groups = builder.components(rows)
    value = (
        canonical_family_hash(groups[0][0]) if kind == "families" else groups[0][2][0].media_sha256
    )
    path = Path(manifest["quarantine"]["path"])
    path.write_text(json.dumps({"families": [], "media": [], kind: [value]}))
    manifest["quarantine"]["sha256"] = builder.file_digest(path)
    receipt = run(tmp_path, manifest, capsys, monkeypatch)
    assert receipt["outcome"] == "built"
    assert receipt["counts"]["excluded_components"] == 1
    assert groups[0][0] not in (tmp_path / "out/allocation.jsonl").read_text()


def test_bound_history_and_alias_ledger(tmp_path, capsys, monkeypatch):
    manifest = fixture(tmp_path)
    run(tmp_path, manifest, capsys, monkeypatch)
    ledger = tmp_path / "out/allocation.jsonl"
    manifest["alias_ledgers"] = [{"path": str(ledger), "sha256": builder.file_digest(ledger)}]
    receipt = run(tmp_path, manifest, capsys, monkeypatch, name="next")
    assert receipt["outcome"] == "built"
    assert receipt["counts"]["excluded_components"] == 3
    historical = json.loads((tmp_path / "inventory.json").read_text())
    Path(historical["samples"][0]["path"]).write_text("PRIVATE_SENTINEL")
    receipt = run(tmp_path, manifest, capsys, monkeypatch, name="invalid")
    assert receipt["category"] == "hash_mismatch"


def test_reserved_unused_component_members(tmp_path, capsys, monkeypatch):
    receipt = run(tmp_path, fixture(tmp_path), capsys, monkeypatch, counts=(5, 5, 5))
    assert receipt["outcome"] == "built"
    ledger = [
        json.loads(line) for line in (tmp_path / "out/allocation.jsonl").read_text().splitlines()
    ]
    assert sum(not m["selected"] for g in ledger for m in g["members"]) == 9


def test_no_overwrite_and_malformed_metadata(tmp_path, capsys, monkeypatch):
    manifest = fixture(tmp_path)
    run(tmp_path, manifest, capsys, monkeypatch)
    original = (tmp_path / "out/calibration.jsonl").read_bytes()
    receipt = run(tmp_path, manifest, capsys, monkeypatch)
    assert receipt["outcome"] == "rejected"
    assert (tmp_path / "out/calibration.jsonl").read_bytes() == original
    manifest["SECRET_SENTINEL"] = "PRIVATE_SENTINEL"
    receipt = run(tmp_path, manifest, capsys, monkeypatch, name="malformed")
    assert receipt["category"] == "invalid_metadata"
    assert "SENTINEL" not in json.dumps(receipt)


def test_invalid_decode_does_not_claim_built(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        builder,
        "validate_wav",
        lambda _: (_ for _ in ()).throw(builder.BuildError("invalid_decoded_audio")),
    )
    receipt = run(tmp_path, fixture(tmp_path), capsys, monkeypatch)
    assert receipt["category"] == "invalid_decoded_audio"
    assert receipt["outcome"] == "rejected"


@pytest.mark.parametrize("kind", ["chapter", "transcript", "audio"])
def test_malformed_source_metadata(tmp_path, capsys, monkeypatch, kind):
    manifest = fixture(tmp_path)
    path = Path(manifest["dev_clean"]["path"])
    with tarfile.open(path, "r:gz") as tar:
        values = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    if kind == "chapter":
        values["LibriSpeech/CHAPTERS.TXT"] = b"PRIVATE_SENTINEL | invalid"
    elif kind == "transcript":
        values["LibriSpeech/dev-clean/10/10/10-10.trans.txt"] = b"PRIVATE_SENTINEL"
    else:
        values["LibriSpeech/dev-clean/10/10/10-10-0.flac"] = b"PRIVATE_SENTINEL"
    with tarfile.open(path, "w:gz") as tar:
        for name, value in values.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            tar.addfile(info, io.BytesIO(value))
    manifest["dev_clean"]["sha256"] = builder.file_digest(path)
    receipt = run(tmp_path, manifest, capsys, monkeypatch)
    assert (
        receipt["category"]
        == {
            "chapter": "invalid_chapters",
            "transcript": "invalid_transcript",
            "audio": "invalid_audio",
        }[kind]
    )
    assert "SENTINEL" not in json.dumps(receipt)


def test_historical_speech_requires_full_alias_ledger(tmp_path, capsys, monkeypatch):
    manifest = fixture(tmp_path)
    run(tmp_path, manifest, capsys, monkeypatch)
    output_inventory = json.loads((tmp_path / "out/inventory-manifest.json").read_text())
    source_inventory = Path(manifest["inventory"]["path"])
    source_inventory.write_text(json.dumps(output_inventory))
    manifest["inventory"]["sha256"] = builder.file_digest(source_inventory)
    receipt = run(tmp_path, manifest, capsys, monkeypatch, name="without-aliases")
    assert receipt["category"] == "missing_alias_history"
    ledger = tmp_path / "out/allocation.jsonl"
    manifest["alias_ledgers"] = [{"path": str(ledger), "sha256": builder.file_digest(ledger)}]
    receipt = run(tmp_path, manifest, capsys, monkeypatch, name="with-aliases")
    assert receipt["outcome"] == "built"
    assert receipt["counts"]["excluded_components"] == 3


def test_history_family_excludes_component(tmp_path, capsys, monkeypatch):
    manifest = fixture(tmp_path)
    rows = builder.source_rows(Path(manifest["dev_clean"]["path"]), "dev-clean")
    family = builder.components(rows)[0][0]
    history_path = tmp_path / "history.jsonl"
    history = json.loads(history_path.read_text())
    history["meta"]["family_id"] = family
    history_path.write_text(json.dumps(history))
    inventory_path = Path(manifest["inventory"]["path"])
    inventory = json.loads(inventory_path.read_text())
    inventory["samples"][0]["sha256"] = builder.file_digest(history_path)
    inventory_path.write_text(json.dumps(inventory))
    manifest["inventory"]["sha256"] = builder.file_digest(inventory_path)
    receipt = run(tmp_path, manifest, capsys, monkeypatch)
    assert receipt["counts"]["excluded_components"] == 1
    assert family not in (tmp_path / "out/allocation.jsonl").read_text()


def test_duplicate_source_identity_rejected():
    row = builder.Utterance("1", "dev-clean", "public", "public", "a" * 64, "b" * 64, {"a"}, True)
    with pytest.raises(builder.BuildError, match="duplicate_utterance"):
        builder.components([row, row])


def test_actual_ffmpeg_roundtrip():
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable; real decoding not exercised")
    wav = fake_decode(b"public synthetic PCM")
    flac = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "flac", "pipe:1"],
        input=wav,
        capture_output=True,
        check=True,
    ).stdout
    decoded = builder.decode(flac)
    builder.validate_wav(decoded)
    with wave.open(io.BytesIO(decoded), "rb") as stream:
        assert stream.getframerate() == 16000
        frames = stream.readframes(480001)
        assert stream.getnframes() == len(frames) // 2 == 16000
    assert decoded[:4] == b"RIFF"
    assert int.from_bytes(decoded[4:8], "little") == len(decoded) - 8
    assert decoded[36:40] == b"data"
    assert int.from_bytes(decoded[40:44], "little") == len(frames)


def test_actual_ffmpeg_oversized_decode_rejected():
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable; real decoding not exercised")
    oversized = io.BytesIO()
    with wave.open(oversized, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 480001)
    with pytest.raises(builder.BuildError, match="invalid_decoded_duration"):
        builder.decode(oversized.getvalue())
