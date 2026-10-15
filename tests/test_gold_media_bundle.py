"""Public synthetic relocation tests; no private gold contents."""

import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.gold_inventory import digest
from scripts.gold_media_bundle import bundle, main
from tests.test_gold_inventory import SECRET, sample


def setup(tmp_path: Path, uris: list[str] | None = None) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "trusted"
    root.mkdir()
    media = root / "asset.wav"
    media.write_bytes(b"public synthetic bytes")
    row = sample(modality="audio")
    rows = []
    for index, uri in enumerate(uris or ["asset.wav", media.as_uri()]):
        item = json.loads(json.dumps(row))
        item["id"] += str(index)
        item["prefix"]["media"][0].update(uri=uri, sha256=digest(media.read_bytes()))
        rows.append(item)
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "gold-media-bundle-manifest-v1",
                "samples": [
                    {
                        "path": str(source),
                        "sha256": digest(source.read_bytes()),
                        "cohort": "audio",
                        "exposure": "historical",
                        "expected_pool": "dev",
                        "media_root": str(root),
                    }
                ],
            }
        )
    )
    return manifest, source, rows[0]


def test_bundle_preserves_payload_deduplicates_and_is_private(tmp_path: Path) -> None:
    manifest, source, _ = setup(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "output"
    receipt = bundle(manifest, output)
    assert receipt.outcome == "bundled"
    assert (receipt.artifacts, receipt.samples, receipt.media_references, receipt.unique_media) == (
        1,
        2,
        2,
        1,
    )
    assert source.read_bytes() == original
    new = [json.loads(line) for line in (output / "samples-0000.jsonl").read_text().splitlines()]
    old = [json.loads(line) for line in original.splitlines()]
    for before, after in zip(old, new, strict=True):
        uri = after["prefix"]["media"][0]["uri"]
        assert (output / uri).read_bytes() == b"public synthetic bytes"
        after["prefix"]["media"][0]["uri"] = before["prefix"]["media"][0]["uri"]
        assert after == before
    inventory = json.loads((output / "inventory-manifest.json").read_text())
    assert inventory["samples"][0]["exposure"] == "historical"
    assert inventory["samples"][0]["sha256"] == digest((output / "samples-0000.jsonl").read_bytes())
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert SECRET not in receipt.model_dump_json()
    assert str(tmp_path) not in receipt.model_dump_json()
    assert bundle(manifest, output).categories == ["output_exists"]


@pytest.mark.parametrize(
    "uri",
    [
        "../asset.wav",
        "https://example.com/asset.wav",
        "file://host/a.wav",
        "asset.wav?secret",
        "asset.wav#secret",
        "/absolute.wav",
        "file:relative.wav",
        "%2e%2e/asset.wav",
        "missing.wav",
    ],
)
def test_bad_paths_rejected(tmp_path: Path, uri: str) -> None:
    manifest, _, _ = setup(tmp_path, [uri])
    assert bundle(manifest, tmp_path / "output").outcome == "rejected"


@pytest.mark.parametrize("change", ["symlink", "hash", "artifact_hash", "type"])
def test_invalid_media_rejected(tmp_path: Path, change: str) -> None:
    manifest, source, _ = setup(tmp_path)
    if change == "symlink":
        media = tmp_path / "trusted" / "asset.wav"
        other = tmp_path / "other.wav"
        media.rename(other)
        media.symlink_to(other)
    elif change == "hash":
        (tmp_path / "trusted" / "asset.wav").write_bytes(b"changed")
    elif change == "artifact_hash":
        source.write_text(source.read_text() + "\n")
    else:
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[1]["prefix"]["media"][0]["type"] = "image"
        rows[1]["meta"]["modality"] = ["image"]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        value = json.loads(manifest.read_text())
        value["samples"][0]["sha256"] = digest(source.read_bytes())
        manifest.write_text(json.dumps(value))
    result = bundle(manifest, tmp_path / "output")
    assert result.outcome == "rejected"
    assert result.categories == [
        {
            "symlink": "unsafe_path",
            "hash": "hash_mismatch",
            "artifact_hash": "hash_mismatch",
            "type": "type_conflict",
        }[change]
    ]


def test_cli_failure_redacted_and_no_report_overwrite(tmp_path: Path, capsys: Any) -> None:
    manifest, _, _ = setup(tmp_path, [f"../{SECRET}"])
    report = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.gold_media_bundle",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "output"),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert process.stderr == ""
    assert process.stdout == report.read_text()
    assert SECRET not in process.stdout and str(tmp_path) not in process.stdout
    before = report.read_bytes()
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "other"),
                "--report",
                str(report),
            ]
        )
        == 1
    )
    assert report.read_bytes() == before
    assert "output_exists" in capsys.readouterr().out


def test_absolute_outside_and_directory_symlink_rejected(tmp_path: Path) -> None:
    manifest, source, _ = setup(tmp_path)
    root = tmp_path / "trusted"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "asset.wav").write_bytes(b"public synthetic bytes")
    (root / "link").symlink_to(outside, target_is_directory=True)
    for uri in ((outside / "asset.wav").as_uri(), "link/asset.wav"):
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        for row in rows:
            row["prefix"]["media"][0]["uri"] = uri
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        value = json.loads(manifest.read_text())
        value["samples"][0]["sha256"] = digest(source.read_bytes())
        manifest.write_text(json.dumps(value))
        output = tmp_path / ("outside-output" if uri.startswith("file") else "link-output")
        assert bundle(manifest, output).categories == ["unsafe_path"]


def test_cli_success(tmp_path: Path, capsys: Any) -> None:
    manifest, _, _ = setup(tmp_path)
    report = tmp_path / "receipt.json"
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "output"),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == report.read_text()
