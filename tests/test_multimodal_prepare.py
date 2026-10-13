"""Public input boundaries; these tests are not real-processor acceptance."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.s0_prepare import main, read_manifest
from train.multimodal import PreparationCase, file_hash, resolve_media


def public_case() -> dict:
    return {
        "id": "public-text",
        "category": "text",
        "prefix": {"state": "2 + 2 = 4"},
        "question": {"qid": "q", "type": "noul", "instruction": "Is the statement true?"},
        "meta": {
            "modality": ["text"],
            "source_dataset": "programmatic-arithmetic",
            "source_revision": "v1",
            "source_sha256": "a" * 64,
            "license": "CC0-1.0",
            "usage": "commercial",
            "pool": "dev",
            "family_id": "arithmetic-1",
            "data_version": "v1",
        },
    }


def test_holdout_and_empty_are_rejected() -> None:
    row = public_case()
    row["meta"]["pool"] = "holdout"
    with pytest.raises(ValidationError, match="holdout"):
        PreparationCase.model_validate(row)
    row = public_case()
    row["prefix"]["state"] = " "
    with pytest.raises(ValidationError, match="at least 1 character|empty"):
        PreparationCase.model_validate(row)


@pytest.mark.parametrize("uri", ["https://example.com/a", "file:a", "../a", "/tmp/a"])
def test_resolver_rejects_nonlocal_paths(tmp_path: Path, uri: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_media(tmp_path, uri, "a" * 64, max_bytes=20)


def test_resolver_rejects_links_hash_empty_and_limits(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    asset.write_bytes(b"real local bytes")
    digest = file_hash(asset)
    assert resolve_media(tmp_path, "asset", digest, max_bytes=100) == asset
    with pytest.raises(ValueError, match="SHA256"):
        resolve_media(tmp_path, "asset", "a" * 64, max_bytes=100)
    with pytest.raises(ValueError, match="limit"):
        resolve_media(tmp_path, "asset", digest, max_bytes=1)
    (tmp_path / "link").symlink_to(asset)
    with pytest.raises(ValueError, match="symlinks"):
        resolve_media(tmp_path, "link", digest, max_bytes=100)
    asset.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        resolve_media(tmp_path, "asset", digest, max_bytes=100)


def test_screenshot_is_image_with_provenance() -> None:
    row = public_case()
    row.update(category="screenshot")
    row["meta"]["modality"] = ["image"]
    row["prefix"]["media"] = [{"type": "image", "uri": "app.png", "sha256": "b" * 64}]
    assert PreparationCase.model_validate(row).category == "screenshot"
    row["prefix"]["media"][0]["type"] = "video"
    with pytest.raises(ValidationError, match="agree"):
        PreparationCase.model_validate(row)


def test_manifest_requires_explicit_partial_and_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([public_case()]))
    with pytest.raises(ValueError, match="100"):
        read_manifest(path)
    assert len(read_manifest(path, allow_partial=True)) == 1
    path.write_text(json.dumps([public_case(), public_case()]))
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path, allow_partial=True)


def test_cli_failure_keeps_private_content_out_of_report(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    row = public_case()
    row["prefix"]["state"] = "PRIVATE INPUT MUST NOT APPEAR"
    row["meta"]["pool"] = "holdout"
    manifest.write_text(json.dumps([row]))
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--asset-root",
                str(tmp_path),
                "--model-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--allow-partial",
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "blocked"
    assert report["complete_100_cases"] is False
    assert "PRIVATE INPUT" not in output.read_text()


def test_bad_media_is_rejected_before_processor(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from train.multimodal import load_media

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe unavailable; not media-decode acceptance")
    media = tmp_path / "bad.wav"
    media.write_bytes(b"not a WAV file")
    with pytest.raises(subprocess.CalledProcessError):
        load_media(media, "audio", num_frames=8, max_seconds=30)


def test_manifest_rejects_cross_pool_family(tmp_path: Path) -> None:
    first, second = public_case(), public_case()
    second["id"] = "second"
    second["meta"]["pool"] = "train"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([first, second]))
    with pytest.raises(ValueError, match="family crosses"):
        read_manifest(manifest, allow_partial=True)


def test_audio_duration_limit_rejects_without_truncating(tmp_path: Path) -> None:
    import shutil
    import wave

    from train.multimodal import load_media

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe unavailable; not media-decode acceptance")
    media = tmp_path / "duration.wav"
    with wave.open(str(media), "wb") as source:
        source.setnchannels(1)
        source.setsampwidth(2)
        source.setframerate(8000)
        source.writeframes(b"\0\0" * 800)
    with pytest.raises(ValueError, match="duration"):
        load_media(media, "audio", num_frames=8, max_seconds=0.01)
