"""Archive safety and controlled failure are observable offline validation behavior."""

import io
import json
import os
import stat
import struct
import subprocess
import wave
import zipfile
from pathlib import Path

import pytest

from scripts.public_eval_media import audio_worker, checked_zip_members, main, worker_decode

MEDIA_PYTHON = os.environ.get("PUBLIC_EVAL_MEDIA_PYTHON")


@pytest.mark.parametrize("name", ["../escape.mp4", "/absolute.mp4", "C:/drive.mp4", "a\\b"])
def test_archive_rejects_unsafe_names(name: str) -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, b"video")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="unsafe_zip_path"):
        checked_zip_members(archive)


def test_archive_rejects_symlink_before_any_extraction() -> None:
    data = io.BytesIO()
    link = zipfile.ZipInfo("star/link.mp4")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(link, "../../outside")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="special_file"):
        checked_zip_members(archive)


def test_archive_rejects_duplicate_canonical_path() -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("star/media", b"one")
        archive.writestr("star/media/", b"")
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="duplicate_zip_path"):
        checked_zip_members(archive)


def test_archive_rejects_total_expansion_over_budget() -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("star/video.mp4", b"0" * 1000)
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="expansion_limit"):
        checked_zip_members(archive, max_expanded_bytes=999)


def test_archive_accepts_normal_directory_and_video() -> None:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("star/", b"")
        archive.writestr("star/video.mp4", b"content")
    with zipfile.ZipFile(data) as archive:
        assert set(checked_zip_members(archive)) == {"star", "star/video.mp4"}


def test_cli_records_missing_media_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mmau": {"revision": "public", "file": {"path": "mini"}}}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "public_eval_media",
            "--data-root",
            str(tmp_path),
            "--sources",
            str(tmp_path),
            "--config",
            str(config),
            "--report",
            str(output),
        ],
    )
    assert main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "FileNotFoundError"
    assert report["training_admitted"] is False
    assert report["gpu_seconds"] == 0


@pytest.mark.skipif(not MEDIA_PYTHON, reason="explicit dedicated SoundFile runtime required")
def test_worker_decodes_every_pcm_frame(tmp_path: Path) -> None:
    audio = tmp_path / "finite.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<64h", *range(64)))
    assert MEDIA_PYTHON is not None
    with audio_worker(Path(MEDIA_PYTHON)) as worker:
        result = worker_decode(worker, audio)
    assert result["frames_decoded"] == result["frames_declared"] == 64
    assert result["finite_samples_verified"] is True
    assert worker.returncode == 0


@pytest.mark.skipif(not MEDIA_PYTHON, reason="explicit dedicated SoundFile runtime required")
def test_worker_rejects_non_finite_float_audio(tmp_path: Path) -> None:
    audio = tmp_path / "nonfinite.wav"
    assert MEDIA_PYTHON is not None
    subprocess.run(
        [
            MEDIA_PYTHON,
            "-c",
            "import numpy as n,soundfile as s,sys; "
            "s.write(sys.argv[1],n.array([0.,n.nan,0.]),16000,subtype='FLOAT')",
            str(audio),
        ],
        check=True,
    )
    with audio_worker(Path(MEDIA_PYTHON)) as worker:
        with pytest.raises(ValueError, match="soundfile_audio_decode_failed"):
            worker_decode(worker, audio)
