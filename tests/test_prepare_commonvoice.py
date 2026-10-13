import hashlib
import io
import json
import stat
import tarfile
import wave
from pathlib import Path

import pytest

from scripts import prepare_commonvoice as prep


def write_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 32)
    return buffer.getvalue()


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def make_archive(tmp_path: Path, entries: list[tuple[str, bytes]]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in entries:
            add_bytes(archive, name, data)
    return archive_path


def patch_expected(monkeypatch: pytest.MonkeyPatch, archive: Path) -> None:
    monkeypatch.setattr(prep, "EXPECTED_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        prep,
        "EXPECTED_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def test_archive_rejects_unsafe_paths_links_special_and_duplicates(tmp_path: Path) -> None:
    cases = []
    cases.append(make_archive(tmp_path / "up", [("../escape.mp3", b"x")]))
    cases.append(make_archive(tmp_path / "absolute", [("/escape.mp3", b"x")]))
    cases.append(
        make_archive(tmp_path / "duplicate", [("clips/a.mp3", b"x"), ("clips/a.mp3", b"y")])
    )
    link_path = tmp_path / "link" / "archive.tar.gz"
    link_path.parent.mkdir()
    with tarfile.open(link_path, "w:gz") as archive:
        info = tarfile.TarInfo("clips/link.mp3")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        archive.addfile(info)
    fifo_path = tmp_path / "fifo" / "archive.tar.gz"
    fifo_path.parent.mkdir()
    with tarfile.open(fifo_path, "w:gz") as archive:
        info = tarfile.TarInfo("clips/fifo")
        info.type = tarfile.FIFOTYPE
        info.mode = stat.S_IFIFO | 0o644
        archive.addfile(info)
    cases.extend([link_path, fifo_path])

    for archive in cases:
        with pytest.raises(ValueError, match="unsafe_tar|duplicate_tar|special_file|link_denied"):
            prep.audit_archive(archive)


def test_valid_archive_reports_schema_counts_and_redacts_sensitive_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_sentence = "private transcript must not leak"
    secret_client = "speaker-id-must-not-leak"
    metadata = (
        "client_id\tpath\tsentence\tlocale\tcustom_public\n"
        f"{secret_client}\tclips/a.wav\t{secret_sentence}\ten\tok\n"
    ).encode()
    archive = make_archive(
        tmp_path,
        [
            ("cv-corpus/metadata.tsv", metadata),
            ("cv-corpus/clips/a.wav", write_wav()),
        ],
    )
    patch_expected(monkeypatch, archive)
    monkeypatch.setattr(
        prep,
        "probe_audio",
        lambda path: {
            "codec": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "duration_seconds": 0.002,
            "bytes": path.stat().st_size,
            "decoder": "ffmpeg",
        },
    )

    result = prep.validate_commonvoice(archive, tmp_path / "isolated")
    rendered = json.dumps(result, sort_keys=True)

    assert result["status"] == "commonvoice_archive_cpu_ready"
    assert result["metadata"]["total_rows"] == 1
    assert result["audio_decode"]["decoded"] == 1
    assert result["training_admitted"] is False
    assert result["network_used"] is False
    assert secret_sentence not in rendered
    assert secret_client not in rendered
    assert "sentence" not in rendered
    assert "client_id" not in rendered
    assert "custom_public" in rendered
    assert (tmp_path / "isolated" / "metadata-schema").is_dir()


def test_audio_sample_selection_is_bounded_and_deterministic() -> None:
    names = [f"clips/{index:03d}.mp3" for index in range(80)]
    selected = prep.selected_samples(names)
    assert len(selected) == 32
    assert selected == prep.selected_samples(list(reversed(names)))
    assert selected[0] == "clips/000.mp3"


def test_cli_writes_failure_report_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_commonvoice",
            "--archive",
            str(tmp_path / "missing.tar.gz"),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(report),
        ],
    )

    assert prep.main() == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed"
    assert payload["training_admitted"] is False
    assert payload["network_used"] is False
    assert "synthetic-private-credential" not in report.read_text()
