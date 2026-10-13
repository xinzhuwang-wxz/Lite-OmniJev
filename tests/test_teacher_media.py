"""Exercise preprocessing through its public context, without a model or dataset."""

from __future__ import annotations

import hashlib
import shutil
import sys
import wave
from pathlib import Path

import pytest

from train.teacher_media import TeacherAudioError, prepared_teacher_audio


def _audio(path: Path, *, seconds: float = 0.1, rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * int(seconds * rate) * channels)
    return path


def _executable(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}\n")
    path.chmod(0o700)
    return str(path)


def test_real_ffmpeg_caps_multichannel_audio_and_cleans_owned_output(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("real FFmpeg is required for the CPU media integration test")
    source = _audio(tmp_path / "source.wav", seconds=32.25, rate=48000, channels=6)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    parent = tmp_path / "temporary"
    parent.mkdir()
    unrelated = parent / "user-file.wav"
    unrelated.write_bytes(b"do not replace")

    with prepared_teacher_audio(source, temp_parent=parent, ffmpeg_binary=ffmpeg) as output:
        assert output.is_file() and output != source
        with wave.open(str(output), "rb") as stream:
            assert (stream.getnchannels(), stream.getframerate(), stream.getsampwidth()) == (
                1,
                16000,
                2,
            )
            assert stream.getnframes() == 480000
            assert len(stream.readframes(480001)) == 960000
        assert unrelated.read_bytes() == b"do not replace"
    assert not output.exists()
    assert list(parent.iterdir()) == [unrelated]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize("payload", [b"", b"not a wave file"])
def test_success_exit_cannot_admit_empty_or_unreadable_audio(
    tmp_path: Path, payload: bytes
) -> None:
    source = _audio(tmp_path / "source.wav")
    executable = _executable(
        tmp_path / "invalid-converter",
        f"import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes({payload!r})",
    )
    parent = tmp_path / "temporary"
    parent.mkdir()
    with pytest.raises(TeacherAudioError):
        with prepared_teacher_audio(source, temp_parent=parent, ffmpeg_binary=executable):
            pytest.fail("invalid converted output must not be yielded")
    assert list(parent.iterdir()) == []
    assert source.is_file()


def test_converter_failure_cleans_partial_output(tmp_path: Path) -> None:
    source = _audio(tmp_path / "source.wav")
    executable = _executable(
        tmp_path / "failed-converter",
        "import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes(b'partial')\nsys.exit(3)",
    )
    parent = tmp_path / "temporary"
    parent.mkdir()
    with pytest.raises(TeacherAudioError, match="conversion failed"):
        with prepared_teacher_audio(source, temp_parent=parent, ffmpeg_binary=executable):
            pytest.fail("failed conversion must not be yielded")
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("kind", ["truncated", "wrong-rate"])
def test_readable_header_cannot_admit_incomplete_or_wrong_rate_pcm(
    tmp_path: Path, kind: str
) -> None:
    source = _audio(tmp_path / "source.wav")
    bad = _audio(tmp_path / "bad.wav", rate=8000 if kind == "wrong-rate" else 16000)
    payload = bad.read_bytes()[:-2] if kind == "truncated" else bad.read_bytes()
    executable = _executable(
        tmp_path / "bad-pcm-converter",
        f"import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes({payload!r})",
    )
    parent = tmp_path / "temporary"
    parent.mkdir()
    with pytest.raises(TeacherAudioError):
        with prepared_teacher_audio(source, temp_parent=parent, ffmpeg_binary=executable):
            pytest.fail("invalid converted PCM must not be yielded")
    assert list(parent.iterdir()) == []


def test_converter_timeout_cleans_partial_output(tmp_path: Path) -> None:
    source = _audio(tmp_path / "source.wav")
    executable = _executable(
        tmp_path / "slow-converter",
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'partial')\ntime.sleep(10)",
    )
    parent = tmp_path / "temporary"
    parent.mkdir()
    with pytest.raises(TeacherAudioError, match="timed out"):
        with prepared_teacher_audio(
            source, temp_parent=parent, ffmpeg_binary=executable, timeout_seconds=0.1
        ):
            pytest.fail("timed-out conversion must not be yielded")
    assert list(parent.iterdir()) == []


def test_caller_failure_still_cleans_converted_audio(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("real FFmpeg is required for the CPU media integration test")
    source = _audio(tmp_path / "source.wav")
    parent = tmp_path / "temporary"
    parent.mkdir()
    with pytest.raises(RuntimeError, match="caller failed"):
        with prepared_teacher_audio(source, temp_parent=parent, ffmpeg_binary=ffmpeg):
            raise RuntimeError("caller failed")
    assert list(parent.iterdir()) == []
