"""Project-owned audio preparation for the pinned Jev-Omni input contract.

This context does not call a model or modify the upstream loader. Its yielded
file must be consumed by the teacher processor before leaving the context.
"""

from __future__ import annotations

import math
import stat
import subprocess
import tempfile
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_FRAMES = SAMPLE_RATE * MAX_AUDIO_SECONDS


class TeacherAudioError(ValueError):
    """Conversion did not produce a complete audio input meeting the contract."""


def _validate_wav(path: Path) -> None:
    output_stat = path.lstat()
    if not stat.S_ISREG(output_stat.st_mode) or not 0 < output_stat.st_size <= 2 * 1024**2:
        raise TeacherAudioError("converted audio must be a nonempty bounded regular file")
    try:
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels() != 1
                or stream.getframerate() != SAMPLE_RATE
                or stream.getsampwidth() != 2
                or stream.getcomptype() != "NONE"
                or not 0 < stream.getnframes() <= MAX_AUDIO_FRAMES
            ):
                raise TeacherAudioError("converted audio violates mono/16kHz/PCM16/30s contract")
            expected_bytes = stream.getnframes() * 2
            decoded_bytes = 0
            while block := stream.readframes(65536):
                decoded_bytes += len(block)
            if decoded_bytes != expected_bytes:
                raise TeacherAudioError("converted audio has incomplete PCM frames")
    except (wave.Error, EOFError, OSError) as error:
        raise TeacherAudioError("converted audio is not a readable PCM WAV") from error


@contextmanager
def prepared_teacher_audio(
    source: Path,
    *,
    temp_parent: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 30,
) -> Iterator[Path]:
    """Yield a validated temporary WAV; clean it on conversion or caller failure.

    Only the adapter's exclusive temporary output is overwritten. ``source`` is
    read only. The first <=30 seconds become mono16kHz PCM16, using FFmpeg's
    default channel remix/resampling, matching the reference WAV output intent.
    ``ffmpeg_binary`` lets the caller select an audited installed executable.
    """
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 180:
        raise ValueError("audio conversion timeout must be finite and in (0, 180]")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("teacher audio source must be a regular local file")
    with tempfile.TemporaryDirectory(prefix="teacher-audio-", dir=temp_parent) as directory:
        output = Path(directory) / "audio.wav"
        # Deliberately precreate an owned empty output: -y fixes the reference
        # NamedTemporaryFile overwrite failure without touching caller files.
        output.touch(mode=0o600, exist_ok=False)
        command = [
            ffmpeg_binary,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(source),
            "-t",
            str(MAX_AUDIO_SECONDS),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-threads",
            "1",
            str(output),
        ]
        try:
            subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired as error:
            raise TeacherAudioError("teacher audio conversion timed out") from error
        except (subprocess.CalledProcessError, OSError) as error:
            raise TeacherAudioError("teacher audio conversion failed") from error
        try:
            _validate_wav(output)
        except OSError as error:
            raise TeacherAudioError("converted audio output is missing or unreadable") from error
        yield output
