"""Run a deterministic Kokoro ONNX CPU WAV canary.

This helper intentionally uses fixed token IDs from the ONNX model card so it
checks ONNX graph execution and voice embedding shape without depending on a
text phonemizer path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import wave
from pathlib import Path
from typing import Any

README_TOKENS = [
    50,
    157,
    43,
    135,
    16,
    53,
    135,
    46,
    16,
    43,
    102,
    16,
    56,
    156,
    57,
    135,
    6,
    16,
    102,
    62,
    61,
    16,
    70,
    56,
    16,
    138,
    56,
    156,
    72,
    56,
    61,
    85,
    123,
    83,
    44,
    83,
    54,
    16,
    53,
    65,
    156,
    86,
    61,
    62,
    131,
    83,
    56,
    4,
    16,
    54,
    156,
    43,
    102,
    53,
    16,
    156,
    72,
    61,
    53,
    102,
    112,
    16,
    70,
    56,
    16,
    138,
    56,
    44,
    156,
    76,
    158,
    123,
    56,
    16,
    62,
    131,
    156,
    43,
    102,
    54,
    46,
    16,
    102,
    48,
    16,
    81,
    47,
    102,
    54,
    16,
    54,
    156,
    51,
    158,
    46,
    16,
    70,
    16,
    92,
    156,
    135,
    46,
    16,
    54,
    156,
    43,
    102,
    48,
    4,
    16,
    81,
    47,
    102,
    16,
    50,
    156,
    72,
    64,
    83,
    56,
    62,
    16,
    156,
    51,
    158,
    64,
    83,
    56,
    16,
    44,
    157,
    102,
    56,
    16,
    44,
    156,
    76,
    158,
    123,
    56,
    4,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime() -> tuple[Any, Any]:
    """Load optional runtime dependencies only when executing the canary."""
    np = importlib.import_module("numpy")
    ort = importlib.import_module("onnxruntime")
    return np, ort


def write_wav(path: Path, audio: Any, sample_rate: int, np: Any) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def run_canary(model: Path, voice: Path, wav: Path, report: Path) -> dict[str, Any]:
    np, ort = load_runtime()
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    voices = np.fromfile(voice, dtype=np.float32).reshape(-1, 1, 256)
    ref_s = voices[len(README_TOKENS)]
    input_ids = np.array([[0, *README_TOKENS, 0]], dtype=np.int64)
    speed = np.ones(1, dtype=np.float32)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    audio = session.run(None, {"input_ids": input_ids, "style": ref_s, "speed": speed})[0]
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all():
        raise ValueError("audio contains non-finite samples")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0 or math.isclose(peak, 0.0):
        raise ValueError("audio waveform is empty or silent")

    wav.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_wav(wav, audio, 24_000, np)
    result = {
        "status": "passed",
        "command_summary": (
            "Direct ONNX Runtime CPU inference using fixed Kokoro README token IDs, "
            "af_heart voice embedding, speed=1.0, ORT_DISABLE_TELEMETRY=1, one ORT CPU thread."
        ),
        "python": sys.version.split()[0],
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "model_path": str(model),
        "model_sha256": sha256(model),
        "voice_path": str(voice),
        "voice_sha256": sha256(voice),
        "voice_shape": list(voices.shape),
        "selected_style_index": len(README_TOKENS),
        "token_count_without_padding": len(README_TOKENS),
        "input_ids_shape": list(input_ids.shape),
        "graph_inputs": {
            item.name: {"shape": item.shape, "type": item.type} for item in session.get_inputs()
        },
        "graph_outputs": {
            item.name: {"shape": item.shape, "type": item.type} for item in session.get_outputs()
        },
        "providers": session.get_providers(),
        "sample_rate_hz": 24_000,
        "channels": 1,
        "samples": int(audio.size),
        "duration_seconds": round(float(audio.size / 24_000), 6),
        "finite_waveform": True,
        "float_peak_abs": peak,
        "float_mean": float(np.mean(audio)),
        "float_rms": float(np.sqrt(np.mean(np.square(audio)))),
        "wav_path": str(wav),
        "wav_bytes": wav.stat().st_size,
        "wav_sha256": sha256(wav),
        "quality_claim": "none; this is an execution canary only",
    }
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--voice", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_canary(args.model, args.voice, args.wav, args.report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
