"""Run a Kokoro text -> Misaki G2P -> tokenizer -> ONNX CPU WAV canary.

The module is import-safe in the project environment. Optional runtime packages
(numpy, onnxruntime, and Misaki) are imported only while executing the canary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import wave
from pathlib import Path
from typing import Any, cast


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], data)


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def load_runtime() -> tuple[Any, Any, Any]:
    np = importlib.import_module("numpy")
    ort = importlib.import_module("onnxruntime")
    misaki_en = importlib.import_module("misaki.en")
    return np, ort, misaki_en


def package_version(name: str) -> str | None:
    try:
        version = importlib.metadata.version(name)
    except Exception:
        return None
    return str(version)


def cgroup_snapshot() -> dict[str, str | None]:
    paths = {
        "cpu_max": "/sys/fs/cgroup/cpu.max",
        "cpuset_cpus_effective": "/sys/fs/cgroup/cpuset.cpus.effective",
        "memory_max": "/sys/fs/cgroup/memory.max",
        "memory_current": "/sys/fs/cgroup/memory.current",
    }
    result: dict[str, str | None] = {}
    for key, raw_path in paths.items():
        path = Path(raw_path)
        result[key] = path.read_text().strip() if path.exists() else None
    return result


def write_wav(path: Path, audio: Any, sample_rate: int, np: Any) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def load_vocab(tokenizer_path: Path) -> dict[str, int]:
    tokenizer = read_json(tokenizer_path)
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"tokenizer vocab missing in {tokenizer_path}")
    typed_vocab: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("tokenizer vocab contains a non-string token or non-int id")
        typed_vocab[token] = token_id
    return typed_vocab


def phonemes_to_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    unsupported = sorted({char for char in phonemes if char not in vocab})
    if unsupported:
        rendered = " ".join(f"U+{ord(char):04X}({char})" for char in unsupported)
        raise ValueError(f"unsupported phoneme/tokenizer characters: {rendered}")
    return [vocab[char] for char in phonemes]


def ensure_env(thread_env: dict[str, str]) -> None:
    for key, value in thread_env.items():
        os.environ[key] = value


def run_case(
    *,
    np: Any,
    ort: Any,
    session: Any,
    g2p: Any,
    vocab: dict[str, int],
    asset_dir: Path,
    output_dir: Path,
    model_max_length: int,
    sentence: dict[str, str],
    voice: dict[str, str],
) -> dict[str, Any]:
    text = sentence["text"]
    sentence_id = sentence["id"]
    voice_id = voice["id"]
    voice_path = asset_dir / voice["path"]
    require_sha256(voice_path, voice["sha256"], f"voice {voice_id}")

    g2p_result = g2p(text)
    if not isinstance(g2p_result, tuple) or len(g2p_result) < 1:
        raise ValueError("Misaki G2P returned an unexpected result shape")
    phonemes = g2p_result[0]
    if not isinstance(phonemes, str) or not phonemes:
        raise ValueError(f"Misaki G2P returned empty phonemes for {sentence_id}")
    token_ids = phonemes_to_ids(phonemes, vocab)
    if len(token_ids) > model_max_length - 2:
        raise ValueError(
            f"{sentence_id} has {len(token_ids)} tokens; "
            f"max without padding is {model_max_length - 2}"
        )

    voices = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 1, 256)
    style_index = len(token_ids)
    if style_index >= int(voices.shape[0]):
        raise ValueError(
            f"style index {style_index} for {sentence_id}/{voice_id} exceeds voice rows "
            f"{int(voices.shape[0])}"
        )
    style = voices[style_index]
    input_ids = np.array([[0, *token_ids, 0]], dtype=np.int64)
    speed = np.ones(1, dtype=np.float32)

    started = time.perf_counter()
    audio = session.run(None, {"input_ids": input_ids, "style": style, "speed": speed})[0]
    elapsed_seconds = round(time.perf_counter() - started, 6)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all():
        raise ValueError(f"audio contains non-finite samples for {sentence_id}/{voice_id}")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0 or math.isclose(peak, 0.0):
        raise ValueError(f"audio waveform is empty or silent for {sentence_id}/{voice_id}")

    wav_path = output_dir / f"{sentence_id}-{voice_id}.wav"
    write_wav(wav_path, audio, 24_000, np)
    return {
        "sentence_id": sentence_id,
        "text": text,
        "voice_id": voice_id,
        "voice_label": voice.get("label"),
        "voice_path": str(voice_path),
        "voice_sha256": sha256(voice_path),
        "phonemes": phonemes,
        "unsupported_phoneme_characters": [],
        "token_count_without_padding": len(token_ids),
        "input_ids_shape": list(input_ids.shape),
        "style_index": style_index,
        "voice_shape": list(voices.shape),
        "onnx_elapsed_seconds": elapsed_seconds,
        "sample_rate_hz": 24_000,
        "channels": 1,
        "samples": int(audio.size),
        "duration_seconds": round(float(audio.size / 24_000), 6),
        "finite_waveform": True,
        "float_peak_abs": peak,
        "float_mean": float(np.mean(audio)),
        "float_rms": float(np.sqrt(np.mean(np.square(audio)))),
        "wav_path": str(wav_path),
        "wav_bytes": wav_path.stat().st_size,
        "wav_sha256": sha256(wav_path),
    }


def run_canary(config_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    config = read_json(config_path)
    runtime = config["runtime"]
    ensure_env(runtime["thread_env"])
    np, ort, misaki_en = load_runtime()

    asset_dir = Path(config["asset_dir"])
    model_path = asset_dir / config["model"]["path"]
    tokenizer_path = asset_dir / config["tokenizer"]["path"]
    require_sha256(model_path, config["model"]["sha256"], "model")
    require_sha256(tokenizer_path, config["tokenizer"]["sha256"], "tokenizer")
    vocab = load_vocab(tokenizer_path)

    output_dir = Path(config["output"]["remote_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_report_path = report_path or Path(config["output"]["report"])
    final_report_path.parent.mkdir(parents=True, exist_ok=True)

    g2p = misaki_en.G2P(**config["g2p"]["settings"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = runtime["onnxruntime_threads"]["intra_op_num_threads"]
    options.inter_op_num_threads = runtime["onnxruntime_threads"]["inter_op_num_threads"]
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=[runtime["onnxruntime_provider"]]
    )

    cases = []
    for sentence in config["sentences"]:
        for voice in config["voices"]:
            cases.append(
                run_case(
                    np=np,
                    ort=ort,
                    session=session,
                    g2p=g2p,
                    vocab=vocab,
                    asset_dir=asset_dir,
                    output_dir=output_dir,
                    model_max_length=config["tokenizer"]["model_max_length"],
                    sentence=sentence,
                    voice=voice,
                )
            )

    freeze_subset = {
        name: package_version(name)
        for name in [
            "numpy",
            "onnxruntime",
            "misaki",
            "num2words",
            "spacy",
            "en-core-web-sm",
        ]
    }
    dependency_fingerprint = hashlib.sha256(
        json.dumps(freeze_subset, sort_keys=True).encode("utf-8")
    ).hexdigest()
    result = {
        "status": "passed",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "asset_dir": str(asset_dir),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "g2p": config["g2p"],
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__,
            "providers": session.get_providers(),
            "thread_env": {key: os.environ.get(key) for key in runtime["thread_env"]},
            "onnxruntime_threads": runtime["onnxruntime_threads"],
            "cgroup": cgroup_snapshot(),
        },
        "dependency_versions": freeze_subset,
        "dependency_fingerprint_sha256": dependency_fingerprint,
        "graph_inputs": {
            item.name: {"shape": item.shape, "type": item.type} for item in session.get_inputs()
        },
        "graph_outputs": {
            item.name: {"shape": item.shape, "type": item.type} for item in session.get_outputs()
        },
        "case_count": len(cases),
        "sentences": config["sentences"],
        "voices": config["voices"],
        "cases": cases,
        "quality_claim": "none; this is an execution and text-to-token coverage canary only",
    }
    final_report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_canary(args.config, args.report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
