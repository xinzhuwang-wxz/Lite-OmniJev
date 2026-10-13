"""CPU-only Gemma 4 processor probe for development media canaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from data.schema import Question, answer_tokens
from scripts.assets import write_json
from scripts.check_tokens import OfficialTokenizerAdapter

ANSWER_PRIMITIVES = {
    "choice": Question(
        qid="choice",
        type="choice",
        instruction="Choose the correct canary label.",
        options=[f"option {token}" for token in "ABCDEFGHIJKLMNOPQRST"],
    ),
    "score": Question(
        qid="score",
        type="score",
        instruction="Score the canary from 0 to 9.",
        options=None,
        score_levels=10,
    ),
    "noul": Question(
        qid="noul",
        type="noul",
        instruction="Is the canary present?",
        options=None,
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_files(model_dir: Path) -> dict[str, dict[str, int | str]]:
    if not model_dir.is_dir():
        raise ValueError(f"model dir does not exist: {model_dir}")
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(p for p in model_dir.iterdir() if p.is_file()):
        result[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    required = {
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"missing required model files: {missing}")
    return result


def make_canaries(work_dir: Path) -> dict[str, Any]:
    import numpy as np

    Image = importlib.import_module("PIL.Image")
    ImageDraw = importlib.import_module("PIL.ImageDraw")

    work_dir.mkdir(parents=True, exist_ok=True)
    image_path = work_dir / "canary-image.png"
    screenshot_path = work_dir / "canary-screenshot.png"
    wav_path = work_dir / "canary-sine-16khz.wav"
    video_path = work_dir / "canary-video-3frames.mp4"
    frames_dir = work_dir / "video-frames"
    frames_dir.mkdir(exist_ok=True)

    image = Image.new("RGB", (64, 48), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 56, 40], outline="red", width=3)
    draw.text((12, 18), "IMG", fill="black")
    image.save(image_path)

    screenshot = Image.new("RGB", (96, 64), "#eeeeee")
    screen = ImageDraw.Draw(screenshot)
    screen.rectangle([5, 5, 91, 59], outline="blue", width=2)
    screen.rectangle([18, 24, 78, 44], fill="#ffffff", outline="#333333")
    screen.text((26, 29), "OK", fill="black")
    screenshot.save(screenshot_path)

    sample_rate = 16000
    seconds = 0.5
    samples = (
        0.2 * np.sin(2 * math.pi * 440 * np.arange(int(sample_rate * seconds)) / sample_rate)
    ).astype(np.float32)
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=3",
            "-frames:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            str(frames_dir / "frame-%02d.png"),
        ],
        check=True,
    )
    frames = [Image.open(path).convert("RGB") for path in sorted(frames_dir.glob("frame-*.png"))]
    if len(frames) != 3:
        raise ValueError(f"expected 3 extracted video frames, got {len(frames)}")

    media_files = {
        "image": image_path,
        "screenshot": screenshot_path,
        "audio_wav": wav_path,
        "video_mp4": video_path,
        **{
            f"video_frame_{index}": path
            for index, path in enumerate(sorted(frames_dir.glob("frame-*.png")))
        },
    }
    return {
        "image": image,
        "screenshot": screenshot,
        "audio": samples,
        "video_frames": frames,
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in media_files.items()
        },
        "audio_sample_rate": sample_rate,
    }


def tensor_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    if not hasattr(value, "shape"):
        return {"type": type(value).__name__}
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    return {
        "type": type(value).__name__,
        "dtype": str(getattr(value, "dtype", array.dtype)),
        "shape": list(array.shape),
        "sha256": sha256_bytes(array.tobytes()),
    }


def answer_boundary_bindings(adapter: OfficialTokenizerAdapter, prompt: str) -> dict[str, Any]:
    prompt_ids = adapter.encode(prompt, add_special_tokens=False)
    result: dict[str, Any] = {}
    failures: list[str] = []
    special_ids = {
        token_id
        for token_id, token in adapter.tokenizer.get_added_tokens_decoder().items()
        if token.special
    }
    unknown_id = adapter.tokenizer.token_to_id(adapter.config.get("unk_token", "<unk>"))
    for primitive, question in ANSWER_PRIMITIVES.items():
        mapping: dict[str, int] = {}
        primitive_failures: list[str] = []
        for token in (*answer_tokens(question), "U"):
            ids = adapter.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                primitive_failures.append(f"{token}: expected one token, got {len(ids)}")
                continue
            token_id = ids[0]
            if token_id in special_ids or token_id == unknown_id:
                primitive_failures.append(
                    f"{token}: resolved to special or unknown token {token_id}"
                )
            if adapter.tokenizer.decode(ids, skip_special_tokens=False) != token:
                primitive_failures.append(f"{token}: decode mismatch")
            continuation = adapter.encode(prompt + token, add_special_tokens=False)
            if continuation != prompt_ids + ids:
                primitive_failures.append(f"{token}: generation boundary changed prefix or token")
            mapping[token] = token_id
        if len(set(mapping.values())) != len(mapping):
            primitive_failures.append("token ID collision")
        if primitive_failures:
            failures.extend(f"{primitive}: {failure}" for failure in primitive_failures)
        result[primitive] = {
            "answer_tokens": list(mapping),
            "answer_token_ids": mapping,
            "answer_count_including_u": len(mapping),
            "failures": primitive_failures,
        }
    result["_failures"] = failures
    return result


def probe(model_dir: Path, work_dir: Path) -> dict[str, Any]:
    transformers = importlib.import_module("transformers")
    AutoProcessor = transformers.AutoProcessor

    script_path = Path(__file__) if "__file__" in globals() else Path("scripts/cpu_media_probe.py")
    model_hashes = hash_model_files(model_dir)
    canaries = make_canaries(work_dir)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    adapter = OfficialTokenizerAdapter(model_dir)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "audio"},
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        "Development canary only. State contains an image, a screenshot, "
                        "a 16 kHz sine wave, and a 3-frame video. Return exactly one answer token."
                    ),
                },
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("official processor chat template returned empty prompt")
    placeholder_counts = {
        token: rendered.count(token)
        for token in ("<|image|>", "<|audio|>", "<|video|>", "<|think|>")
    }
    if placeholder_counts != {"<|image|>": 2, "<|audio|>": 1, "<|video|>": 1, "<|think|>": 0}:
        raise ValueError(f"unexpected template placeholder counts: {placeholder_counts}")

    inputs = processor(
        text=[rendered],
        images=[[canaries["image"], canaries["screenshot"]]],
        audio=[canaries["audio"]],
        videos=[canaries["video_frames"]],
        num_frames=3,
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )
    output_summary = {key: tensor_summary(value) for key, value in inputs.items()}
    required_outputs = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_position_ids",
        "input_features",
        "input_features_mask",
        "pixel_values_videos",
        "video_position_ids",
        "num_frames_per_video",
        "mm_token_type_ids",
    }
    missing = sorted(required_outputs - output_summary.keys())
    if missing:
        raise ValueError(f"processor output missing required keys: {missing}")

    bindings = answer_boundary_bindings(adapter, rendered)
    failures = list(bindings.pop("_failures"))
    return {
        "status": "failed" if failures else "passed",
        "exit_code": 1 if failures else 0,
        "scope": (
            "CPU AutoProcessor/tokenizer media canary only; no model weights loaded, "
            "no logits, no GPU, no S0 parity pass"
        ),
        "model_dir": str(model_dir.resolve()),
        "work_dir": str(work_dir.resolve()),
        "source": {
            "script_sha256": sha256_file(script_path) if script_path.is_file() else None,
            "transformers_version": transformers.__version__,
            "processor_class": type(processor).__name__,
            "processor_module": type(processor).__module__,
        },
        "model_files": model_hashes,
        "media": canaries["files"],
        "audio_sample_rate": canaries["audio_sample_rate"],
        "template": {
            "sha256": sha256_bytes(rendered.encode()),
            "placeholder_counts": placeholder_counts,
            "prompt_tokens": len(adapter.encode(rendered, add_special_tokens=False)),
            "preview": rendered[:500],
        },
        "processor_inputs": {
            "kwargs": {
                "text": "one rendered prompt",
                "images": "two PIL images",
                "audio": "one float32 16 kHz sine array",
                "videos": "three PIL frames extracted from ffmpeg MP4",
                "num_frames": 3,
                "return_tensors": "pt",
                "return_mm_token_type_ids": True,
            }
        },
        "processor_outputs": output_summary,
        "answer_boundary_bindings": bindings,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/cpu-media-probe.json"))
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    work_dir = args.work_dir
    temp: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temp = tempfile.TemporaryDirectory(prefix="omnijev-cpu-media-probe-")
        work_dir = Path(temp.name)
    try:
        report = probe(args.model_dir, work_dir)
    except Exception as exc:
        report = {
            "status": "blocked",
            "exit_code": 1,
            "blocker": str(exc)
            if isinstance(exc, (ValueError, FileNotFoundError))
            else type(exc).__name__,
            "scope": (
                "CPU AutoProcessor/tokenizer media canary only; "
                "fail-closed before any model/GPU claim"
            ),
        }
    finally:
        if temp is not None:
            temp.cleanup()
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
