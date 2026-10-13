"""Bounded local media preparation shared by offline decision callers."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from data.schema import Contract, Metadata, Prefix, Question
from train.template import check_answer_tokens, render_payload


class PreparationCase(Contract):
    id: str = Field(min_length=1)
    category: Literal["text", "image", "screenshot", "audio", "video"]
    video_frames: Literal[8, 16] = 16
    prefix: Prefix
    question: Question
    meta: Metadata

    @model_validator(mode="after")
    def public_input(self) -> Self:
        if self.meta.pool == "holdout":
            raise ValueError("holdout is forbidden for development preparation")
        expected = "image" if self.category == "screenshot" else self.category
        if expected == "text":
            if self.prefix.media:
                raise ValueError("text cases cannot carry media")
        elif not self.prefix.media or any(m.type != expected for m in self.prefix.media):
            raise ValueError("category and media must agree")
        if not self.prefix.state.strip() or not self.question.instruction.strip():
            raise ValueError("empty decision input")
        if expected not in self.meta.modality:
            raise ValueError("category must be declared in metadata")
        return self


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media(root: Path, uri: str, sha256: str, *, max_bytes: int) -> Path:
    """Only relative regular files, without links, beneath an explicit trusted root."""
    relative = Path(uri)
    if ":" in uri or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("media URI must be a relative local asset path")
    if root.is_symlink() or root.absolute() != root.resolve() or not root.is_dir():
        raise ValueError("asset root must be a real directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("media symlinks are forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("media must be a regular file inside asset root")
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError("media file is empty or exceeds byte limit")
    if file_hash(path) != sha256:
        raise ValueError("media SHA256 mismatch")
    return path


def load_media(path: Path, kind: str, *, num_frames: int, max_seconds: float) -> Any:
    """Decode only preverified local files; reject overlong inputs before decoding."""
    if kind == "image":
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(path) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("image exceeds pixel limit")
            source.load()
            return source.convert("RGB")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    probed = json.loads(result.stdout)
    duration = float(probed["format"]["duration"])
    if not 0 < duration <= max_seconds:
        raise ValueError("media duration is invalid or exceeds limit")
    if kind == "audio":
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        np = importlib.import_module("numpy")
        audio = np.frombuffer(result.stdout, dtype="<f4").copy()
        if audio.size == 0 or audio.size > int(max_seconds * 16000) or not np.isfinite(audio).all():
            raise ValueError("decoded audio is empty, nonfinite or over limit")
        return audio
    stream = next(s for s in probed["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(stream["avg_frame_rate"]))
    total_frames = int(stream["nb_frames"])
    if fps <= 0 or total_frames < num_frames:
        raise ValueError("video lacks sufficient frame metadata")
    indices = [round(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    image_module = importlib.import_module("PIL.Image")
    with tempfile.TemporaryDirectory(prefix="omnijev-video-") as directory:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(path),
                "-vf",
                f"select={selection},scale=640:640:force_original_aspect_ratio=decrease",
                "-vsync",
                "0",
                "-frames:v",
                str(num_frames),
                str(Path(directory) / "%04d.png"),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        frames = []
        for frame in sorted(Path(directory).glob("*.png")):
            with image_module.open(frame) as source:
                frames.append(source.convert("RGB"))
    if len(frames) != num_frames:
        raise ValueError("decoded video does not match requested frame count")
    return {
        "frames": frames,
        "metadata": {
            "fps": fps,
            "total_num_frames": total_frames,
            "duration": duration,
            "frames_indices": indices,
            "video_backend": "ffmpeg",
            "width": frames[0].width,
            "height": frames[0].height,
        },
    }


@dataclass(frozen=True)
class PreparedDecision:
    """In-memory shared input; only summary is safe for evidence reports."""

    messages: list[dict[str, Any]]
    prompt: str
    inputs: Any
    answer_token_ids: dict[str, int]
    summary: dict[str, Any]


def prepare_case(
    case: PreparationCase,
    processor: Any,
    root: Path,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    max_audio_seconds: float = 30,
    max_video_seconds: float = 60,
) -> PreparedDecision:
    """Build official CPU tensors and a separate content-free evidence summary."""
    num_frames = case.video_frames
    if max_bytes <= 0 or max_audio_seconds <= 0 or max_video_seconds <= 0:
        raise ValueError("invalid preparation limits")
    media: dict[str, list[Any]] = {"image": [], "audio": [], "video": []}
    content: list[dict[str, Any]] = []
    for item in case.prefix.media:
        path = resolve_media(root, item.uri, item.sha256, max_bytes=max_bytes)
        media[item.type].append(
            load_media(
                path,
                item.type,
                num_frames=num_frames,
                max_seconds=max_audio_seconds if item.type == "audio" else max_video_seconds,
            )
        )
        content.append({"type": item.type})
    payload = render_payload(Prefix(state=case.prefix.state), case.question)
    content.append({"type": "text", "text": payload})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt or "<|think|>" in prompt:
        raise ValueError("invalid official prompt or thinking not disabled")
    tokenizer = processor.tokenizer
    mapping = check_answer_tokens(tokenizer, case.question)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for answer, token_id in mapping.items():
        if tokenizer.encode(prompt + answer, add_special_tokens=False) != prompt_ids + [token_id]:
            raise ValueError("answer generation boundary changed")
        if token_id in tokenizer.all_special_ids or token_id == tokenizer.unk_token_id:
            raise ValueError("answer is a special or unknown token")
        if tokenizer.decode([token_id], skip_special_tokens=False) != answer:
            raise ValueError("answer roundtrip mismatch")
    kwargs: dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
        "return_mm_token_type_ids": True,
        "truncation": False,
    }
    if media["image"]:
        kwargs["images"] = [media["image"]]
    if media["audio"]:
        kwargs["audio"] = media["audio"]
        kwargs["sampling_rate"] = 16000
    if media["video"]:
        kwargs.update(
            videos=[video["frames"] for video in media["video"]],
            video_metadata=[video["metadata"] for video in media["video"]],
            do_sample_frames=False,
        )
    inputs = processor(**kwargs)
    required = {"input_ids", "attention_mask", "mm_token_type_ids"}
    if media["image"]:
        required.update({"pixel_values", "image_position_ids"})
    if media["audio"]:
        required.update({"input_features", "input_features_mask"})
    if media["video"]:
        required.update({"pixel_values_videos", "video_position_ids", "num_frames_per_video"})
    if not required.issubset(inputs):
        raise ValueError("official processor omitted required modality tensors")
    final_ids = inputs["input_ids"][0].tolist()
    # Media expansion may change the prefix, but cannot change the generation suffix.
    suffix_size = min(8, len(prompt_ids))
    if not suffix_size or final_ids[-suffix_size:] != prompt_ids[-suffix_size:]:
        raise ValueError("processor changed generation boundary suffix")
    summaries = {}
    for key, value in inputs.items():
        if not hasattr(value, "detach"):
            raise ValueError("processor returned a non-tensor output")
        array = value.detach().cpu().numpy()
        if array.dtype.kind == "f" and not importlib.import_module("numpy").isfinite(array).all():
            raise ValueError("processor returned nonfinite tensor")
        summaries[key] = {
            "shape": list(array.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    summary = {
        "id": case.id,
        "provenance": case.meta.model_dump(),
        "configuration": {
            "max_bytes": max_bytes,
            "max_audio_seconds": max_audio_seconds,
            "max_video_seconds": max_video_seconds,
            "video_frames": num_frames,
            "truncation": False,
            "enable_thinking": False,
        },
        "category": case.category,
        "primitive": case.question.type,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "media_sha256": [item.sha256 for item in case.prefix.media],
        "video_metadata": [video["metadata"] for video in media["video"]],
        "decoded_video_frame_sha256": [
            [hashlib.sha256(frame.tobytes()).hexdigest() for frame in video["frames"]]
            for video in media["video"]
        ],
        "answer_count_including_u": len(mapping),
        "answer_token_mapping_sha256": hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode()
        ).hexdigest(),
        "generation_boundary": "passed",
        "tensors": summaries,
    }

    return PreparedDecision(messages, prompt, inputs, mapping, summary)
