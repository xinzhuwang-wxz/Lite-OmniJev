"""Bounded local media preparation shared by offline decision callers."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import subprocess
import tempfile
import wave
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


# Exclude playlists, concat and image sequences before probing can follow references.
# MOV external-track loading stays disabled (FFmpeg's enable_drefs default).
_DECODE_INPUT_OPTIONS = [
    "-protocol_whitelist",
    "file,pipe",
    "-format_whitelist",
    "wav,flac,mp3,ogg,aac,mov,matroska,webm,avi,mpegts",
]


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
            *_DECODE_INPUT_OPTIONS,
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
        stdin=subprocess.DEVNULL,
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
                *_DECODE_INPUT_OPTIONS,
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
            stdin=subprocess.DEVNULL,
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
                *_DECODE_INPUT_OPTIONS,
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
            stdin=subprocess.DEVNULL,
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
class CanonicalMedia:
    """HTTP bytes and the HF values redecoded from those exact bytes, kept in memory."""

    hf_value: Any
    http_part: dict[str, Any]
    summary: dict[str, Any]
    video_io_kwargs: dict[str, Any] | None = None


def canonicalize_media(kind: str, value: Any) -> CanonicalMedia:
    image_module = importlib.import_module("PIL.Image") if kind != "audio" else None
    if kind == "image":
        assert image_module is not None
        buffer = io.BytesIO()
        value.save(buffer, format="PNG", optimize=False, compress_level=6)
        data = buffer.getvalue()
        with image_module.open(io.BytesIO(data)) as image:
            decoded = image.convert("RGB")
        return CanonicalMedia(
            decoded,
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()},
            },
            {
                "type": "image",
                "encoding": "RGB PNG compress_level=6 optimize=false",
                "source_rgb_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "canonical_bytes_sha256": hashlib.sha256(data).hexdigest(),
                "canonical_rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            },
        )
    if kind == "audio":
        np = importlib.import_module("numpy")
        pcm = np.rint(np.clip(value, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(pcm)
        data = buffer.getvalue()
        with wave.open(io.BytesIO(data), "rb") as audio:
            decoded_pcm = audio.readframes(audio.getnframes())
        decoded = np.frombuffer(decoded_pcm, dtype="<i2").astype(np.float32) / 32768.0
        return CanonicalMedia(
            decoded,
            {
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"},
            },
            {
                "type": "audio",
                "encoding": "mono 16000Hz PCM16 WAV; round(clip(float,-1,1)*32767)",
                "sample_rate": 16000,
                "frames": len(decoded),
                "source_float32_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "canonical_bytes_sha256": hashlib.sha256(data).hexdigest(),
                "canonical_pcm16_sha256": hashlib.sha256(decoded_pcm).hexdigest(),
                "canonical_float32_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            },
        )
    if kind != "video":
        raise ValueError("unsupported canonical media type")
    assert image_module is not None
    frames, encoded, source_hashes, decoded_hashes = [], [], [], []
    for frame in value["frames"]:
        buffer = io.BytesIO()
        frame.save(
            buffer, format="JPEG", quality=95, subsampling=0, optimize=False, progressive=False
        )
        data = buffer.getvalue()
        with image_module.open(io.BytesIO(data)) as image:
            decoded = image.convert("RGB")
        frames.append(decoded)
        encoded.append(data)
        source_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        decoded_hashes.append(hashlib.sha256(decoded.tobytes()).hexdigest())
    metadata = {**value["metadata"], "video_backend": "canonical_jpeg"}
    io_kwargs = {
        key: metadata[key] for key in ("fps", "frames_indices", "total_num_frames", "duration")
    }
    io_kwargs["do_sample_frames"] = False
    return CanonicalMedia(
        {"frames": frames, "metadata": metadata},
        {
            "type": "video_url",
            "video_url": {
                "url": "data:video/jpeg;base64,"
                + ",".join(base64.b64encode(data).decode() for data in encoded)
            },
        },
        {
            "type": "video",
            "encoding": "RGB JPEG quality=95 subsampling=0 optimize=false progressive=false",
            "source_rgb_sha256": source_hashes,
            "canonical_frame_bytes_sha256": [hashlib.sha256(data).hexdigest() for data in encoded],
            "canonical_rgb_sha256": decoded_hashes,
            "metadata": io_kwargs,
        },
        io_kwargs,
    )


@dataclass(frozen=True)
class PreparedDecision:
    """Shared in-memory input; only summary is safe to write as evidence."""

    messages: list[dict[str, Any]]
    prompt: str
    inputs: Any
    answer_token_ids: dict[str, int]
    summary: dict[str, Any]
    http_messages: list[dict[str, Any]] | None = None
    http_media_io_kwargs: dict[str, Any] | None = None


def native_processor_kwargs(media_io_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Keep native video frames on the pinned checkpoint's official video normalization.

    vLLM 0.26 processes frames through its image processor, whose default is False;
    the frozen Gemma4 video processor uses True with identity mean/std.
    """
    return {"images_kwargs": {"do_normalize": True}} if (media_io_kwargs or {}).get("video") else {}


def prepare_decision(
    prefix: Prefix,
    question: Question,
    processor: Any,
    root: Path,
    *,
    profile: Literal["hf_only", "canonical_http"] = "hf_only",
    video_frames: Literal[8, 16] = 16,
    max_bytes: int = 64 * 1024 * 1024,
    max_total_image_pixels: int = 40_000_000,
    max_audio_seconds: float = 30,
    max_video_seconds: float = 60,
) -> PreparedDecision:
    """Prepare inference inputs without inventing any dataset provenance metadata."""
    num_frames = video_frames
    if profile not in {"hf_only", "canonical_http"} or num_frames not in {8, 16}:
        raise ValueError("invalid preparation profile or video frame count")
    if not prefix.state.strip() or not question.instruction.strip():
        raise ValueError("empty decision input")
    if (
        max_bytes <= 0
        or max_audio_seconds <= 0
        or max_video_seconds <= 0
        or max_total_image_pixels <= 0
    ):
        raise ValueError("invalid preparation limits")
    if profile == "canonical_http":
        types = [item.type for item in prefix.media]
        if types.count("video") > 1 or types.count("audio") > 1:
            raise ValueError("canonical S0 profile supports at most one video and one audio")
        if "video" in types and "image" in types:
            raise ValueError("image plus video canonical profile CPU compatibility remains pending")
    if len(prefix.media) > 8:
        raise ValueError("media count exceeds local resource limit")
    # Hash/path checks and cheap image headers precede every full decode.
    paths = [
        resolve_media(root, item.uri, item.sha256, max_bytes=max_bytes) for item in prefix.media
    ]
    total_image_pixels = 0
    for item, path in zip(prefix.media, paths, strict=True):
        if item.type == "image":
            with importlib.import_module("PIL.Image").open(path) as source:
                total_image_pixels += source.width * source.height
                if total_image_pixels > max_total_image_pixels:
                    raise ValueError("total image pixel budget exceeded")
    media: dict[str, list[Any]] = {"image": [], "audio": [], "video": []}
    content: list[dict[str, Any]] = []
    http_content: list[dict[str, Any]] = []
    canonical_summaries = []
    http_media_io_kwargs: dict[str, Any] = {}
    for item, path in zip(prefix.media, paths, strict=True):
        decoded = load_media(
            path,
            item.type,
            num_frames=num_frames,
            max_seconds=max_audio_seconds if item.type == "audio" else max_video_seconds,
        )
        if profile == "canonical_http":
            canonical = canonicalize_media(item.type, decoded)
            decoded = canonical.hf_value
            http_content.append(canonical.http_part)
            canonical_summaries.append(canonical.summary)
            if canonical.video_io_kwargs is not None:
                http_media_io_kwargs["video"] = canonical.video_io_kwargs
        media[item.type].append(decoded)
        content.append({"type": item.type})
    payload = render_payload(Prefix(state=prefix.state), question)
    content.append({"type": "text", "text": payload})
    http_content.append({"type": "text", "text": payload})
    messages = [{"role": "user", "content": content}]
    http_messages = (
        [{"role": "user", "content": http_content}] if profile == "canonical_http" else None
    )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt or "<|think|>" in prompt:
        raise ValueError("invalid official prompt or thinking not disabled")
    tokenizer = processor.tokenizer
    mapping = check_answer_tokens(tokenizer, question)
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
    if profile == "canonical_http":
        kwargs["add_special_tokens"] = False
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
        # Transformers 5.17 declares these two video inputs. Later versions also
        # return num_frames_per_video; preserve it when present, never fabricate it.
        required.update({"pixel_values_videos", "video_position_ids"})
    if not required.issubset(inputs):
        raise ValueError("official processor omitted required modality tensors")
    final_ids = inputs["input_ids"][0].tolist()
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
    summary: dict[str, Any] = {
        "input_profile": "canonical-http-v1" if profile == "canonical_http" else "hf-only-v1",
        "configuration": {
            "max_bytes": max_bytes,
            "max_media_items": 8,
            "max_total_image_pixels": max_total_image_pixels,
            "max_audio_seconds": max_audio_seconds,
            "max_video_seconds": max_video_seconds,
            "video_frames": num_frames,
            "truncation": False,
            "enable_thinking": False,
        },
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "media_sha256": [item.sha256 for item in prefix.media],
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
    if http_messages is not None:
        summary["configuration"]["add_special_tokens"] = False
        summary.update(
            canonical_media=canonical_summaries,
            http_messages_sha256=hashlib.sha256(
                json.dumps(http_messages, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            http_media_io_kwargs_sha256=hashlib.sha256(
                json.dumps(http_media_io_kwargs, sort_keys=True).encode()
            ).hexdigest(),
            http_mm_processor_kwargs_sha256=hashlib.sha256(
                json.dumps(native_processor_kwargs(http_media_io_kwargs), sort_keys=True).encode()
            ).hexdigest(),
        )
    return PreparedDecision(
        messages,
        prompt,
        inputs,
        mapping,
        summary,
        http_messages,
        http_media_io_kwargs if http_messages is not None else None,
    )


def prepare_case(
    case: PreparationCase,
    processor: Any,
    root: Path,
    *,
    profile: Literal["hf_only", "canonical_http"] = "hf_only",
    max_bytes: int = 64 * 1024 * 1024,
    max_audio_seconds: float = 30,
    max_video_seconds: float = 60,
) -> PreparedDecision:
    prepared = prepare_decision(
        case.prefix,
        case.question,
        processor,
        root,
        profile=profile,
        video_frames=case.video_frames,
        max_bytes=max_bytes,
        max_audio_seconds=max_audio_seconds,
        max_video_seconds=max_video_seconds,
    )
    prepared.summary.update(
        id=case.id,
        category=case.category,
        primitive=case.question.type,
        provenance=case.meta.model_dump(),
    )
    return prepared
