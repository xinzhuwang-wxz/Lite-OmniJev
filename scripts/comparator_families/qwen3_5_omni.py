"""qwen3_5 comparator family (Qwen3-VL): OneJev-4B/27B video path.

SMOKE-GATED: validated by a --limit smoke on the box before scored runs.
These checkpoints are vision-language only (Qwen3VLProcessor; the chat
template accepts image/text/video and has no audio branch), so MMAU is
structurally unrunnable for them and MVBench is their native board. The
canonical 16-frame@4fps mp4 is passed as a video path with fps=4 so the
processor samples exactly the 16 canonical frames. Audio raises: those
comparators cannot consume the audio board.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.comparator_eval import DevEvalError, Loaded, PreparedRow
from train.template import payload_dict

_MEDIA_BY_SHA_MAX_BYTES = 64 * 1024 * 1024


def build() -> Any:
    return Qwen35OmniAdapter()


def _resolve_media(media_root: Path, uri: str, sha256: str) -> Path:
    path = media_root / uri
    if not path.is_file():
        raise DevEvalError(f"media file missing: {path}")
    if path.stat().st_size > _MEDIA_BY_SHA_MAX_BYTES:
        raise DevEvalError(f"media file over byte limit: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise DevEvalError(f"media sha mismatch: {path}")
    return path


def _payload_text(sample: Sample) -> str:
    return json.dumps(
        payload_dict(sample.prefix, sample.questions[0]),
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    )


class Qwen35OmniAdapter:
    def load(self, model_dir: Path) -> Loaded:
        transformers = importlib.import_module("transformers")
        processor = transformers.AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=False, local_files_only=True
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True
            )
        config = transformers.AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        architectures = getattr(config, "architectures", None) or []
        model = None
        for name in architectures:
            candidate = getattr(transformers, name, None)
            if candidate is not None:
                model = candidate.from_pretrained(
                    str(model_dir), torch_dtype="auto", local_files_only=True
                )
                break
        if model is None:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(model_dir), torch_dtype="auto", local_files_only=True
            )
        return Loaded(model=model.eval().cuda(), tokenizer=tokenizer, processor=processor)

    def prepare(
        self, loaded: Loaded, sample: Sample, media_root: Path, *, max_length: int
    ) -> PreparedRow:
        media = sample.prefix.media
        if not media:
            raise DevEvalError(f"{sample.id}: qwen3_5 family handles media samples only")
        if media[0].type != "video":
            raise DevEvalError(
                f"{sample.id}: qwen3_5 checkpoints have no audio/image tower; "
                f"{media[0].type} samples are structurally unrunnable"
            )
        path = _resolve_media(media_root, media[0].uri, media[0].sha256)

        processor = loaded.processor
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(path)},
                    {"type": "text", "text": _payload_text(sample)},
                ],
            }
        ]
        try:
            rendered = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            inputs = processor(
                text=[rendered],
                videos=[[str(path)]],
                fps=4.0,
                return_tensors="pt",
            )
        except Exception as exc:  # noqa: BLE001 - processor API varies across versions
            raise DevEvalError(
                f"{sample.id}: qwen processor call failed ({exc}); "
                "adjust the family path after a smoke inspection"
            ) from exc
        input_ids = inputs["input_ids"][0].tolist()
        if not input_ids or len(input_ids) > max_length:
            raise DevEvalError(f"{sample.id}: empty or over-length prompt ({len(input_ids)})")
        forward_kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
        return PreparedRow(
            input_ids=input_ids,
            prompt_text=rendered,
            forward_kwargs=forward_kwargs,
            boundary_tail="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
