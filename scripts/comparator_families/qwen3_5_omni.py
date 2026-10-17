"""qwen3_5 comparator family: OneJev-4B/27B audio path for comparator_eval.

SMOKE-GATED: the exact processor/model classes are validated by a --limit
smoke on the box (transformers 5.17) before any scored run. The audio path
follows the Qwen-Omni processor convention: chat template over a user turn
holding one audio clip plus the canonical decision payload text, then the
processor produces the forward inputs. Video raises until its path is
written and smoke-validated.
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
            raise DevEvalError(f"{sample.id}: qwen3_5_omni family handles media samples only")
        if media[0].type == "video":
            raise DevEvalError(f"{sample.id}: qwen3_5_omni video path not smoke-validated yet")
        if media[0].type != "audio":
            raise DevEvalError(f"{sample.id}: unsupported media type {media[0].type!r}")
        path = _resolve_media(media_root, media[0].uri, media[0].sha256)

        soundfile = importlib.import_module("soundfile")
        audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)

        processor = loaded.processor
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": path.name},
                    {"type": "text", "text": _payload_text(sample)},
                ],
            }
        ]
        try:
            rendered = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=[rendered], audio=[audio], sampling_rate=sample_rate, return_tensors="pt"
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
        )
