"""Strict native chat raw-logprob client; service attestation remains the caller's job."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from train.multimodal import PreparedDecision


class VllmChatProtocolError(ValueError):
    """A static error code without original input, media or backend response content."""


@dataclass(frozen=True)
class ChatLogprobs:
    raw_logprobs: dict[str, float]
    prompt_token_ids: list[int]
    usage: dict[str, int]
    summary: dict[str, Any]


def _token_id(token: Any) -> int:
    if not isinstance(token, str) or re.fullmatch(r"token_id:(0|[1-9][0-9]*)", token) is None:
        raise VllmChatProtocolError("invalid_token_id")
    return int(token.split(":", 1)[1])


def _logprob(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not -9999 < value <= 0:
        raise VllmChatProtocolError("invalid_raw_logprob")
    return float(value)


def parse_chat_logprobs(
    body: Any, *, model: str, answer_token_ids: dict[str, int], expected_prompt_ids: list[int]
) -> ChatLogprobs:
    """Only the complete requested raw-token set, after exact expanded-input equality."""
    try:
        if not isinstance(body, dict) or body.get("model") != model:
            raise VllmChatProtocolError("model_identity_mismatch")
        prompt_ids = body["prompt_token_ids"]
        if (
            not isinstance(prompt_ids, list)
            or not prompt_ids
            or any(type(token) is not int or token < 0 for token in prompt_ids)
            or prompt_ids != expected_prompt_ids
        ):
            raise VllmChatProtocolError("expanded_prompt_ids_mismatch")
        usage = body["usage"]
        counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if (
            any(type(value) is not int for value in counts.values())
            or counts["prompt_tokens"] != len(prompt_ids)
            or counts["completion_tokens"] != 1
            or counts["total_tokens"] != len(prompt_ids) + 1
        ):
            raise VllmChatProtocolError("usage_mismatch")
        choices = body["choices"]
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or type(choices[0]["index"]) is not int
            or choices[0]["index"] != 0
        ):
            raise VllmChatProtocolError("expected_one_choice")
        choice = choices[0]
        if choice.get("finish_reason") not in {"stop", "length"}:
            raise VllmChatProtocolError("invalid_finish_reason")
        positions = choice["logprobs"]["content"]
        if not isinstance(positions, list) or len(positions) != 1:
            raise VllmChatProtocolError("expected_one_generation_position")
        sampled_id = _token_id(positions[0]["token"])
        _logprob(positions[0]["logprob"])
        entries = positions[0]["top_logprobs"]
        if (
            not isinstance(entries, list)
            or not len(answer_token_ids) <= len(entries) <= len(answer_token_ids) + 1
        ):
            raise VllmChatProtocolError("incomplete_requested_logprobs")
        by_id: dict[int, float] = {}
        for entry in entries:
            token_id = _token_id(entry["token"])
            if token_id in by_id:
                raise VllmChatProtocolError("duplicate_token_id")
            by_id[token_id] = _logprob(entry["logprob"])
        if not (set(by_id) - set(answer_token_ids.values())).issubset({sampled_id}):
            raise VllmChatProtocolError("unexpected_unsolicited_token")
        if not set(answer_token_ids.values()).issubset(by_id):
            raise VllmChatProtocolError("missing_requested_token")
        raw = {answer: by_id[token_id] for answer, token_id in answer_token_ids.items()}
    except VllmChatProtocolError:
        raise
    except (KeyError, TypeError, IndexError, OverflowError, ValueError):
        raise VllmChatProtocolError("invalid_chat_response_schema") from None
    summary = {
        "model": model,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "specified_answer_tokens_including_u": len(raw),
        "expanded_prompt_ids_match_hf": True,
        "raw_logprobs_source": "caller-attested raw_logprobs service; exact requested IDs",
    }
    return ChatLogprobs(raw, prompt_ids, counts, summary)


class VllmChatClient:
    """Caller verifies the pinned launch record before setting raw_logprobs_confirmed.

    Matching the response model alias checks the server declaration, not its weights.
    No bool or response field can independently attest the actual service configuration.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        raw_logprobs_confirmed: bool,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30,
    ) -> None:
        if raw_logprobs_confirmed is not True:
            raise VllmChatProtocolError("raw_logprobs_launch_record_not_confirmed")
        if not model.strip():
            raise VllmChatProtocolError("empty_model_identity")
        self.model = model
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.client = httpx.AsyncClient(
            timeout=timeout, transport=transport, follow_redirects=False
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def read_logprobs(self, prepared: PreparedDecision) -> ChatLogprobs:
        if (
            prepared.http_messages is None
            or prepared.summary.get("input_profile") != "canonical-http-v1"
        ):
            raise VllmChatProtocolError("canonical_http_profile_required")
        messages_hash = hashlib.sha256(
            json.dumps(prepared.http_messages, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        media_kwargs_hash = hashlib.sha256(
            json.dumps(prepared.http_media_io_kwargs or {}, sort_keys=True).encode()
        ).hexdigest()
        if messages_hash != prepared.summary.get(
            "http_messages_sha256"
        ) or media_kwargs_hash != prepared.summary.get("http_media_io_kwargs_sha256"):
            raise VllmChatProtocolError("canonical_http_binding_mismatch")
        ids = prepared.answer_token_ids
        if (
            "U" not in ids
            or len(ids) < 3
            or len(set(ids.values())) != len(ids)
            or any(type(token) is not int or token < 0 for token in ids.values())
        ):
            raise VllmChatProtocolError("invalid_answer_token_mapping")
        if len(prepared.inputs["input_ids"]) != 1:
            raise VllmChatProtocolError("expected_one_hf_input")
        expected_prompt_ids = prepared.inputs["input_ids"][0].tolist()
        if not expected_prompt_ids or any(
            type(token) is not int or token < 0 for token in expected_prompt_ids
        ):
            raise VllmChatProtocolError("invalid_hf_prompt_ids")
        payload = {
            "model": self.model,
            "messages": prepared.http_messages,
            "max_completion_tokens": 1,
            "stream": False,
            "n": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": 42,
            "logprobs": True,
            "top_logprobs": 0,
            "logprob_token_ids": list(ids.values()),
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if prepared.http_media_io_kwargs:
            payload["media_io_kwargs"] = prepared.http_media_io_kwargs
        try:
            response = await self.client.post(self.url, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException:
            raise VllmChatProtocolError("upstream_timeout") from None
        except (httpx.HTTPError, ValueError):
            raise VllmChatProtocolError("upstream_request_failed") from None
        return parse_chat_logprobs(
            body,
            model=self.model,
            answer_token_ids=ids,
            expected_prompt_ids=expected_prompt_ids,
        )
