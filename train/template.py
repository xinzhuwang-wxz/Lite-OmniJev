"""One payload and the tokenizer's own chat template for training and inference."""

import json
from typing import Any, Protocol

from data.schema import Prefix, Question, answer_tokens


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]: ...
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def render_payload(prefix: Prefix, question: Question) -> str:
    if prefix.media:
        raise ValueError(
            "media requires an official multimodal processor; text-only helper refuses it"
        )
    tokens = answer_tokens(question)
    labels = (
        question.options
        if question.type == "choice"
        else question.score_descriptions or list(tokens)
    )
    return json.dumps(
        {
            "state": prefix.state,
            "question": question.instruction,
            "answers": dict(zip(tokens, labels or [], strict=True)),
            "U": "Insufficient evidence to decide",
            "response": "Return exactly one answer token, or U if evidence is insufficient.",
        },
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def render_prompt(tokenizer: ChatTokenizer, prefix: Prefix, question: Question) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": render_payload(prefix, question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("official chat template must return a nonempty string")
    return rendered


def check_answer_tokens(tokenizer: ChatTokenizer, question: Question) -> dict[str, int]:
    """Preflight only; real-model gate must also verify IDs at the generation boundary."""
    result = {}
    for token in (*answer_tokens(question), "U"):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer {token!r} must encode to exactly one token")
        result[token] = ids[0]
    if len(set(result.values())) != len(result):
        raise ValueError("answer token IDs must be distinct")
    return result
