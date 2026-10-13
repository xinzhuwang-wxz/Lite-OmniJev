"""CPU-only probe of the downloaded official tokenizer and chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.assets import write_json

ANSWERS = list("ABCDEFGHIJKLMNOPQRSTU") + list("0123456789") + ["yes", "no"]


def render_chat(
    template_source: str,
    tokenizer_config: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Render the pinned official text template without importing remote Python code."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def reject(message: str) -> None:
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = reject
    template = environment.from_string(template_source)
    return template.render(
        **{
            **tokenizer_config,
            "messages": messages,
            "tools": None,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
    )


class OfficialTokenizerAdapter:
    """Shared local text-only adapter for train, serving and the CPU probe."""

    def __init__(self, model_dir: Path) -> None:
        from tokenizers import Tokenizer

        self.tokenizer_path = model_dir / "tokenizer.json"
        config_path = model_dir / "tokenizer_config.json"
        template_path = model_dir / "chat_template.jinja"
        for path in (self.tokenizer_path, config_path):
            if not path.is_file():
                raise ValueError(f"Missing required file: {path.name}")
        self.config = json.loads(config_path.read_text())
        template = (
            template_path.read_text()
            if template_path.is_file()
            else self.config.get("chat_template")
        )
        if not isinstance(template, str) or not template:
            raise ValueError("Missing official chat template")
        self.template = template
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        rendered = render_chat(
            self.template,
            self.config,
            messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )
        return self.encode(rendered) if tokenize else rendered


def probe(model_dir: Path) -> dict[str, Any]:
    adapter = OfficialTokenizerAdapter(model_dir)
    tokenizer_path, config, template = adapter.tokenizer_path, adapter.config, adapter.template
    messages = [
        {
            "role": "user",
            "content": (
                "Read the evidence. Reply with exactly one decision token.\nQuestion: Is 2+2=4?"
            ),
        }
    ]
    prompt = str(adapter.apply_chat_template(messages, enable_thinking=False))
    thinking_prompt = str(adapter.apply_chat_template(messages, enable_thinking=True))
    thinking_token = config.get("think_token", "<|think|>")
    if not thinking_token or thinking_token in prompt or thinking_token not in thinking_prompt:
        raise ValueError("Cannot prove explicit thinking disablement with this template")
    prompt_ids = adapter.encode(prompt)
    if not prompt_ids:
        raise ValueError("Empty generation prompt")
    unknown_id = adapter.tokenizer.token_to_id(config.get("unk_token", "<unk>"))
    special_ids = {
        token_id
        for token_id, token in adapter.tokenizer.get_added_tokens_decoder().items()
        if token.special
    }
    mapping: dict[str, int] = {}
    failures = []
    for answer in ANSWERS:
        ids = adapter.encode(answer)
        if len(ids) != 1:
            failures.append(f"{answer}: standalone token count {len(ids)}")
            continue
        if ids[0] == unknown_id or ids[0] in special_ids:
            failures.append(f"{answer}: resolves to unknown or special token")
        if adapter.tokenizer.decode(ids, skip_special_tokens=False) != answer:
            failures.append(f"{answer}: token does not decode to the literal answer")
        if ids[0] in mapping.values():
            failures.append(f"{answer}: token ID collision")
        mapping[answer] = ids[0]
        continuation = adapter.encode(prompt + answer)
        if continuation != prompt_ids + ids:
            failures.append(f"{answer}: generation boundary changes answer or prefix tokens")
    second_messages = [
        {
            "role": "user",
            "content": (
                "Read the evidence. Reply with exactly one decision token.\nQuestion: Is 2+2=5?"
            ),
        }
    ]
    second = str(adapter.apply_chat_template(second_messages, enable_thinking=False))
    second_ids = adapter.encode(second)
    common = 0
    for left, right in zip(prompt_ids, second_ids, strict=False):
        if left != right:
            break
        common += 1
    if common == 0:
        failures.append("No shared token prefix across related requests")
    return {
        "status": "failed" if failures else "passed",
        "exit_code": 1 if failures else 0,
        "model_dir": str(model_dir.resolve()),
        "answer_token_ids": mapping,
        "answer_count": len(ANSWERS),
        "failures": failures,
        "thinking_disabled": True,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "shared_prefix_tokens": common,
        "prompt_tokens": len(prompt_ids),
        "rendered_prompt": prompt,
        "scope": (
            "CPU text tokenizer/template only; "
            "no model logits, multimodal processor, or GPU parity claim"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/token-probe.json"))
    args = parser.parse_args(argv)
    try:
        report = probe(args.model_dir)
    except Exception as exc:
        report = {
            "status": "blocked",
            "exit_code": 1,
            "blocker": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
        }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
