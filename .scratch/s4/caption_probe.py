"""B2 perception probe: frozen base generates structured video summaries (CPU-prepared).

The generation loop mirrors the proven loaders in scripts/s0_development_baselines.py
(same offline flags, official chat template, thinking disabled). The decision layer
stays untouched: this probe only produces text summaries that later condition the
existing single-token decision readout. CPU part (prompt builder, JSON contract,
parser, self-test) runs anywhere; the forward pass requires the model dir and is
scheduled by main on the GPU window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PERCEPTION_PROMPT = (
    "Watch the video and answer with a JSON object only, no other text.\n"
    "Schema:\n"
    '{"objects": [{"name": "<color> <material> <shape>", "first_seen": '
    '"start|early|middle|late|end", "last_seen": "start|early|middle|late|end"}],\n'
    ' "entry_order": ["<name> before <name>", ...],\n'
    ' "collisions": [{"between": ["<name>", "<name>"], "when": '
    '"early|middle|late"}],\n'
    ' "present_at_start": ["<name>", ...], "present_at_end": ["<name>", ...]}\n'
    "Use only objects you actually see. If unsure about ordering, omit it from "
    "entry_order rather than guessing."
)
PHASES = ("start", "early", "middle", "late", "end")


def build_prompt(state: str) -> str:
    """Perception prompt reuses the official prefix state of the source item."""
    return f"{state}\n{PERCEPTION_PROMPT}"


def validate_summary(text: str) -> dict | None:
    """Parse and structurally validate a generated summary; None on any violation."""
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        summary = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(summary, dict):
        return None
    objects = summary.get("objects")
    if not isinstance(objects, list) or not objects:
        return None
    names = set()
    for entry in objects:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            return None
        for phase_key in ("first_seen", "last_seen"):
            if entry.get(phase_key) not in PHASES:
                return None
        if entry["name"] in names:
            return None
        names.add(entry["name"])
    for key in ("present_at_start", "present_at_end"):
        values = summary.get(key)
        if not isinstance(values, list) or any(v not in names for v in values):
            return None
    order = summary.get("entry_order")
    if not isinstance(order, list):
        return None
    for edge in order:
        if not isinstance(edge, str) or " before " not in edge:
            return None
        left, right = edge.split(" before ", 1)
        if left not in names or right not in names or left == right:
            return None
    collisions = summary.get("collisions")
    if not isinstance(collisions, list):
        return None
    for hit in collisions:
        if (
            not isinstance(hit, dict)
            or not isinstance(hit.get("between"), list)
            or len(hit["between"]) != 2
            or any(name not in names for name in hit["between"])
            or hit.get("when") not in ("early", "middle", "late")
        ):
            return None
    return summary


def self_test() -> None:
    good = (
        '{"objects": [{"name": "red metal cube", "first_seen": "start", '
        '"last_seen": "end"}], "entry_order": [], "collisions": [], '
        '"present_at_start": ["red metal cube"], "present_at_end": ["red metal cube"]}'
    )
    assert validate_summary("noise " + good + " trailing") is not None
    assert validate_summary('{"objects": []}') is None
    assert validate_summary("no json here") is None
    bad_phase = good.replace('"last_seen": "end"', '"last_seen": "sometime"')
    assert validate_summary(bad_phase) is None
    bad_order = good.replace('"entry_order": []', '"entry_order": ["ghost before red metal cube"]')
    assert validate_summary(bad_order) is None
    prompt = build_prompt("Answer the yes/no question using the attached video.")
    assert prompt.startswith("Answer the yes/no question") and "JSON" in prompt
    print(json.dumps({"self_test": "passed", "prompt_tokens_heuristic": len(prompt.split())}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data", type=Path, help="source jsonl with video rows")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not (args.model_dir and args.data and args.media_root and args.output):
        parser.error("generation requires --model-dir --data --media-root --output")
    import torch
    from transformers import AutoModelForImageTextToText

    from scripts import s0_development_baselines as baseline

    samples = [
        sample
        for sample in baseline.load_samples(args.data)
        if sample.meta.modality == ["video"]
    ][: args.limit]
    processor = baseline.default_processor_factory(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for sample in samples:
            # Media decoding identical to the proven readout path; only the final
            # text payload is swapped for the perception prompt.
            messages, kwargs, identity = baseline.build_messages_and_media(
                sample, sample.questions[0], args.media_root
            )
            messages[0]["content"][-1] = {"type": "text", "text": PERCEPTION_PROMPT}
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            kwargs["text"] = [prompt]
            inputs = processor(**kwargs)
            inputs = baseline.move_inputs_to_cuda(inputs, torch, torch.bfloat16)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False
                )
            text = processor.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            summary = validate_summary(text)
            stream.write(
                json.dumps(
                    {
                        "id": sample.id,
                        "family_id": sample.meta.family_id,
                        "media_sha256": sample.prefix.media[0].sha256,
                        "prompt_sha256": baseline.sha256_bytes(prompt.encode()),
                        "raw_text_sha256": baseline.sha256_bytes(text.encode()),
                        "summary_valid": summary is not None,
                        "summary": summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
