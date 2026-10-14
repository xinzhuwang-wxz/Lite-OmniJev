"""Freeze ten non-quality probes from one public image family with the official CPU processor."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from eval.prefix_cache import CacheCase, CacheProtocol, digest
from scripts.s0_prepare import read_manifest
from train.multimodal import PreparationCase, prepare_case


def _common_ids(rows: list[list[int]]) -> list[int]:
    common = []
    for tokens in zip(*rows, strict=False):
        if len(set(tokens)) != 1:
            break
        common.append(tokens[0])
    return common


def freeze(
    cases: list[PreparationCase],
    processor: Any,
    asset_root: Path,
    *,
    repeats: int = 5,
    warmup_requests: int = 2,
) -> tuple[CacheProtocol, dict[str, Any], dict[str, Any]]:
    """All synthetic variants stay in the selected source family and outside quality gates."""
    original = next(
        (
            case
            for case in cases
            if case.category == "image"
            and len(case.prefix.media) == 1
            and case.prefix.media[0].type == "image"
        ),
        None,
    )
    if original is None:
        raise ValueError("public single-image source required")
    prepared = []
    probes = []
    for index in range(10):
        row = original.model_dump()
        row["id"] = original.id + f"-cache-probe-{index}"
        row["question"]["qid"] = f"cache-probe-{index}"
        row["question"]["instruction"] += (
            f"\nNon-quality prefix-cache protocol probe {index}; answer the original question."
        )
        probe = PreparationCase.model_validate(row)
        probes.append(probe)
        prepared.append(prepare_case(probe, processor, asset_root, profile="canonical_http"))
    messages = [item.http_messages for item in prepared]
    if any(message is None or len(message) != 1 for message in messages):
        raise ValueError("canonical single-message inputs required")
    contents = [message[0]["content"] for message in messages if message is not None]
    if any(
        len(content) != 2
        or content[0].get("type") != "image_url"
        or content[1].get("type") != "text"
        for content in contents
    ):
        raise ValueError("exact canonical image plus text shape required")
    image_parts = [content[0] for content in contents]
    if any(part != image_parts[0] for part in image_parts):
        raise ValueError("all cases must share identical canonical image bytes")
    mappings = [item.answer_token_ids for item in prepared]
    if any(mapping != mappings[0] for mapping in mappings):
        raise ValueError("all probes must share the legal answer mapping")
    url = image_parts[0]["image_url"]["url"]
    marker = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(marker):
        raise ValueError("canonical lossless PNG transport required")
    encoded = url[len(marker) :]
    image_bytes = base64.b64decode(encoded, validate=True)
    texts = [content[1]["text"] for content in contents]
    common_text = os.path.commonprefix(texts)
    expanded = [item.inputs["input_ids"][0].tolist() for item in prepared]
    common_ids = _common_ids(expanded)
    media_ends = []
    for item, ids in zip(prepared, expanded, strict=True):
        types = item.inputs["mm_token_type_ids"][0].tolist()
        if len(types) != len(ids) or not any(types):
            raise ValueError("actual expanded multimodal token positions required")
        media_ends.append(max(index for index, token in enumerate(types) if token != 0) + 1)
    if len(set(media_ends)) != 1 or len(common_ids) < media_ends[0]:
        raise ValueError("common expanded prefix does not contain the complete image")
    protocol = CacheProtocol(
        image_png_base64=encoded,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        shared_text=common_text,
        shared_prompt_token_ids=common_ids,
        media_prefix_end=media_ends[0],
        answer_token_ids=mappings[0],
        cases=[
            CacheCase(tail=text[len(common_text) :], prompt_token_ids=ids)
            for text, ids in zip(texts, expanded, strict=True)
        ],
        repeats=repeats,
        warmup_requests=warmup_requests,
        seed=42,
        output_tolerance=0.0,
    )
    # This binding proves the collector will send the exact officially prepared messages.
    for index, item in enumerate(prepared):
        if protocol.payload("unused-model", index)["messages"] != item.http_messages:
            raise ValueError("frozen native payload differs from canonical official preparation")
    questions = {}
    for probe in probes:
        question = probe.question
        public: dict[str, Any] = {"type": question.type, "instructions": question.instruction}
        if question.type == "choice":
            public["criteria"] = {
                chr(65 + i): value for i, value in enumerate(question.options or [])
            }
        elif question.type == "score":
            public["criteria"] = question.score_descriptions or [
                str(i) for i in range(question.score_levels or 0)
            ]
        questions[question.qid] = public
    gateway = {
        "state": original.prefix.state,
        "media": [media.model_dump() for media in original.prefix.media],
        "video_frames": original.video_frames,
        "model": "lite-omnijev-e4b-dev",
        "questions": questions,
    }
    summary = {
        "status": "frozen-cpu",
        "scope": "non-quality cache probes only",
        "source_case_id": original.id,
        "source_family": original.meta.family_id,
        "source_manifest_case_sha256": digest(original.model_dump()),
        "protocol_sha256": digest(protocol.model_dump()),
        "gateway_request_sha256": digest(gateway),
        "canonical_image_sha256": protocol.image_sha256,
        "probes": 10,
        "common_prefix_tokens": len(common_ids),
        "media_prefix_end": media_ends[0],
        "same_canonical_image_and_answer_mapping": True,
        "native_payloads_equal_official_prepared_messages": True,
        "weights_loaded": False,
        "gpu_used": False,
        "s0_acceptance": False,
        "prepared_input_summaries_sha256": digest([item.summary for item in prepared]),
    }
    return protocol, gateway, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--gateway-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-requests", type=int, default=2)
    args = parser.parse_args()
    paths = (args.protocol_output, args.gateway_output, args.report)
    if len(set(path.resolve() for path in paths)) != 3 or any(path.exists() for path in paths):
        raise FileExistsError("choose distinct new artifact and report paths")
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "blocked",
        "exit_code": 1,
        "weights_loaded": False,
        "gpu_used": False,
        "s0_acceptance": False,
    }
    try:
        report.update(
            code_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
            ).strip(),
            environment={"python": platform.python_version(), "platform": platform.platform()},
            manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            model_files={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in args.model_dir.iterdir()
                if path.is_file() and path.suffix in {".json", ".jinja"}
            },
            implementation_sha256={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    Path(__file__),
                    Path(__file__).resolve().parents[1] / "eval/prefix_cache.py",
                )
            },
            cost_usd=0,
        )
        cases = read_manifest(args.manifest)
        processor = importlib.import_module("transformers").AutoProcessor.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        protocol, gateway, summary = freeze(
            cases,
            processor,
            args.asset_root,
            repeats=args.repeats,
            warmup_requests=args.warmup_requests,
        )
        with args.protocol_output.open("x") as handle:
            handle.write(protocol.model_dump_json(indent=2) + "\n")
        with args.gateway_output.open("x") as handle:
            handle.write(json.dumps(gateway, indent=2) + "\n")
        report.update(summary, exit_code=0)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(error_type=type(exc).__name__)
    report["duration_seconds"] = time.monotonic() - started
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
