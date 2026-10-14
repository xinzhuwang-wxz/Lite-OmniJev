"""Same-condition dev evaluation for a text decision adapter; two identical passes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample, answer_tokens
from eval.metrics import evaluate
from scripts.check_tokens import OfficialTokenizerAdapter
from train.readout import read_probabilities
from train.template import check_answer_tokens, render_prompt


class DevEvalError(ValueError):
    """User-correctable input or evaluation contract failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sample = Sample.model_validate_json(stripped)
            except ValueError as exc:
                raise DevEvalError(f"invalid sample JSON at line {line_no}") from exc
            if sample.meta.modality != ["text"] or sample.prefix.media:
                raise DevEvalError(f"sample {sample.id} is not text-only")
            if len(sample.questions) != 1 or sample.questions[0].type != "choice":
                raise DevEvalError(f"sample {sample.id} is not a single Choice question")
            target = sample.questions[0].target
            if not isinstance(target, HardTarget) or type(target.value) is not str:
                raise DevEvalError(f"sample {sample.id} lacks a hard answer-token target")
            samples.append(sample)
    ids = [sample.id for sample in samples]
    if len(set(ids)) != len(ids):
        raise DevEvalError("duplicate sample ids")
    return samples


def forward_readout(
    model: Any,
    tokenizer: OfficialTokenizerAdapter,
    sample: Sample,
    max_length: int,
) -> tuple[dict[str, float], Any]:
    """Boundary-verified original logits for every legal answer token plus U."""
    torch = importlib.import_module("torch")
    question = sample.questions[0]
    prompt = render_prompt(tokenizer, sample.prefix, question)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids or len(input_ids) > max_length:
        raise DevEvalError(f"sample {sample.id} has an empty or over-length prompt")
    mapping = check_answer_tokens(tokenizer, question)
    for token, index in mapping.items():
        if tokenizer.encode(prompt + token, add_special_tokens=False) != input_ids + [index]:
            raise DevEvalError(f"answer token for {sample.id} changes at the generation boundary")
    ids = torch.tensor([input_ids], device="cuda", dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    row = output.logits[0, -1]
    logits = {token: float(row[index]) for token, index in mapping.items()}
    readout = read_probabilities(logits, list(answer_tokens(question)), temperature=1.0)
    return logits, readout


def attach_adapter(model: Any, adapter_dir: Path) -> Any:
    peft = importlib.import_module("peft")
    adapted = peft.PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)
    trainable = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    if trainable:
        raise DevEvalError("evaluation adapter exposes trainable parameters")
    return adapted.eval()


def record_for(sample: Sample, readout: Any, logits: dict[str, float]) -> dict[str, Any]:
    target = sample.questions[0].target
    assert isinstance(target, HardTarget)
    return {
        "id": sample.id,
        "family_hash": sample.meta.family_id,
        "pool": sample.meta.pool,
        "n_options": len(sample.questions[0].options or []),
        "raw_logits": {key: logits[key] for key in sorted(logits)},
        "probabilities": readout.probabilities,
        "joint_probabilities": readout.joint_probabilities,
        "p_insufficient": readout.p_insufficient,
        "confidence": readout.confidence,
        "prediction": readout.prediction,
        "target": target.value,
    }


def metrics_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[list[float]] = []
    targets: list[int] = []
    for record in records:
        probabilities = record["probabilities"]
        rows.append([probabilities[letter] for letter in sorted(probabilities)])
        targets.append(ord(str(record["target"])) - ord("A"))
    core = evaluate(rows, targets)
    count = len(records)
    return {
        **core,
        "product_accuracy": sum(record["prediction"] == record["target"] for record in records)
        / count,
        "mean_p_insufficient": sum(record["p_insufficient"] for record in records) / count,
        "max_p_insufficient": max(record["p_insufficient"] for record in records),
        "predicted_insufficient": sum(record["prediction"] == "U" for record in records),
    }


def run(
    *,
    model_dir: Path,
    adapter_dir: Path | None,
    data_path: Path,
    output_dir: Path,
    report_out: Path,
    max_length: int,
    passes: int,
    command: list[str],
) -> int:
    started_at = time.time()
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        samples = load_samples(data_path)
        base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = attach_adapter(base, adapter_dir) if adapter_dir else base
        model = model.to("cuda").eval()
        tokenizer = OfficialTokenizerAdapter(model_dir)
        all_passes: list[list[dict[str, Any]]] = []
        for _ in range(passes):
            records = []
            for sample in samples:
                logits, readout = forward_readout(model, tokenizer, sample, max_length)
                records.append(record_for(sample, readout, logits))
            all_passes.append(records)
        repeat_identical = all(
            canonical_json(first) == canonical_json(second)
            for first, second in zip(all_passes, all_passes[1:], strict=True)
        )
        sidecar = output_dir / "records.jsonl"
        with sidecar.open("w", encoding="utf-8") as handle:
            for pass_index, records in enumerate(all_passes):
                for record in records:
                    handle.write(canonical_json({**record, "pass": pass_index}) + "\n")
        metrics = metrics_from_records(all_passes[0])
        report = {
            "status": "passed" if repeat_identical else "passed_unstable_repeat",
            "exit_code": 0,
            "command": command,
            "model_revision": model_dir.name,
            "adapter_sha256": (
                sha256_file(adapter_dir / "adapter_model.safetensors") if adapter_dir else None
            ),
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            "data_sha256": sha256_file(data_path),
            "samples": len(samples),
            "passes": passes,
            "repeat_identical": repeat_identical,
            "metrics": metrics,
            "code_revision": git_head(),
            "script_sha256": script_sha256(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "peft", "accelerate", "safetensors")
            },
            "gpu_used": True,
            "quality_gate": "not_run",
            "cost": None,
            "sidecar_sha256": sha256_file(sidecar),
            "timing": {
                "started_at": started_at,
                "finished_at": time.time(),
                "duration_seconds": time.time() - started_at,
            },
        }
        code = 0
    except (OSError, DevEvalError, ValueError) as exc:
        import traceback

        traceback.print_exc()
        report = {
            "status": "blocked",
            "exit_code": 1,
            "blocker": f"{type(exc).__name__}: {exc}",
            "command": command,
            "cost": None,
            "timing": {"started_at": started_at, "finished_at": time.time()},
        }
        code = 1
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--passes", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        model_dir=args.model_dir.resolve(),
        adapter_dir=args.adapter.resolve() if args.adapter else None,
        data_path=args.data.resolve(),
        output_dir=args.output_dir,
        report_out=args.report,
        max_length=args.max_length,
        passes=args.passes,
        command=[sys.executable, "-m", "scripts.s1_dev_eval", *(argv or sys.argv[1:])],
    )


if __name__ == "__main__":
    raise SystemExit(main())
