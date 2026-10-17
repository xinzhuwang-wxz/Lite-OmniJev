"""Score public-bench Sample contracts against non-Gemma comparator models.

ADR0025 requires the four comparators measured "same protocol, same questions,
measured by us". This runner keeps the question contracts, letter mapping,
answer-letter readout semantics and media bytes identical to
scripts/s1_dev_eval, while letting each comparator model use its own chat
template and processor. Gemma-family models (our students, Jev-Omni) must use
s1_dev_eval; this harness exists for qwen3_5-family comparators.

Scoring semantics mirrored from s1_dev_eval.forward_readout: deterministic
forward, last-position logits, every legal answer token plus U, temperature-1
read_probabilities. The letter boundary check is unchanged: a letter must be
a single token that continues the prompt exactly.

Media preparation is family-specific and gated behind a smoke run: a family
path must pass a --limit smoke on the box against the installed transformers
API before any scored run (--smoked unlocks the full run). Video is refused
until its family path is written and validated. Scored reports carry
protocol_notes and the harness tag so they cannot masquerade as s1_dev_eval
numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.schema import HardTarget, Sample, answer_tokens  # noqa: E402
from eval.metrics import evaluate  # noqa: E402
from train.readout import read_probabilities  # noqa: E402

HARNESS_TAG = "comparator-eval-v1"


class DevEvalError(RuntimeError):
    """Raised on protocol violations; the report records the failure."""


@dataclass(frozen=True)
class PreparedRow:
    input_ids: list[int]
    prompt_text: str
    forward_kwargs: dict[str, Any]
    boundary_tail: str | None = None


@dataclass(frozen=True)
class Loaded:
    model: Any
    tokenizer: Any
    processor: Any


class FamilyAdapter(Protocol):
    """Per-model-family loading plus Sample -> forward inputs."""

    def load(self, model_dir: Path) -> Loaded: ...

    def prepare(
        self, loaded: Loaded, sample: Sample, media_root: Path, *, max_length: int
    ) -> PreparedRow: ...


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        try:
            samples.append(Sample.model_validate_json(line))
        except ValueError as exc:
            raise DevEvalError(f"invalid sample JSON at line {line_no}") from exc
    ids = [sample.id for sample in samples]
    if len(set(ids)) != len(ids):
        raise DevEvalError("duplicate sample ids")
    return samples


def letter_mapping(
    tokenizer: Any,
    sample: Sample,
    input_ids: list[int],
    prompt_text: str,
    boundary_tail: str | None = None,
) -> dict[str, int]:
    """s1_dev_eval boundary check: prompt+letter encodes to prompt+single token.

    Families whose processor expands media placeholder tokens mid-prompt (vision
    pads) pass boundary_tail: the literal generation-position suffix; the check
    then requires input_ids to end with that tail and each letter to continue it
    as one token, which pins the same generation boundary.
    """
    tail_ids = tokenizer.encode(boundary_tail, add_special_tokens=False) if boundary_tail else None
    if tail_ids is not None and input_ids[-len(tail_ids) :] != tail_ids:
        raise DevEvalError(f"{sample.id}: rendered tail does not match boundary_tail")
    prefix = boundary_tail if boundary_tail is not None else prompt_text
    mapping: dict[str, int] = {}
    for token in (*answer_tokens(sample.questions[0]), "U"):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise DevEvalError(f"{sample.id}: answer token {token!r} is not a single token")
        continuation = tokenizer.encode(prefix + token, add_special_tokens=False)
        expected = [*tail_ids, *encoded] if tail_ids is not None else [*input_ids, *encoded]
        if continuation != expected:
            raise DevEvalError(f"{sample.id}: answer token boundary broken for {token!r}")
        mapping[token] = encoded[0]
    return mapping


def score_row(
    model: Any,
    input_ids: list[int],
    mapping: dict[str, int],
    forward_kwargs: dict[str, Any],
    *,
    device: str = "cuda",
) -> dict[str, float]:
    torch = importlib.import_module("torch")
    ids = torch.tensor([input_ids], device=device, dtype=torch.long)
    # Media-padded sequences must keep the processor's attention mask; text-only
    # families pass no mask and get all-ones.
    kwargs = {key: value.to(ids.device) for key, value in forward_kwargs.items()}
    if "attention_mask" not in kwargs:
        kwargs["attention_mask"] = torch.ones_like(ids)
    with torch.no_grad():
        output = model(
            input_ids=ids,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
            **kwargs,
        )
    row = output.logits[0, -1]
    return {token: float(row[index]) for token, index in mapping.items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    data_path = args.data.resolve()
    samples = load_samples(data_path)
    if args.limit:
        samples = samples[: args.limit]

    module = importlib.import_module(f"scripts.comparator_families.{args.family}")
    builder: Callable[[], FamilyAdapter] = module.build
    adapter = builder()
    loaded = adapter.load(args.model_dir.resolve())

    tokens_by_sample = {sample.id: answer_tokens(sample.questions[0]) for sample in samples}
    rows: list[dict[str, Any]] = []
    probabilities: list[list[float]] = []
    targets: list[int] = []
    for sample in samples:
        question = sample.questions[0]
        target = question.target
        if not isinstance(target, HardTarget) or type(target.value) is not str:
            raise DevEvalError(f"{sample.id}: comparator scoring needs a hard letter target")
        try:
            prepared = adapter.prepare(
                loaded, sample, args.media_root.resolve(), max_length=args.max_length
            )
            mapping = letter_mapping(
                loaded.tokenizer,
                sample,
                prepared.input_ids,
                prepared.prompt_text,
                prepared.boundary_tail,
            )
            logits = score_row(
                loaded.model,
                prepared.input_ids,
                mapping,
                prepared.forward_kwargs,
                device=args.device,
            )
            readout = read_probabilities(logits, list(tokens_by_sample[sample.id]), temperature=1.0)
        except (DevEvalError, RuntimeError, ValueError) as exc:
            # per-row containment: one over-length prompt or an OOM must not
            # kill the run; the row is recorded and the run continues
            rows.append({"id": sample.id, "error": type(exc).__name__, "detail": str(exc)[:200]})
            continue
        tokens = tokens_by_sample[sample.id]
        probabilities.append([readout.probabilities[token] for token in tokens])
        targets.append(tokens.index(target.value))
        rows.append(
            {
                "id": sample.id,
                "logits": logits,
                "predicted": readout.prediction,
                "correct": readout.prediction == target.value,
            }
        )
    metrics = evaluate(probabilities, targets) if probabilities else {"count": 0}
    report = {
        "harness": HARNESS_TAG,
        "family": args.family,
        "model_dir": str(args.model_dir),
        "data_sha256": file_sha256(data_path),
        "samples": len(samples),
        "scored": len(probabilities),
        "protocol_violations": len(rows) - len(probabilities),
        "metrics": metrics,
        "protocol_notes": [
            "same Sample contracts, option order and letter mapping as s1_dev_eval runs",
            "comparator model rendered with its own chat template; prefix state preserved",
            "identical letter-logit readout (temperature 1, U normalized jointly)",
            "single pass; two-pass consistency required before entering the ADR0025 table",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="scripts/comparator_families/<name>.py")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int, help="smoke-run row cap; never for scored runs")
    parser.add_argument(
        "--smoked",
        action="store_true",
        help="confirm the family path passed its GPU smoke; required for scored runs",
    )
    args = parser.parse_args(argv)
    if not args.smoked and not args.limit:
        raise SystemExit(
            "refusing scored run: pass --limit for a smoke or --smoked after it passes"
        )
    try:
        report = run(args)
    except DevEvalError as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"harness": HARNESS_TAG, "status": "failed", "error": str(exc)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"comparator_eval failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"scored": report["scored"], "accuracy": report["metrics"].get("accuracy")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
