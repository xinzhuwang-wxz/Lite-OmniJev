"""CPU engineering smoke with explicitly synthetic logits, never a model quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from data.schema import HardTarget, answer_tokens, validate_pool_isolation
from data.synth import generate_samples
from eval.gate import evaluate_gate
from eval.metrics import evaluate
from scripts.assets import write_json
from train.readout import read_probabilities


def run(output: Path) -> int:
    samples = generate_samples(count=100, seed=42)
    validate_pool_isolation(samples)
    corpus = "\n".join(sample.model_dump_json() for sample in samples) + "\n"
    dataset = Path("artifacts/data/program-smoke.jsonl")
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(corpus)
    distributions: list[list[float]] = []
    targets: list[int] = []
    for sample in samples:
        question = sample.questions[0]
        assert isinstance(question.target, HardTarget)
        options = answer_tokens(question)
        logits = {
            token: 4.0 if token == question.target.value else -4.0 for token in (*options, "U")
        }
        readout = read_probabilities(logits, options)
        distributions.append(list(readout.probabilities.values()))
        targets.append(options.index(str(question.target.value)))
    metrics = evaluate(distributions, targets)
    gate = evaluate_gate(metrics, {"accuracy": {"min": 1.0}, "brier": {"max": 0.001}})
    report = {
        "scope": "CPU engineering fixture; logits derived from truth, NOT model predictions",
        "real_model_run": False,
        "p0_gpu_complete": False,
        "training_stage_gate_passed": False,
        "seed": 42,
        "samples": len(samples),
        "questions": sum(len(sample.questions) for sample in samples),
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_worktree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "python": platform.python_version(),
        "hardware": platform.platform(),
        "metrics": metrics,
        "fixture_gate": gate,
        "exit_code": 0 if gate["passed"] else 1,
    }
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if gate["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/cpu-smoke.json"))
    return run(parser.parse_args().output)


if __name__ == "__main__":
    raise SystemExit(main())
