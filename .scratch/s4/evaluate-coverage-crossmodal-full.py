"""One source and one pinned adapter per invocation; full dev crossmodal measurement.

Default CPU preparation reads only the chosen dev source and adapter identities.
Explicit --execute enables a bounded GPU job (default100/180s), with progress every25.
No small-screen threshold gates; --limit500 explicitly opts into a full source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import time
from pathlib import Path

from scripts import s0_development_baselines as baseline

SHARED = Path(__file__).with_name("screen-crossmodal-retention.py")
_spec = importlib.util.spec_from_file_location("coverage_screen", SHARED)
assert _spec is not None and _spec.loader is not None
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
ROOT, BASE, CROSS = _shared.ROOT, _shared.BASE, _shared.CROSS
sha, require, cross_source = _shared.sha, _shared.require, _shared.cross_source
ADAPTERS = {
    "first": (
        ROOT / "artifacts/s4-joint-pilot-train-001/adapter",
        "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac",
    ),
    "coverage": (
        ROOT / "artifacts/s4-crossmodal-retention-train-001/adapter",
        "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818",
    ),
}


def full_samples(name: str, data: Path) -> list:
    require(sha(data) == CROSS[name][0], "full dev source SHA mismatch")
    samples = baseline.load_samples(data)
    baseline.validate_samples(samples, cohort="crossmodal")
    require(all(s.meta.pool == "dev" for s in samples), "full crossmodal must be dev")
    require(len({s.id for s in samples}) == len(samples), "duplicate full dev IDs")
    require(
        len(samples) == 500 if name == "supplemental" else len(samples) in (50, 500),
        "unexpected actual full source count",
    )
    # Same order as the prior fixed50 screen, without slicing or repetition.
    return sorted(
        samples, key=lambda s: hashlib.sha256(("42:s4-crossmodal:" + s.id).encode()).hexdigest()
    )


def full_forward(samples: list, media: Path, adapter: Path, out: Path, paired: dict) -> dict:
    # Same fixed first-pilot selection and evaluator; imports/loads CUDA only on execution.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from scripts import s0_development_baselines as baseline

    processor = baseline.default_processor_factory(BASE)

    class BudgetProcessor:
        def __getattr__(self, key):
            return getattr(processor, key)

        def __call__(self, **kwargs):
            if "images" in kwargs:
                kwargs["images_kwargs"] = {"max_soft_tokens": 1120}
            return processor(**kwargs)

    baseline.configure_torch_runtime(torch)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    try:

        def forward(prepared):
            if paired:
                previous = paired[prepared.sample.id]
                require(
                    previous["input_sha256"] == prepared.input_sha256
                    and previous["input_summary_sha256"]
                    == baseline.sha256_json(prepared.input_summary)
                    and previous["answer_token_ids"] == prepared.answer_token_ids,
                    "paired prepared inputs or answer mapping differ",
                )
            inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                logits = result.logits[0, -1].float()
                require(bool(torch.isfinite(logits).all()), "nonfinite crossmodal logits")
                logprobs = logits.log_softmax(-1)
                return {
                    int(i): float(logprobs[int(i)].item())
                    for i in prepared.answer_token_ids.values()
                }

        records = []
        started = time.monotonic()
        previous = started
        for sample in samples:
            records.extend(
                baseline.run_pass(
                    [sample],
                    processor=BudgetProcessor(),
                    media_root=media,
                    pass_index=1,
                    forward=forward,
                    records_sidecar=out / "records.jsonl",
                )
            )
            if len(records) % 25 == 0 or len(records) == len(samples):
                now = time.monotonic()
                progress = {
                    "status": "evaluating",
                    "completed": len(records),
                    "total": len(samples),
                    "metrics": baseline.score_records(records),
                    "evaluation_seconds": now - started,
                    "batch_seconds": now - previous,
                    "scope": "public dev prefix only; not stage acceptance",
                }
                temp = out / "progress.tmp"
                temp.write_text(json.dumps(progress, indent=2) + "\n")
                temp.replace(out / "progress.json")
                previous = now
        metrics = baseline.score_records(records)
        require(metrics["count"] == len(samples), "full crossmodal result count mismatch")
        return {
            "status": "completed",
            "exit_code": 0,
            "metrics": metrics,
            "records_sha256": sha(out / "records.jsonl"),
            "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "image_max_soft_tokens": 1120,
            "code_sha256": {p: baseline.sha256_file(Path(p)) for p in baseline.REQUIRED_CODE_FILES},
        }
    finally:
        model = None
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("original", "supplemental"), required=True)
    parser.add_argument("--adapter", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--expected-adapter-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-records", type=Path)
    parser.add_argument("--paired-records-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, choices=(50, 100, 200, 500), default=100)
    parser.add_argument("--budget-seconds", type=int, choices=(180, 300, 600), default=180)
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = {
        "status": "failed",
        "exit_code": 1,
        "stage_gate": False,
        "scope": "full chosen public-dev file; no calibration/holdout; no screening threshold",
        "gpu_requested": args.execute,
        "cost": None,
        "source": args.source,
        "adapter": args.adapter,
        "runner_sha256": sha(Path(__file__)),
        "shared_source_sha256": sha(SHARED),
        "budget_seconds": args.budget_seconds,
        "requested_count": args.limit,
    }
    start = time.monotonic()

    def interrupted(signum, frame):
        raise TimeoutError(f"job interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        data, media = cross_source(args.source)
        report["actual_full_source_count"] = sum(
            bool(line.strip()) for line in data.read_text().splitlines()
        )
        samples = full_samples(args.source, data)
        source_count = len(samples)
        require(args.limit <= source_count, "limit exceeds actual source count")
        samples = samples[: args.limit]
        adapter, expected = ADAPTERS[args.adapter]
        require(sha(adapter / "adapter_model.safetensors") == expected, "adapter SHA mismatch")
        require(
            sha(adapter / "adapter_config.json") == args.expected_adapter_config_sha256,
            "adapter config SHA mismatch",
        )
        report.update(
            data_sha256=sha(data),
            data_path=str(data),
            actual_full_source_count=source_count,
            evaluated_count=len(samples),
            complete_source=len(samples) == source_count,
            expanded_beyond_fixed50=len(samples) > 50,
            adapter_sha256=expected,
            adapter_config_sha256=args.expected_adapter_config_sha256,
            selected_ids_sha256=hashlib.sha256(
                "\n".join(s.id for s in samples).encode()
            ).hexdigest(),
            image_max_soft_tokens=1120,
            dtype="bfloat16",
            seed=42,
            passes=1,
            code_sha256={p: baseline.sha256_file(Path(p)) for p in baseline.REQUIRED_CODE_FILES},
        )
        paired = {}
        require(
            bool(args.paired_records) == bool(args.paired_records_sha256),
            "paired path/hash must appear together",
        )
        if args.paired_records:
            require(
                sha(args.paired_records) == args.paired_records_sha256,
                "paired records SHA mismatch",
            )
            rows = [
                json.loads(line)
                for line in args.paired_records.read_text().splitlines()
                if line.strip()
            ]
            paired = {r["id"]: r for r in rows}
            require(
                len(rows) == len(paired) == len(samples) and set(paired) == {s.id for s in samples},
                "paired full source IDs differ",
            )
            report["paired_records_sha256"] = args.paired_records_sha256
        report.update(status="prepared", exit_code=0)
        if args.execute:
            signal.signal(signal.SIGALRM, interrupted)
            signal.alarm(args.budget_seconds)
            report.update(status="running", exit_code=1)
            (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
            report.update(full_forward(samples, media, adapter, args.output, paired))
            require(
                sha(data) == report["data_sha256"]
                and sha(adapter / "adapter_model.safetensors") == expected
                and sha(adapter / "adapter_config.json") == args.expected_adapter_config_sha256,
                "source or adapter changed during execution",
            )
            report["paired_inputs_verified"] = len(samples) if paired else None
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        if args.execute:
            signal.alarm(0)
        report["duration_seconds"] = time.monotonic() - start
        sidecar = args.output / "records.jsonl"
        report["records_sidecar"] = baseline.sidecar_evidence(sidecar)
        temporary = args.output / "receipt.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output / "receipt.json")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
