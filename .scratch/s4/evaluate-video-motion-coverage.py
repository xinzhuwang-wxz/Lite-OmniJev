"""Fixed temporal dev50, one explicit16-frame policy per GPU job; CPU preparation default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from scripts import s0_development_baselines as baseline
from scripts.s1_text_pools import canonical_sample_sha
from scripts.video_motion_manifest import build_manifest
from train.multimodal import file_hash, load_media, resolve_media
from train.trainer import code_revision

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = ROOT / "reports/s3-dev-eval-video-stable-001.json"
DATA_SHA = "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2"
ADAPTER = ROOT / "artifacts/s4-claim-pair-ce0-train-001/adapter"
ADAPTER_SHA = "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"
CODE = (
    *baseline.REQUIRED_CODE_FILES,
    "train/video_sampling.py",
    "scripts/video_motion_manifest.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_new(path: Path, payload: Any) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def select_samples(samples: list) -> list:
    baseline.validate_samples(samples, cohort="video")
    temporal = [
        sample
        for sample in samples
        if set(sample.questions[0].instruction.lower().split())
        & {"before", "after", "when", "while"}
    ]
    require(len(temporal) >= 50, "fewer than50 heuristic temporal dev samples")
    return sorted(
        temporal,
        key=lambda sample: hashlib.sha256(
            ("42:s4-motion-coverage:" + sample.id).encode()
        ).hexdigest(),
    )[:50]


def source_inputs() -> tuple[Path, Path, Path, list]:
    source = json.loads(SOURCE.read_bytes())
    require(
        source["status"] == "passed"
        and source["exit_code"] == 0
        and source["data_sha256"] == DATA_SHA
        and source["samples"] == 500,
        "source report is not accepted original video500",
    )
    command = source["command"]
    require(command[1:3] == ["-m", "scripts.s1_dev_eval"], "unexpected source evaluator")
    data, media, model = [
        Path(command[command.index(flag) + 1]) for flag in ("--data", "--media-root", "--model-dir")
    ]
    require(file_hash(data) == DATA_SHA, "original video500SHA changed")
    require(model.name == "ee0ef6023621cff504d758262d4e04895a5af4a2", "model revision changed")
    return data, media, model, select_samples(baseline.load_samples(data))


def hook_loader(manifests: dict[str, dict], policy: str):
    """Experimental process-local baseline hook; other callers keep load_media default."""

    def selected(path: Path, kind: str, *, num_frames: int, max_seconds: float):
        require(
            kind == "video" and num_frames == 16 and max_seconds == 60,
            "experimental hook only admits fixed16-frame video",
        )
        record = manifests[file_hash(path)]
        indices = record["uniform_indices" if policy == "uniform" else "adaptive_indices"]
        decoded = load_media(
            path, kind, num_frames=num_frames, max_seconds=max_seconds, frame_indices=indices
        )
        metadata = decoded["metadata"]
        require(
            metadata["fps"] == record["fps"]
            and metadata["total_num_frames"] == record["total_num_frames"]
            and metadata["duration"] == record["duration"]
            and metadata["frames_indices"] == indices,
            "decode metadata differs from CPU manifest",
        )
        return decoded

    return selected


def execute(
    samples: list,
    media: Path,
    model_dir: Path,
    manifests: dict[str, dict],
    policy: str,
    output: Path,
    report: dict,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    original_loader = baseline.load_media
    original_budget = baseline.MAX_PROMPT_TOKENS
    baseline.load_media = hook_loader(manifests, policy)
    baseline.MAX_PROMPT_TOKENS = 4096
    started = time.monotonic()
    report.update(phase="loading_processor", original_completed=0, original_total=50)
    atomic(output / "progress.json", report)
    processor = baseline.default_processor_factory(model_dir)

    class FixedVideoBudget:
        def __getattr__(self, key):
            return getattr(processor, key)

        def __call__(self, **kwargs):
            kwargs["videos_kwargs"] = {**kwargs.get("videos_kwargs", {}), "max_soft_tokens": 70}
            return processor(**kwargs)

    baseline.configure_torch_runtime(torch)
    torch.cuda.reset_peak_memory_stats()
    report["phase"] = "loading_model"
    atomic(output / "progress.json", report)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    records = []
    batch_started = time.monotonic()
    try:
        report["phase"] = "evaluating"
        atomic(output / "progress.json", report)

        def forward(prepared):
            inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                logits = result.logits[0, -1].float()
                require(bool(torch.isfinite(logits).all()), "nonfinite video logits")
                logprobs = logits.log_softmax(-1)
                return {
                    int(i): float(logprobs[int(i)].item())
                    for i in prepared.answer_token_ids.values()
                }

        for sample in samples:
            prepared = baseline.prepare_sample(sample, FixedVideoBudget(), media)
            record = baseline.make_record(prepared, pass_index=1, forward=forward)
            record["frame_indices"] = prepared.input_summary["video_metadata"][0]["frames_indices"]
            record["sampling_policy"] = policy
            baseline.append_record_jsonl(output / "records.jsonl", record)
            records.append(record)
            if len(records) % 25 == 0:
                now = time.monotonic()
                correct = sum(
                    r["prediction"] == str(samples[index].questions[0].target.value)
                    for index, r in enumerate(records)
                )
                report.update(
                    original_completed=len(records),
                    original_correct=correct,
                    original_accuracy=correct / len(records),
                    elapsed_seconds=now - started,
                    last_batch_seconds=now - batch_started,
                )
                atomic(output / "progress.json", report)
                batch_started = now
        report.update(
            status="completed",
            exit_code=0,
            phase="complete",
            metrics=baseline.score_records(records),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(),
            records_sha256=file_hash(output / "records.jsonl"),
        )
    finally:
        baseline.load_media = original_loader
        baseline.MAX_PROMPT_TOKENS = original_budget
        model = None
        torch.cuda.empty_cache()
        atomic(output / "progress.json", report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/s4-motion-coverage-pilot-001"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy", choices=("uniform", "motion"))
    args = parser.parse_args()
    require(
        args.execute == bool(args.policy),
        "--execute requires one explicit --policy; CPUprepare takes no policy",
    )
    args.output = args.output.resolve()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    receipt = args.output / "preparation.json"
    output = args.output / args.policy if args.execute else args.output
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "cost": None,
        "gpu_requested": args.execute,
        "policy": args.policy,
        "stage_gate": False,
        "command": __import__("sys").argv,
        "observed_code_revision": code_revision(),
        "runner_sha256": file_hash(Path(__file__)),
        "budget_seconds": 180 if args.execute else 120,
    }

    def interrupted(signum, frame):
        raise TimeoutError(f"motion coverage job interrupted:{signum}")

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.alarm(report["budget_seconds"])
    try:
        data, media, model, samples = source_inputs()
        require(
            file_hash(ADAPTER / "adapter_model.safetensors") == ADAPTER_SHA,
            "frozen CE0 adapter SHA changed",
        )
        identity = {
            "data_sha256": file_hash(data),
            "source_report_sha256": file_hash(SOURCE),
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(s.id for s in samples).encode()
            ).hexdigest(),
            "adapter_sha256": ADAPTER_SHA,
            "adapter_config_sha256": file_hash(ADAPTER / "adapter_config.json"),
            "runner_sha256": file_hash(Path(__file__)),
            "code_sha256": {name: file_hash(Path(name)) for name in CODE},
            "model_revision": model.name,
            "num_frames": 16,
            "video_max_soft_tokens": 70,
            "max_prompt_tokens": 4096,
            "dtype": "bfloat16",
            "passes": 1,
        }
        report.update(
            identity=identity,
            samples=50,
            experimental_hook=(
                "process-local baseline.load_media calls real load_media(frame_indices=...) only"
            ),
        )
        if not args.execute:
            manifests = {}
            entries = []
            directory = output / "motion-manifests"
            directory.mkdir(mode=0o700)
            for sample in samples:
                require(
                    len(sample.prefix.media) == 1 and sample.prefix.media[0].type == "video",
                    "requires single-video source",
                )
                video = sample.prefix.media[0]
                if video.sha256 not in manifests:
                    path = resolve_media(
                        media, video.uri, video.sha256, max_bytes=baseline.MAX_MEDIA_BYTES
                    )
                    manifest = build_manifest(path, video.sha256)
                    name = directory / (video.sha256 + ".json")
                    write_new(name, manifest)
                    manifests[video.sha256] = {"path": str(name), "sha256": file_hash(name)}
                entries.append(
                    {
                        "id": sample.id,
                        "sample_sha256": canonical_sample_sha(sample),
                        "video_sha256": video.sha256,
                    }
                )
                atomic(
                    output / "progress.json",
                    {
                        "phase": "cpu_motion_manifests",
                        "completed_samples": len(entries),
                        "total_samples": 50,
                        "unique_videos": len(manifests),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            report.update(status="prepared", exit_code=0, selected=entries, manifests=manifests)
            require(file_hash(data) == DATA_SHA, "source changed during CPUprepare")
        else:
            prepared = json.loads(receipt.read_bytes())
            require(
                prepared["status"] == "prepared"
                and prepared["exit_code"] == 0
                and prepared["identity"] == identity,
                "CPUpreparation identity changed",
            )
            require(
                prepared["selected"]
                == [
                    {
                        "id": s.id,
                        "sample_sha256": canonical_sample_sha(s),
                        "video_sha256": s.prefix.media[0].sha256,
                    }
                    for s in samples
                ],
                "selected inputs changed",
            )
            manifests = {}
            for digest, bound in prepared["manifests"].items():
                path = Path(bound["path"])
                require(file_hash(path) == bound["sha256"], "motion manifest changed")
                manifest = json.loads(path.read_bytes())
                require(manifest["video_sha256"] == digest, "motion source mismatch")
                manifests[digest] = manifest
            report["preparation_sha256"] = file_hash(receipt)
            execute(samples, media, model, manifests, args.policy, output, report)
            require(
                file_hash(data) == DATA_SHA
                and file_hash(ADAPTER / "adapter_model.safetensors") == ADAPTER_SHA
                and file_hash(ADAPTER / "adapter_config.json") == identity["adapter_config_sha256"],
                "source or adapter changed during GPU job",
            )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        report["duration_seconds"] = time.monotonic() - started
        partial = output / "records.jsonl"
        report["partial_records_sha256"] = file_hash(partial) if partial.exists() else None
        atomic(output / ("receipt.json" if args.execute else "preparation.json"), report)
        if args.execute:
            atomic(output / "progress.json", report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
