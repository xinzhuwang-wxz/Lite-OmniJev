"""Run fixed public S0 teacher benchmarks with resumable evidence.

CPU modes validate the runner, fixed denominators, public mapping, canonical
video cache reads, and MMAU audio adapter path. Real teacher forward is only
loaded in ``--mode real-forward`` and remains CUDA-gated by
``train.teacher_inference.load_teacher_forward_bundle``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.teacher_benchmarks import (
    MMAU_EXPECTED_ROWS,
    BenchmarkExample,
    MVBench_EXPECTED_ROWS,
    build_benchmark_processor_inputs,
    canonical_json,
    file_sha256,
    load_mmau_examples,
    load_mvbench_examples,
    report_preamble,
    run_two_pass_forward,
    summarize_examples,
)
from scripts.assets import write_json
from train.teacher_inference import (
    DEFAULT_HEAD_CONVERSION_REPORT,
    DEFAULT_HEAD_SPEC,
    DEFAULT_MODEL_ASSET_MANIFEST,
    DEFAULT_MODEL_DIR,
    DEFAULT_SAFE_HEAD,
    DEFAULT_SOURCE_DIR,
    PreparedTeacherInput,
    TeacherForwardResult,
    load_teacher_forward_bundle,
    load_teacher_processor,
    teacher_forward_readiness_report,
    validate_teacher_source_tree,
)

ERROR_CODES: dict[type[BaseException], str] = {
    FileExistsError: "report_exists_without_resume",
    ValueError: "validation_failed",
    RuntimeError: "runtime_failed",
    OSError: "io_failed",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CODE_HASH_PATHS = {
    "runner": Path("scripts/s0_teacher_eval.py"),
    "eval_teacher_benchmarks": Path("eval/teacher_benchmarks.py"),
    "train_teacher_inference": Path("train/teacher_inference.py"),
    "train_teacher_media": Path("train/teacher_media.py"),
    "eval_public_video_protocol": Path("eval/public_video_protocol.py"),
}


class _FakeTensor:
    shape = (1, 4)
    dtype = "fake.int64"


class FakeProcessor:
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if kwargs != {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }:
            raise ValueError("unexpected_teacher_processor_call_shape")
        content = messages[0]["content"]
        media_parts = sum(1 for part in content if part["type"] != "text")
        return {"input_ids": _FakeTensor(), "media_parts": [None] * media_parts}


class ProcessorOnlyForward:
    def __init__(self, processor: Any) -> None:
        self.processor = processor


class FakeTeacherForward:
    def __init__(self) -> None:
        self.processor = FakeProcessor()

    def predict_prepared(
        self,
        prepared: PreparedTeacherInput,
        _inputs: Mapping[str, Any],
        options: Sequence[str],
    ) -> TeacherForwardResult:
        if len(options) != prepared.option_count:
            raise ValueError("fake_forward_option_count_mismatch")
        weights = [float(index + 1) for index in range(len(options))]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        best = len(options) - 1
        return TeacherForwardResult(
            prediction=options[best],
            prediction_index=best,
            confidence=probabilities[best],
            probabilities=dict(zip(options, probabilities, strict=True)),
        )


def _default_mmau_parquet(config: dict[str, Any]) -> Path:
    mmau = config["mmau"]
    return (
        Path("artifacts/public-eval-media/mmau") / str(mmau["revision"]) / str(mmau["file"]["path"])
    )


def _load_examples(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, list[BenchmarkExample]]:
    result: dict[str, list[BenchmarkExample]] = {}
    if args.benchmark in {"mmau", "all"}:
        mmau = load_mmau_examples(args.mmau_parquet, config["mmau"]["file"])
        result["mmau"] = mmau[: args.limit_mmau or None]
    if args.benchmark in {"mvbench", "all"}:
        mvbench = load_mvbench_examples(
            frame_manifest=args.mvbench_frame_manifest,
            metadata_dir=args.mvbench_metadata_dir,
            config_path=args.config,
        )
        result["mvbench"] = mvbench[: args.limit_mvbench or None]
    return result


def _all_examples(groups: dict[str, list[BenchmarkExample]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for key in ("mmau", "mvbench"):
        examples.extend(groups.get(key, []))
    return examples


def _refuse_report_overwrite(path: Path, *, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError("report_exists_without_resume")


def _runtime_metadata(started: float) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hardware": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "pid": os.getpid(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cost_usd": 0.0,
    }


def _required_file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError("code_hash_source_missing")
    return file_sha256(path)


def _code_hashes() -> dict[str, str]:
    return {
        name: _required_file_sha256(PROJECT_ROOT / relative_path)
        for name, relative_path in REQUIRED_CODE_HASH_PATHS.items()
    }


def _sanitize_error(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error_code": ERROR_CODES.get(type(error), "unhandled_exception"),
    }


def _collect_preflight_samples(
    *,
    forward: Any,
    examples: dict[str, list[BenchmarkExample]],
    video_cache_root: Path | None,
    sample_mmau_audio: int,
    sample_video: int,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "samples": {},
    }
    mmau_samples = examples.get("mmau", [])[:sample_mmau_audio]
    report["samples"]["mmau_audio"] = []
    for example in mmau_samples:
        prepared = build_benchmark_processor_inputs(
            forward,
            example,
            video_cache_root=None,
            ffmpeg_binary=ffmpeg_binary,
        )
        report["samples"]["mmau_audio"].append(
            {
                "row_hash": example.report_id,
                "processor_summary": prepared.processor_summary,
                "media_summary": prepared.media_summary,
            }
        )
    video_samples = examples.get("mvbench", [])[:sample_video]
    report["samples"]["mvbench_video"] = []
    if video_samples and video_cache_root is None:
        report["samples"]["mvbench_video_pending"] = "video_cache_root_not_provided"
    for example in video_samples if video_cache_root is not None else []:
        prepared = build_benchmark_processor_inputs(
            forward,
            example,
            video_cache_root=video_cache_root,
            ffmpeg_binary=ffmpeg_binary,
        )
        report["samples"]["mvbench_video"].append(
            {
                "row_hash": example.report_id,
                "processor_summary": prepared.processor_summary,
                "media_summary": prepared.media_summary,
            }
        )
    return report


def _preflight_media_samples(
    *,
    examples: dict[str, list[BenchmarkExample]],
    video_cache_root: Path | None,
    sample_mmau_audio: int,
    sample_video: int,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    report = _collect_preflight_samples(
        forward=FakeTeacherForward(),
        examples=examples,
        video_cache_root=video_cache_root,
        sample_mmau_audio=sample_mmau_audio,
        sample_video=sample_video,
        ffmpeg_binary=ffmpeg_binary,
    )
    report.update(
        {
            "fake_processor_used": True,
            "processor_fixture": "fake_processor_control_flow_only",
            "official_processor_used": False,
            "model_forward_run": False,
        }
    )
    return report


def _official_processor_preflight(
    *,
    processor_source_dir: Path,
    examples: dict[str, list[BenchmarkExample]],
    video_cache_root: Path | None,
    sample_mmau_audio: int,
    sample_video: int,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    source_report = validate_teacher_source_tree(processor_source_dir)
    processor = load_teacher_processor(processor_source_dir)
    report = _collect_preflight_samples(
        forward=ProcessorOnlyForward(processor),
        examples=examples,
        video_cache_root=video_cache_root,
        sample_mmau_audio=sample_mmau_audio,
        sample_video=sample_video,
        ffmpeg_binary=ffmpeg_binary,
    )
    report.update(
        {
            "official_processor_used": True,
            "processor_class": type(processor).__name__,
            "processor_source_dir": str(processor_source_dir),
            "processor_source_sha256": hashlib.sha256(
                canonical_json(source_report.get("files", {}))
            ).hexdigest(),
            "source_tree": source_report,
            "fake_processor_used": False,
            "model_forward_run": False,
            "model_backbone_loaded": False,
            "student_head_loaded": False,
            "local_files_only": True,
            "trust_remote_code": False,
        }
    )
    return report


def _write_final_report(path: Path, report: dict[str, Any], *, started_monotonic: float) -> None:
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report["runtime"].update(_runtime_metadata(started_monotonic))
    write_json(path, report)
    print(json.dumps({"status": report["status"], "exit_code": report["exit_code"]}))


def _enforce_real_forward_guards(
    args: argparse.Namespace, examples: dict[str, list[BenchmarkExample]]
) -> None:
    if args.benchmark != "all":
        raise ValueError("real_forward_requires_all_benchmarks")
    if args.limit_mmau or args.limit_mvbench:
        raise ValueError("real_forward_requires_full_denominators")
    if not math.isfinite(args.probability_tolerance) or args.probability_tolerance != 0.0:
        raise ValueError("real_forward_requires_exact_probability_match")
    if len(examples.get("mmau", [])) != MMAU_EXPECTED_ROWS:
        raise ValueError("real_forward_mmau_denominator_mismatch")
    if len(examples.get("mvbench", [])) != MVBench_EXPECTED_ROWS:
        raise ValueError("real_forward_mvbench_denominator_mismatch")


def _teacher_binding(args: argparse.Namespace, *, report: dict[str, Any]) -> dict[str, Any]:
    preprocess = {
        "seed": 0,
        "dtype_policy": "cuda_bfloat16_backbone_float32_head_probability_float64_json",
        "thinking_disabled": True,
        "audio_pcm_sample_rate": 16000,
        "video_source": "canonical_png_16_frames_no_linux_mp4_redecode",
        "probability_tolerance": args.probability_tolerance,
    }
    if args.mode == "fake-forward":
        return {
            "mode": "fake-forward",
            "fake_forward_version": "deterministic_index_weighted_v2",
            "shared_code_hashes": _code_hashes(),
            "preprocess": preprocess,
        }
    kwargs = {
        "protocol_source_dir": args.protocol_source_dir or DEFAULT_SOURCE_DIR,
        "model_dir": args.model_dir or DEFAULT_MODEL_DIR,
        "model_asset_manifest": args.model_asset_manifest or DEFAULT_MODEL_ASSET_MANIFEST,
        "head_path": args.head or DEFAULT_SAFE_HEAD,
        "spec_path": args.head_spec or DEFAULT_HEAD_SPEC,
        "conversion_report": args.head_conversion_report or DEFAULT_HEAD_CONVERSION_REPORT,
    }
    readiness = teacher_forward_readiness_report(
        protocol_source_dir=kwargs["protocol_source_dir"],
        model_dir=kwargs["model_dir"],
        model_asset_manifest=kwargs["model_asset_manifest"],
        head_path=kwargs["head_path"],
        spec_path=kwargs["spec_path"],
        conversion_report=kwargs["conversion_report"],
    )
    report["teacher_readiness"] = {
        key: readiness.get(key)
        for key in (
            "status",
            "exit_code",
            "model_forward_run",
            "model_backbone_loaded",
            "trust_remote_code",
            "local_files_only",
            "use_safetensors_for_model",
            "use_safetensors_for_head",
            "cuda_available",
        )
        if key in readiness
    }
    model_assets = readiness.get("model_assets", {}) if isinstance(readiness, Mapping) else {}
    protocol_source = readiness.get("protocol_source", {}) if isinstance(readiness, Mapping) else {}
    head = readiness.get("head", {}) if isinstance(readiness, Mapping) else {}
    return {
        "mode": "real-forward",
        "model_dir": str(kwargs["model_dir"]),
        "model_asset_manifest": str(kwargs["model_asset_manifest"]),
        "model_source_revision": model_assets.get("source_revision"),
        "model_asset_sha256": hashlib.sha256(
            canonical_json(model_assets.get("files", {}))
        ).hexdigest(),
        "model_asset_bytes_verified": model_assets.get("bytes_verified"),
        "head_path": str(kwargs["head_path"]),
        "head_sha256": head.get("sha256"),
        "teacher_source_revision": protocol_source.get("source_revision"),
        "teacher_source_sha256": hashlib.sha256(
            canonical_json(protocol_source.get("files", {}))
        ).hexdigest(),
        "shared_code_hashes": _code_hashes(),
        "preprocess": preprocess,
    }


def _load_real_forward(args: argparse.Namespace) -> Any:
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = {
        key: value
        for key, value in {
            "protocol_source_dir": args.protocol_source_dir,
            "model_dir": args.model_dir,
            "model_asset_manifest": args.model_asset_manifest,
            "head_path": args.head,
            "spec_path": args.head_spec,
            "conversion_report": args.head_conversion_report,
        }.items()
        if value is not None
    }
    return load_teacher_forward_bundle(**kwargs)


def main(argv: list[str] | None = None) -> int:
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["cpu-preflight", "fake-forward", "real-forward"], required=True
    )
    parser.add_argument("--benchmark", choices=["mmau", "mvbench", "all"], default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/video-eval-assets.json"))
    parser.add_argument("--mmau-parquet", type=Path)
    parser.add_argument(
        "--mvbench-frame-manifest",
        type=Path,
        default=Path("artifacts/public-video-protocol-freeze-001/canonical-frame-manifest.json"),
    )
    parser.add_argument(
        "--mvbench-metadata-dir",
        type=Path,
        default=Path("artifacts/data/research-eval/video-eval-source-metadata"),
    )
    parser.add_argument("--video-cache-root", type=Path)
    parser.add_argument("--processor-source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records-path",
        type=Path,
        default=Path("artifacts/s0-teacher-eval/predictions.jsonl"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-mmau", type=int, default=0)
    parser.add_argument("--limit-mvbench", type=int, default=0)
    parser.add_argument("--sample-mmau-audio", type=int, default=3)
    parser.add_argument("--sample-video", type=int, default=0)
    parser.add_argument("--probability-tolerance", type=float, default=0.0)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--protocol-source-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-asset-manifest", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--head-spec", type=Path)
    parser.add_argument("--head-conversion-report", type=Path)
    args = parser.parse_args(argv)

    report = report_preamble(mode=args.mode)
    report.update(
        {
            "status": "running",
            "exit_code": 1,
            "benchmark": args.benchmark,
            "runtime": {
                "started_monotonic_seconds": started_monotonic,
                "seed": 0,
                "hardware": {},
            },
            "release_or_stage_acceptance": False,
            "report_contains_prompt_option_or_label_text": False,
        }
    )
    forward: Any | None = None
    try:
        _refuse_report_overwrite(args.output, resume=args.resume)
        report["code_sha256"] = _code_hashes()
        config = json.loads(args.config.read_text())
        args.mmau_parquet = args.mmau_parquet or _default_mmau_parquet(config)
        examples = _load_examples(args, config)
        all_examples = _all_examples(examples)
        report.update(
            {
                "config_sha256": file_sha256(args.config),
                "runner_code_sha256": file_sha256(Path(__file__)),
                "example_sets": {name: summarize_examples(rows) for name, rows in examples.items()},
                "total_rows_selected": len(all_examples),
                "limits": {"mmau": args.limit_mmau, "mvbench": args.limit_mvbench},
                "full_denominator_available": {
                    "mmau": len(examples.get("mmau", [])) in {0, MMAU_EXPECTED_ROWS},
                    "mvbench": len(examples.get("mvbench", [])) in {0, MVBench_EXPECTED_ROWS},
                },
            }
        )
        if args.mode == "real-forward":
            _enforce_real_forward_guards(args, examples)
        if args.mode == "cpu-preflight":
            report["media_preflight"] = _preflight_media_samples(
                examples=examples,
                video_cache_root=args.video_cache_root,
                sample_mmau_audio=args.sample_mmau_audio,
                sample_video=args.sample_video,
                ffmpeg_binary=args.ffmpeg_binary,
            )
            if args.processor_source_dir is not None:
                report["official_processor_preflight"] = _official_processor_preflight(
                    processor_source_dir=args.processor_source_dir,
                    examples=examples,
                    video_cache_root=args.video_cache_root,
                    sample_mmau_audio=args.sample_mmau_audio,
                    sample_video=args.sample_video,
                    ffmpeg_binary=args.ffmpeg_binary,
                )
                report["status"] = "passed_cpu_preflight_official_processor_no_model_forward"
            else:
                report["status"] = "passed_cpu_preflight_no_model_forward_fake_processor"
            report["exit_code"] = 0
        else:
            if args.mode == "fake-forward":
                forward = FakeTeacherForward()
                report["forward"] = {
                    "mode": "fake",
                    "model_forward_run": False,
                    "release_or_stage_acceptance": False,
                }
            else:
                report["gpu_seconds"] = "measured_by_outer_runner"
                report["forward"] = {"mode": "real_cuda"}
            binding = {
                "mode": args.mode,
                "benchmark": args.benchmark,
                "rows": [example.row_id for example in all_examples],
                "config_sha256": report["config_sha256"],
                "video_manifest_sha256": file_sha256(args.mvbench_frame_manifest)
                if args.benchmark in {"mvbench", "all"}
                else None,
                "mmau_parquet_sha256": file_sha256(args.mmau_parquet)
                if args.benchmark in {"mmau", "all"}
                else None,
                "teacher": _teacher_binding(args, report=report),
            }
            if args.mode == "real-forward":
                forward = _load_real_forward(args)
            assert forward is not None
            forward_summary = run_two_pass_forward(
                forward=forward,
                examples=all_examples,
                video_cache_root=args.video_cache_root,
                records_path=args.records_path,
                binding=binding,
                probability_tolerance=args.probability_tolerance,
                ffmpeg_binary=args.ffmpeg_binary,
                resume=args.resume,
                release_or_stage_candidate=args.mode == "real-forward",
            )
            report["two_pass"] = forward_summary
            report["release_or_stage_acceptance"] = bool(
                forward_summary["release_or_stage_acceptance"]
            )
            if args.mode == "fake-forward":
                report["status"] = "passed_fake_forward_control_flow_not_acceptance"
                report["exit_code"] = 0 if forward_summary["two_pass_identical"] else 1
            else:
                report["status"] = (
                    "passed_real_forward_release_gate"
                    if report["release_or_stage_acceptance"]
                    else "failed_real_forward_release_gate"
                )
                report["exit_code"] = 0 if report["release_or_stage_acceptance"] else 1
    except Exception as error:
        report["status"] = "failed"
        report["exit_code"] = 1
        report.update(_sanitize_error(error))
    finally:
        close = getattr(forward, "close", None)
        if callable(close):
            try:
                close()
                report["forward_closed"] = True
            except Exception:
                report["forward_closed"] = False
        _write_final_report(args.output, report, started_monotonic=started_monotonic)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
