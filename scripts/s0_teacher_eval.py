"""Run fixed public S0 teacher benchmarks with resumable evidence.

CPU modes validate the runner, fixed denominators, public mapping, canonical
video cache reads, and MMAU audio adapter path. Real teacher forward is only
loaded in ``--mode real-forward`` and remains CUDA-gated by
``train.teacher_inference.load_teacher_forward_bundle``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.teacher_benchmarks import (
    BenchmarkExample,
    build_benchmark_processor_inputs,
    file_sha256,
    load_mmau_examples,
    load_mvbench_examples,
    report_preamble,
    run_two_pass_forward,
    summarize_examples,
)
from scripts.assets import write_json
from train.teacher_inference import (
    PreparedTeacherInput,
    TeacherForwardResult,
    load_teacher_forward_bundle,
)


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
            raise ValueError("unexpected teacher processor call shape")
        content = messages[0]["content"]
        media_parts = sum(1 for part in content if part["type"] != "text")
        return {"input_ids": _FakeTensor(), "media_parts": [None] * media_parts}


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
            raise ValueError("fake forward option count mismatch")
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
        raise FileExistsError(f"refusing to overwrite existing report: {path}")


def _preflight_media_samples(
    *,
    examples: dict[str, list[BenchmarkExample]],
    video_cache_root: Path | None,
    sample_mmau_audio: int,
    sample_video: int,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    forward = FakeTeacherForward()
    report: dict[str, Any] = {
        "fake_processor_used": True,
        "model_forward_run": False,
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


def _write_final_report(path: Path, report: dict[str, Any]) -> None:
    report["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(path, report)
    print(json.dumps({"status": report["status"], "exit_code": report["exit_code"]}))


def main(argv: list[str] | None = None) -> int:
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

    _refuse_report_overwrite(args.output, resume=args.resume)
    config = json.loads(args.config.read_text())
    args.mmau_parquet = args.mmau_parquet or _default_mmau_parquet(config)
    examples = _load_examples(args, config)
    all_examples = _all_examples(examples)

    report = report_preamble(mode=args.mode)
    report.update(
        {
            "status": "running",
            "exit_code": 1,
            "benchmark": args.benchmark,
            "config_sha256": file_sha256(args.config),
            "runner_code_sha256": file_sha256(Path(__file__)),
            "example_sets": {name: summarize_examples(rows) for name, rows in examples.items()},
            "total_rows_selected": len(all_examples),
            "limits": {"mmau": args.limit_mmau, "mvbench": args.limit_mvbench},
            "full_denominator_available": {
                "mmau": len(examples.get("mmau", [])) in {0, 1000},
                "mvbench": len(examples.get("mvbench", [])) in {0, 1800},
            },
            "report_contains_prompt_option_or_label_text": False,
        }
    )

    try:
        if args.mode == "cpu-preflight":
            report["media_preflight"] = _preflight_media_samples(
                examples=examples,
                video_cache_root=args.video_cache_root,
                sample_mmau_audio=args.sample_mmau_audio,
                sample_video=args.sample_video,
                ffmpeg_binary=args.ffmpeg_binary,
            )
            report["status"] = "passed_cpu_preflight_no_model_forward"
            report["exit_code"] = 0
        else:
            if args.mode == "fake-forward":
                forward: Any = FakeTeacherForward()
                report["forward"] = {"mode": "fake", "model_forward_run": False}
            else:
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
                forward = load_teacher_forward_bundle(**kwargs)
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
                "runner_code_sha256": file_sha256(Path(__file__)),
            }
            forward_summary = run_two_pass_forward(
                forward=forward,
                examples=all_examples,
                video_cache_root=args.video_cache_root,
                records_path=args.records_path,
                binding=binding,
                probability_tolerance=args.probability_tolerance,
                ffmpeg_binary=args.ffmpeg_binary,
            )
            report["two_pass"] = forward_summary
            report["status"] = (
                "passed_two_pass_teacher_eval"
                if forward_summary["two_pass_identical"]
                else "failed_two_pass_teacher_eval"
            )
            report["exit_code"] = 0 if forward_summary["two_pass_identical"] else 1
    except Exception as error:
        report["status"] = "failed"
        report["exit_code"] = 1
        report["error_type"] = type(error).__name__
        report["reason"] = str(error)[:400]
    _write_final_report(args.output, report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
