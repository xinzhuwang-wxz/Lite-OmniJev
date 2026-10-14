"""Teacher benchmark runners for fixed public S0 inputs.

The module is intentionally payload-aware but report-light: it may read public
benchmark questions and labels to prepare and score the teacher, but summaries
and resumable records store only row identifiers, option indices, probabilities,
and hashes. It never reads sealed data and never changes benchmark denominators.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from eval.public_video_protocol import REFERENCE_SHA as PUBLIC_MVBENCH_REFERENCE_SHA256
from eval.public_video_protocol import TASKS as PUBLIC_MVBENCH_TASKS
from eval.public_video_protocol import references as public_video_references
from train.teacher_inference import (
    PreparedTeacherInput,
    TeacherForwardResult,
    build_teacher_processor_inputs,
    official_teacher_prompt,
    prepared_teacher_input,
    validate_teacher_options,
)

SCHEMA_VERSION = 1
MVBECH_REFERENCE_SHA256 = PUBLIC_MVBENCH_REFERENCE_SHA256
MVBench_REFERENCE_SHA256 = PUBLIC_MVBENCH_REFERENCE_SHA256
MMAU_EXPECTED_ROWS = 1000
MMAU_SCORE_TARGET = 0.631
MMAU_SCORE_TOLERANCE = 0.02
MVBench_EXPECTED_ROWS = 1800
MVBench_TASKS = PUBLIC_MVBENCH_TASKS
EXPECTED_ROWS_BY_BENCHMARK = {
    "mmau-test-mini": MMAU_EXPECTED_ROWS,
    "mvbench-public-9task": MVBench_EXPECTED_ROWS,
}


class TeacherForward(Protocol):
    processor: Any

    def predict_prepared(
        self,
        prepared: PreparedTeacherInput,
        inputs: Mapping[str, Any],
        options: Sequence[str],
    ) -> TeacherForwardResult: ...


@dataclass(frozen=True)
class VideoFrameRef:
    index: int
    png: str
    png_sha256: str
    rgb24_sha256: str


@dataclass(frozen=True)
class BenchmarkExample:
    benchmark: str
    row_id: str
    group: str
    question: str
    options: tuple[str, ...]
    label_index: int
    modality: str
    source_sha256: str
    media_bytes_sha256: str | None = None
    media_bytes: bytes | None = None
    video_frames: tuple[VideoFrameRef, ...] = ()
    width: int | None = None
    height: int | None = None

    @property
    def report_id(self) -> str:
        return hashlib.sha256(f"{self.benchmark}:{self.row_id}".encode()).hexdigest()


@dataclass(frozen=True)
class PreparedBenchmarkInput:
    prepared: PreparedTeacherInput
    inputs: Mapping[str, Any]
    processor_summary: dict[str, Any]
    media_summary: dict[str, Any]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_prefixed_answer_index(answer: str, options: Sequence[str]) -> int:
    checked = validate_teacher_options(options)
    if answer in checked:
        return checked.index(answer)
    if len(answer) >= 3 and answer[0] == "(" and answer[2] == ")":
        index = ord(answer[1].upper()) - ord("A")
        if 0 <= index < len(checked) and checked[index] == answer:
            return index
    raise ValueError("answer is not one of the fixed options")


def load_mmau_examples(
    parquet: Path, expected: Mapping[str, Any] | None = None
) -> list[BenchmarkExample]:
    if expected is not None:
        if parquet.stat().st_size != expected.get("bytes") or file_sha256(parquet) != expected.get(
            "sha256"
        ):
            raise ValueError("mmau_source_hash_mismatch")
    pyarrow_parquet = __import__("pyarrow.parquet", fromlist=["ParquetFile"])
    source_sha = file_sha256(parquet)
    file = pyarrow_parquet.ParquetFile(parquet)
    if file.metadata.num_rows != MMAU_EXPECTED_ROWS:
        raise ValueError("mmau_public_row_count_mismatch")
    examples: list[BenchmarkExample] = []
    seen: set[str] = set()
    columns = ["context", "instruction", "choices", "answer", "other_attributes"]
    for group in range(file.metadata.num_row_groups):
        for batch in file.iter_batches(
            batch_size=1, row_groups=[group], columns=columns, use_threads=False
        ):
            for row in batch.to_pylist():
                attributes = json.loads(row["other_attributes"])
                row_id = attributes["id"]
                if row_id in seen:
                    raise ValueError("duplicate_mmau_row_id")
                if attributes.get("split") != "test-mini":
                    raise ValueError("non_public_mmau_split_denied")
                options = tuple(row["choices"])
                label_index = parse_prefixed_answer_index(row["answer"], options)
                audio = row["context"]["bytes"]
                if not isinstance(audio, bytes) or not audio:
                    raise ValueError("mmau_audio_bytes_missing")
                examples.append(
                    BenchmarkExample(
                        benchmark="mmau-test-mini",
                        row_id=row_id,
                        group=str(attributes.get("task", "unknown")),
                        question=row["instruction"],
                        options=options,
                        label_index=label_index,
                        modality="audio",
                        source_sha256=source_sha,
                        media_bytes_sha256=hashlib.sha256(audio).hexdigest(),
                        media_bytes=audio,
                    )
                )
                seen.add(row_id)
    if len(examples) != MMAU_EXPECTED_ROWS:
        raise ValueError("mmau_public_id_coverage_incomplete")
    return examples


def _task_metadata_paths(config: Mapping[str, Any], metadata_dir: Path) -> dict[str, Path]:
    result = {}
    for task in config["mvbench"]["tasks"]:
        if task["task"] in MVBench_TASKS:
            result[task["sha256"]] = metadata_dir / ("mvbench-json-" + Path(task["file"]).name)
    if len(result) != len(MVBench_TASKS):
        raise ValueError("mvbench_task_metadata_incomplete")
    return result


def _manifest_rows_by_reference(
    *, config: Mapping[str, Any], metadata_dir: Path, manifest_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    expected = public_video_references(dict(config), metadata_dir)
    if len(expected) != len(manifest_rows):
        raise ValueError("mvbench_manifest_reference_count_mismatch")
    bound = []
    for expected_row, manifest_row in zip(expected, manifest_rows, strict=True):
        row_index_value = manifest_row.get("row_index")
        if not isinstance(row_index_value, int):
            raise ValueError("mvbench_row_index_missing")
        identity = {
            "task_json_sha256": manifest_row.get("task_json_sha256"),
            "row_index": row_index_value,
            "video": manifest_row.get("video"),
        }
        if "start" in expected_row or "end" in expected_row:
            identity["start"] = manifest_row.get("start")
            identity["end"] = manifest_row.get("end")
        if identity != expected_row:
            raise ValueError("mvbench_manifest_reference_identity_mismatch")
        frames = manifest_row.get("frames")
        if not isinstance(frames, list) or len(frames) != 16:
            raise ValueError("mvbench_requires_exactly_16_frames")
        indices = [int(frame.get("index")) for frame in frames]
        if list(manifest_row.get("indices", indices)) != indices:
            raise ValueError("mvbench_frame_indices_mismatch")
        for frame in frames:
            if not isinstance(frame.get("png"), str) or not frame["png"]:
                raise ValueError("mvbench_frame_png_path_missing")
            for field in ("png_sha256", "RGB24_sha256"):
                value = frame.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError("mvbench_frame_hash_missing")
        source_sha256 = manifest_row.get("source_sha256")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("mvbench_source_sha256_missing")
        bound.append(
            {
                "task": manifest_row.get("task"),
                "public_row_id": manifest_row.get("public_row_id"),
                "reference": expected_row,
                "media": manifest_row.get("media"),
                "source_sha256": source_sha256,
                "width": manifest_row.get("width"),
                "height": manifest_row.get("height"),
                "frame_indices": indices,
                "frame_png_sha256": [frame["png_sha256"] for frame in frames],
                "frame_rgb24_sha256": [frame["RGB24_sha256"] for frame in frames],
            }
        )
    return expected, hashlib.sha256(canonical_json(bound)).hexdigest()


def load_mvbench_examples(
    *,
    frame_manifest: Path,
    metadata_dir: Path,
    config_path: Path,
) -> list[BenchmarkExample]:
    manifest = json.loads(frame_manifest.read_text())
    if manifest.get("reference_sha256") != MVBench_REFERENCE_SHA256:
        raise ValueError("mvbench_reference_hash_mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != MVBench_EXPECTED_ROWS:
        raise ValueError("mvbench_manifest_row_count_mismatch")
    config = json.loads(config_path.read_text())
    expected_reference, binding_sha256 = _manifest_rows_by_reference(
        config=config, metadata_dir=metadata_dir, manifest_rows=rows
    )
    metadata_paths = _task_metadata_paths(config, metadata_dir)
    metadata: dict[str, list[dict[str, Any]]] = {}
    for sha, path in metadata_paths.items():
        if file_sha256(path) != sha:
            raise ValueError("mvbench_task_json_hash_mismatch")
        task_rows = json.loads(path.read_text())
        if len(task_rows) != 200:
            raise ValueError("mvbench_task_row_count_mismatch")
        metadata[sha] = task_rows
    examples = []
    seen_ids: set[str] = set()
    for row, expected_row in zip(rows, expected_reference, strict=True):
        task = row["task"]
        if task not in MVBench_TASKS:
            raise ValueError("unexpected_mvbench_task")
        public_row_id = row["public_row_id"]
        if public_row_id in seen_ids:
            raise ValueError("duplicate_mvbench_public_row_id")
        if row["task_json_sha256"] != expected_row["task_json_sha256"]:
            raise ValueError("mvbench_task_json_identity_mismatch")
        original = metadata[row["task_json_sha256"]][int(row["row_index"])]
        if original["video"] != row["video"] or original["video"] != expected_row["video"]:
            raise ValueError("mvbench_video_identity_mismatch")
        options = tuple(original["candidates"])
        label_index = parse_prefixed_answer_index(original["answer"], options)
        frames = tuple(
            VideoFrameRef(
                index=int(frame["index"]),
                png=str(frame["png"]),
                png_sha256=str(frame["png_sha256"]),
                rgb24_sha256=str(frame["RGB24_sha256"]),
            )
            for frame in row["frames"]
        )
        examples.append(
            BenchmarkExample(
                benchmark="mvbench-public-9task",
                row_id=public_row_id,
                group=task,
                question=original["question"],
                options=options,
                label_index=label_index,
                modality="video",
                source_sha256=row["source_sha256"],
                video_frames=frames,
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
        seen_ids.add(public_row_id)
    counts = {task: 0 for task in MVBench_TASKS}
    for example in examples:
        counts[example.group] += 1
    if any(count != 200 for count in counts.values()):
        raise ValueError("mvbench_task_denominator_mismatch")
    if not binding_sha256:
        raise ValueError("mvbench_manifest_binding_hash_missing")
    return examples


def summarize_examples(examples: Sequence[BenchmarkExample]) -> dict[str, Any]:
    groups: dict[str, int] = defaultdict(int)
    for example in examples:
        groups[example.group] += 1
    return {
        "rows": len(examples),
        "groups": dict(sorted(groups.items())),
        "row_id_manifest_sha256": hashlib.sha256(
            canonical_json([example.row_id for example in examples])
        ).hexdigest(),
        "mvbench_reference_sha256": MVBench_REFERENCE_SHA256
        if any(example.benchmark == "mvbench-public-9task" for example in examples)
        else None,
        "source_manifest_sha256": hashlib.sha256(
            canonical_json(
                [
                    {
                        "row_id": example.row_id,
                        "source_sha256": example.source_sha256,
                        "media_bytes_sha256": example.media_bytes_sha256,
                        "frame_png_sha256": [frame.png_sha256 for frame in example.video_frames],
                        "frame_rgb24_sha256": [
                            frame.rgb24_sha256 for frame in example.video_frames
                        ],
                    }
                    for example in examples
                ]
            )
        ).hexdigest(),
    }


def _image_rgb_sha256(image: Any) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _load_canonical_video_frames(
    example: BenchmarkExample, cache_root: Path
) -> tuple[list[Any], dict[str, Any]]:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    images = []
    total_bytes = 0
    for frame in example.video_frames:
        path = (cache_root / frame.png).resolve(strict=True)
        if file_sha256(path) != frame.png_sha256:
            raise ValueError("canonical_png_sha256_mismatch")
        image = image_module.open(path).convert("RGB")
        image.load()
        if example.width is not None and example.height is not None:
            if image.size != (example.width, example.height):
                raise ValueError("canonical_png_dimensions_mismatch")
        if _image_rgb_sha256(image) != frame.rgb24_sha256:
            raise ValueError("canonical_png_rgb24_sha256_mismatch")
        total_bytes += path.stat().st_size
        images.append(image)
    return images, {
        "frame_count": len(images),
        "png_bytes": total_bytes,
        "canonical_png_verified": True,
        "linux_mp4_redecode_used": False,
    }


def build_benchmark_processor_inputs(
    forward: TeacherForward,
    example: BenchmarkExample,
    *,
    video_cache_root: Path | None = None,
    temp_parent: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> PreparedBenchmarkInput:
    if example.modality == "audio":
        if example.media_bytes is None:
            raise ValueError("audio benchmark example lacks embedded bytes")
        with tempfile.TemporaryDirectory(prefix="teacher-mmau-", dir=temp_parent) as tmp:
            media = Path(tmp) / "audio.bin"
            media.write_bytes(example.media_bytes)
            with prepared_teacher_input(
                state="",
                question=example.question,
                options=example.options,
                modality="audio",
                media=media,
                temp_parent=Path(tmp),
                ffmpeg_binary=ffmpeg_binary,
            ) as prepared:
                result = build_teacher_processor_inputs(forward.processor, prepared)
                return PreparedBenchmarkInput(
                    prepared=prepared,
                    inputs=result.inputs,
                    processor_summary=result.summary,
                    media_summary={
                        "media_bytes_sha256": example.media_bytes_sha256,
                        "embedded_audio_bytes": len(example.media_bytes),
                        "prepared_pcm_adapter_used": True,
                    },
                )
    if example.modality == "video":
        if video_cache_root is None:
            raise ValueError("video benchmark examples require a canonical PNG cache root")
        images, media_summary = _load_canonical_video_frames(example, video_cache_root)
        content = [{"type": "image", "image": image} for image in images]
        content.append(
            {
                "type": "text",
                "text": official_teacher_prompt("", example.question, example.options),
            }
        )
        prepared = PreparedTeacherInput(
            "video", len(example.options), [{"role": "user", "content": content}], None
        )
        result = build_teacher_processor_inputs(forward.processor, prepared)
        return PreparedBenchmarkInput(
            prepared=prepared,
            inputs=result.inputs,
            processor_summary=result.summary,
            media_summary=media_summary,
        )
    raise ValueError("unsupported benchmark modality")


def _probabilities_by_index(result: TeacherForwardResult, options: Sequence[str]) -> list[float]:
    return [float(result.probabilities[option]) for option in options]


def example_source_identity(example: BenchmarkExample) -> dict[str, Any]:
    return {
        "benchmark": example.benchmark,
        "row_id": example.row_id,
        "row_hash": example.report_id,
        "group": example.group,
        "modality": example.modality,
        "option_count": len(example.options),
        "label_index": example.label_index,
        "source_sha256": example.source_sha256,
        "media_bytes_sha256": example.media_bytes_sha256,
        "video_frame_indices": [frame.index for frame in example.video_frames],
        "video_frame_png_sha256": [frame.png_sha256 for frame in example.video_frames],
        "video_frame_rgb24_sha256": [frame.rgb24_sha256 for frame in example.video_frames],
        "width": example.width,
        "height": example.height,
    }


def source_identity_sha256(example: BenchmarkExample) -> str:
    return hashlib.sha256(canonical_json(example_source_identity(example))).hexdigest()


def processor_identity_sha256(
    processor_summary: Mapping[str, Any], media_summary: Mapping[str, Any]
) -> str:
    identity = {
        "processor_summary": processor_summary,
        "media_summary": media_summary,
        "preprocess_contract": {
            "official_chat_template_path": True,
            "thinking_disabled": True,
            "canonical_png_rgb_verified": media_summary.get("canonical_png_verified", False),
            "audio_pcm_adapter_used": media_summary.get("prepared_pcm_adapter_used", False),
        },
    }
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def _validate_probability_vector(values: Sequence[Any], *, expected_length: int) -> list[float]:
    probabilities = [float(value) for value in values]
    if len(probabilities) != expected_length or not all(
        math.isfinite(value) and value >= 0 for value in probabilities
    ):
        raise ValueError("teacher_probabilities_not_finite")
    total = sum(probabilities)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError("teacher_probabilities_not_normalized")
    return probabilities


def prediction_record(
    *,
    example: BenchmarkExample,
    pass_name: str,
    result: TeacherForwardResult,
    processor_summary: Mapping[str, Any],
    media_summary: Mapping[str, Any],
) -> dict[str, Any]:
    probabilities = _validate_probability_vector(
        _probabilities_by_index(result, example.options), expected_length=len(example.options)
    )
    if not 0 <= result.prediction_index < len(example.options):
        raise ValueError("teacher_prediction_index_outside_options")
    if not math.isfinite(float(result.confidence)):
        raise ValueError("teacher_confidence_not_finite")
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": pass_name,
        "benchmark": example.benchmark,
        "row_id": example.row_id,
        "row_hash": example.report_id,
        "group": example.group,
        "modality": example.modality,
        "option_count": len(example.options),
        "source_identity_sha256": source_identity_sha256(example),
        "processor_identity_sha256": processor_identity_sha256(processor_summary, media_summary),
        "prediction_index": int(result.prediction_index),
        "confidence": float(result.confidence),
        "probabilities": probabilities,
        "correct": int(result.prediction_index) == example.label_index,
        "processor_summary": dict(processor_summary),
        "media_summary": dict(media_summary),
        "payload_values_exported": False,
    }


def _validate_existing_record(
    record: Mapping[str, Any],
    *,
    expected_examples: Mapping[tuple[str, str], BenchmarkExample],
    binding_sha256: str,
    seen: set[tuple[str, str, str]],
) -> dict[str, Any]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resume_record_schema_mismatch")
    if record.get("binding_sha256") != binding_sha256:
        raise ValueError("resume_binding_mismatch")
    pass_name = str(record.get("pass"))
    if pass_name not in {"first", "second"}:
        raise ValueError("resume_record_unknown_pass")
    key = (str(record.get("benchmark")), str(record.get("row_id")))
    example = expected_examples.get(key)
    if example is None:
        raise ValueError("resume_record_not_in_current_cohort")
    full_key = (key[0], key[1], pass_name)
    if full_key in seen:
        raise ValueError("resume_record_duplicate_pass")
    seen.add(full_key)
    if record.get("row_hash") != example.report_id:
        raise ValueError("resume_record_row_hash_mismatch")
    if record.get("group") != example.group or record.get("modality") != example.modality:
        raise ValueError("resume_record_source_identity_mismatch")
    if int(record.get("option_count", -1)) != len(example.options):
        raise ValueError("resume_record_option_count_mismatch")
    if record.get("source_identity_sha256") != source_identity_sha256(example):
        raise ValueError("resume_record_source_identity_mismatch")
    prediction_index = int(record.get("prediction_index", -1))
    if not 0 <= prediction_index < len(example.options):
        raise ValueError("resume_record_prediction_index_mismatch")
    if bool(record.get("correct")) != (prediction_index == example.label_index):
        raise ValueError("resume_record_correct_tampered")
    if not math.isfinite(float(record.get("confidence", float("nan")))):
        raise ValueError("resume_record_confidence_not_finite")
    _validate_probability_vector(
        record.get("probabilities", []), expected_length=len(example.options)
    )
    processor_summary = record.get("processor_summary")
    media_summary = record.get("media_summary")
    if not isinstance(processor_summary, Mapping) or not isinstance(media_summary, Mapping):
        raise ValueError("resume_record_processor_identity_missing")
    if processor_summary.get("contains_payload_values") is not False:
        raise ValueError("resume_record_processor_payload_leak")
    if processor_summary.get("thinking_disabled") is not True:
        raise ValueError("resume_record_processor_identity_mismatch")
    if record.get("processor_identity_sha256") != processor_identity_sha256(
        processor_summary, media_summary
    ):
        raise ValueError("resume_record_processor_identity_mismatch")
    return dict(record)


def compare_two_pass_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    probability_tolerance: float = 0.0,
    benchmark: str | None = None,
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    failures = []
    for record in records:
        key = (str(record["benchmark"]), str(record["row_id"]))
        pass_name = str(record["pass"])
        if pass_name in by_key[key]:
            failures.append({"row_hash": record.get("row_hash"), "reason": "duplicate_pass"})
        by_key[key][pass_name] = record
    if len(by_key) != expected_rows:
        failures.append({"reason": "row_count_mismatch", "observed": len(by_key)})
    correct = 0
    group_totals: dict[str, int] = defaultdict(int)
    group_correct: dict[str, int] = defaultdict(int)
    mismatches = []
    for key in sorted(by_key):
        pair = by_key[key]
        first, second = pair.get("first"), pair.get("second")
        if first is None or second is None:
            failures.append(
                {"row_hash": next(iter(pair.values())).get("row_hash"), "reason": "missing_pass"}
            )
            continue
        if first["prediction_index"] != second["prediction_index"]:
            mismatches.append({"row_hash": first["row_hash"], "reason": "prediction_index"})
        first_probs = list(first["probabilities"])
        second_probs = list(second["probabilities"])
        if len(first_probs) != len(second_probs) or any(
            abs(float(a) - float(b)) > probability_tolerance
            for a, b in zip(first_probs, second_probs, strict=True)
        ):
            mismatches.append({"row_hash": first["row_hash"], "reason": "probabilities"})
        group = str(first["group"])
        group_totals[group] += 1
        correct += int(bool(first["correct"]))
        group_correct[group] += int(bool(first["correct"]))
    macro_scores = {
        group: (group_correct[group] / total if total else 0.0)
        for group, total in sorted(group_totals.items())
    }
    score = correct / expected_rows if expected_rows else 0.0
    gate: dict[str, Any]
    if benchmark == "mmau-test-mini":
        lower = MMAU_SCORE_TARGET - MMAU_SCORE_TOLERANCE
        upper = MMAU_SCORE_TARGET + MMAU_SCORE_TOLERANCE
        gate = {
            "kind": "mmau_teacher_accuracy",
            "target": MMAU_SCORE_TARGET,
            "tolerance": MMAU_SCORE_TOLERANCE,
            "lower_bound": lower,
            "upper_bound": upper,
            "passed": lower <= score <= upper,
        }
    elif benchmark == "mvbench-public-9task":
        gate = {
            "kind": "mvbench_recorded_real_baseline_only",
            "accuracy_target": None,
            "compared_to_adr_0006_0_531": False,
            "passed": True,
        }
    else:
        gate = {"kind": "control_flow_only", "passed": True}
    identical = not failures and not mismatches and len(by_key) == expected_rows
    return {
        "benchmark": benchmark,
        "expected_rows": expected_rows,
        "observed_rows": len(by_key),
        "record_count": len(records),
        "failures": failures,
        "two_pass_mismatches": mismatches,
        "two_pass_identical": identical,
        "micro_correct": correct,
        "micro_total": expected_rows,
        "micro_score": score,
        "macro_scores": macro_scores,
        "macro_score": sum(macro_scores.values()) / len(macro_scores) if macro_scores else 0.0,
        "fixed_denominator_preserved": True,
        "gate": {**gate, "passed": bool(gate["passed"] and identical)},
    }


def compare_benchmark_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows_by_benchmark: Mapping[str, int],
    probability_tolerance: float = 0.0,
    release_or_stage_candidate: bool = False,
) -> dict[str, Any]:
    by_benchmark: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_benchmark[str(record["benchmark"])].append(record)
    benchmarks: dict[str, dict[str, Any]] = {}
    for benchmark, expected_rows in expected_rows_by_benchmark.items():
        benchmarks[benchmark] = compare_two_pass_records(
            by_benchmark.get(benchmark, []),
            expected_rows=expected_rows,
            probability_tolerance=probability_tolerance,
            benchmark=benchmark,
        )
    unknown = sorted(set(by_benchmark) - set(expected_rows_by_benchmark))
    all_gates_passed = not unknown and all(item["gate"]["passed"] for item in benchmarks.values())
    return {
        "benchmarks": benchmarks,
        "unknown_benchmarks": unknown,
        "two_pass_identical": all(item["two_pass_identical"] for item in benchmarks.values())
        and not unknown,
        "release_or_stage_acceptance": bool(release_or_stage_candidate and all_gates_passed),
        "fixed_denominator_preserved": True,
    }


def run_two_pass_forward(
    *,
    forward: TeacherForward,
    examples: Sequence[BenchmarkExample],
    video_cache_root: Path | None,
    records_path: Path,
    binding: Mapping[str, Any],
    probability_tolerance: float = 0.0,
    temp_parent: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
    resume: bool = False,
    release_or_stage_candidate: bool = False,
) -> dict[str, Any]:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    if records_path.exists() and not resume:
        raise ValueError("records_path_exists_without_explicit_resume")
    if not math.isfinite(probability_tolerance) or probability_tolerance < 0:
        raise ValueError("invalid_probability_tolerance")
    binding_sha = hashlib.sha256(canonical_json(binding)).hexdigest()
    expected_examples = {(example.benchmark, example.row_id): example for example in examples}
    if len(expected_examples) != len(examples):
        raise ValueError("duplicate_examples_in_current_cohort")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(
                _validate_existing_record(
                    record,
                    expected_examples=expected_examples,
                    binding_sha256=binding_sha,
                    seen=seen,
                )
            )
    done = {(record["benchmark"], record["row_id"], record["pass"]) for record in records}
    with records_path.open("a", encoding="utf-8") as stream:
        for pass_name in ("first", "second"):
            for example in examples:
                key = (example.benchmark, example.row_id, pass_name)
                if key in done:
                    continue
                prepared = build_benchmark_processor_inputs(
                    forward,
                    example,
                    video_cache_root=video_cache_root,
                    temp_parent=temp_parent,
                    ffmpeg_binary=ffmpeg_binary,
                )
                result = forward.predict_prepared(
                    prepared.prepared, prepared.inputs, example.options
                )
                record = prediction_record(
                    example=example,
                    pass_name=pass_name,
                    result=result,
                    processor_summary=prepared.processor_summary,
                    media_summary=prepared.media_summary,
                )
                record["binding_sha256"] = binding_sha
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                records.append(record)
                done.add(key)
    expected_rows_by_benchmark = {
        benchmark: sum(1 for example in examples if example.benchmark == benchmark)
        for benchmark in sorted({example.benchmark for example in examples})
    }
    summary = compare_benchmark_records(
        records,
        expected_rows_by_benchmark=expected_rows_by_benchmark,
        probability_tolerance=probability_tolerance,
        release_or_stage_candidate=release_or_stage_candidate,
    )
    summary["records_path"] = str(records_path)
    summary["binding_sha256"] = binding_sha
    summary["resume_used"] = resume
    return summary


def report_preamble(*, mode: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "started_utc": datetime.now(UTC).isoformat(),
        "gpu_seconds": 0,
        "sealed_data_read": False,
        "payload_values_exported": False,
        "fixed_denominator_policy": "failures remain in denominator; no row filtering",
    }
