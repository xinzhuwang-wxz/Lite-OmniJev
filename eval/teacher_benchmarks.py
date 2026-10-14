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

from train.teacher_inference import (
    PreparedTeacherInput,
    TeacherForwardResult,
    build_teacher_processor_inputs,
    official_teacher_prompt,
    prepared_teacher_input,
    validate_teacher_options,
)

SCHEMA_VERSION = 1
MVBECH_REFERENCE_SHA256 = "4530b25b34e464958d26d3f4b01abd73a4ebe53aa287702c5206048524e43efb"
MMAU_EXPECTED_ROWS = 1000
MVBench_EXPECTED_ROWS = 1800
MVBench_TASKS = (
    "Action Sequence",
    "Action Prediction",
    "Unexpected Action",
    "Object Interaction",
    "Object Shuffle",
    "Action Localization",
    "Action Count",
    "State Change",
    "Character Order",
)


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


def load_mvbench_examples(
    *,
    frame_manifest: Path,
    metadata_dir: Path,
    config_path: Path,
) -> list[BenchmarkExample]:
    manifest = json.loads(frame_manifest.read_text())
    if manifest.get("reference_sha256") != MVBECH_REFERENCE_SHA256:
        raise ValueError("mvbench_reference_hash_mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != MVBench_EXPECTED_ROWS:
        raise ValueError("mvbench_manifest_row_count_mismatch")
    config = json.loads(config_path.read_text())
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
    for row in rows:
        task = row["task"]
        if task not in MVBench_TASKS:
            raise ValueError("unexpected_mvbench_task")
        public_row_id = row["public_row_id"]
        if public_row_id in seen_ids:
            raise ValueError("duplicate_mvbench_public_row_id")
        original = metadata[row["task_json_sha256"]][int(row["row_index"])]
        if original["video"] != row["video"]:
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
        if len(frames) != 16:
            raise ValueError("mvbench_requires_exactly_16_frames")
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


def prediction_record(
    *,
    example: BenchmarkExample,
    pass_name: str,
    result: TeacherForwardResult,
    processor_summary: Mapping[str, Any],
    media_summary: Mapping[str, Any],
) -> dict[str, Any]:
    probabilities = _probabilities_by_index(result, example.options)
    if len(probabilities) != len(example.options) or not all(
        math.isfinite(value) and value >= 0 for value in probabilities
    ):
        raise ValueError("teacher_probabilities_not_finite")
    total = sum(probabilities)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError("teacher_probabilities_not_normalized")
    if not 0 <= result.prediction_index < len(example.options):
        raise ValueError("teacher_prediction_index_outside_options")
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": pass_name,
        "benchmark": example.benchmark,
        "row_id": example.row_id,
        "row_hash": example.report_id,
        "group": example.group,
        "modality": example.modality,
        "option_count": len(example.options),
        "prediction_index": int(result.prediction_index),
        "confidence": float(result.confidence),
        "probabilities": probabilities,
        "correct": int(result.prediction_index) == example.label_index,
        "processor_summary": dict(processor_summary),
        "media_summary": dict(media_summary),
        "payload_values_exported": False,
    }


def compare_two_pass_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    probability_tolerance: float = 0.0,
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
    return {
        "expected_rows": expected_rows,
        "observed_rows": len(by_key),
        "record_count": len(records),
        "failures": failures,
        "two_pass_mismatches": mismatches,
        "two_pass_identical": not failures and not mismatches and len(by_key) == expected_rows,
        "micro_correct": correct,
        "micro_total": expected_rows,
        "micro_score": correct / expected_rows if expected_rows else 0.0,
        "macro_scores": macro_scores,
        "macro_score": sum(macro_scores.values()) / len(macro_scores) if macro_scores else 0.0,
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
) -> dict[str, Any]:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    binding_sha = hashlib.sha256(canonical_json(binding)).hexdigest()
    records: list[dict[str, Any]] = []
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("binding_sha256") != binding_sha:
                raise ValueError("resume_binding_mismatch")
            records.append(record)
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
    summary = compare_two_pass_records(
        records, expected_rows=len(examples), probability_tolerance=probability_tolerance
    )
    summary["records_path"] = str(records_path)
    summary["binding_sha256"] = binding_sha
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
