from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.teacher_benchmarks import (
    MVBECH_REFERENCE_SHA256,
    BenchmarkExample,
    TeacherForwardResult,
    VideoFrameRef,
    build_benchmark_processor_inputs,
    compare_two_pass_records,
    load_mmau_examples,
    load_mvbench_examples,
    run_two_pass_forward,
)


class FakeTensor:
    shape = (1, 3)
    dtype = "fake.int64"


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"input_ids": FakeTensor()}


class FakeForward:
    def __init__(self) -> None:
        self.processor = RecordingProcessor()

    def predict_prepared(self, prepared, _inputs, options):
        assert prepared.option_count == len(options)
        probabilities = [0.25, 0.75] if len(options) == 2 else [1 / len(options)] * len(options)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return TeacherForwardResult(
            prediction=options[best],
            prediction_index=best,
            confidence=probabilities[best],
            probabilities=dict(zip(options, probabilities, strict=True)),
        )


def _write_mvbench_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    task_names = [
        "Action Sequence",
        "Action Prediction",
        "Unexpected Action",
        "Object Interaction",
        "Object Shuffle",
        "Action Localization",
        "Action Count",
        "State Change",
        "Character Order",
    ]
    task_files = [name.lower().replace(" ", "_") for name in task_names]
    task_shas = {}
    for task_file in task_files:
        rows = [
            {
                "video": f"{task_file}.mp4",
                "question": "payload question must stay out of reports",
                "candidates": ["left", "right"],
                "answer": "right",
                "start": 0.0,
                "end": 1.0,
            }
            for _ in range(200)
        ]
        path = metadata / f"mvbench-json-{task_file}.json"
        path.write_text(json.dumps(rows))
        task_shas[task_file] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    config = tmp_path / "video-eval-assets.json"
    config.write_text(
        json.dumps(
            {
                "mvbench": {
                    "tasks": [
                        {
                            "task": task,
                            "file": f"json/{task_file}.json",
                            "sha256": task_shas[task_file],
                        }
                        for task, task_file in zip(task_names, task_files, strict=True)
                    ]
                }
            }
        )
    )
    manifest = tmp_path / "frame-manifest.json"
    manifest_rows = []
    for task_index, (task, task_file) in enumerate(zip(task_names, task_files, strict=True)):
        for row_index in range(200):
            manifest_rows.append(
                {
                    "task": task,
                    "task_json_sha256": task_shas[task_file],
                    "row_index": row_index,
                    "public_row_id": f"{task_index}:{row_index}",
                    "video": f"{task_file}.mp4",
                    "source_sha256": "1" * 64,
                    "width": 2,
                    "height": 1,
                    "frames": [],
                }
            )
    manifest.write_text(
        json.dumps({"reference_sha256": MVBECH_REFERENCE_SHA256, "rows": manifest_rows})
    )
    return metadata, config, manifest


def test_synthetic_mvbench_manifest_cannot_self_declare_real_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, config, manifest = _write_mvbench_fixture(tmp_path)
    rows = json.loads(manifest.read_text())["rows"]
    for row in rows:
        row["frames"] = [
            {
                "index": index,
                "png": "cache/abc/0.png",
                "png_sha256": "2" * 64,
                "RGB24_sha256": "3" * 64,
            }
            for index in range(16)
        ]
    manifest.write_text(json.dumps({"reference_sha256": MVBECH_REFERENCE_SHA256, "rows": rows}))
    import eval.teacher_benchmarks as module

    monkeypatch.setattr(
        module,
        "public_video_references",
        lambda _config, _metadata: [
            {
                "task_json_sha256": row["task_json_sha256"],
                "row_index": row["row_index"],
                "video": row["video"],
            }
            for row in rows
        ],
    )

    with pytest.raises(ValueError, match="mvbench_manifest_source_binding_mismatch"):
        load_mvbench_examples(frame_manifest=manifest, metadata_dir=metadata, config_path=config)


def test_canonical_png_read_verifies_sha_and_rgb(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    cache = tmp_path / "cache" / "abc"
    cache.mkdir(parents=True)
    image = image_module.new("RGB", (2, 1), (1, 2, 3))
    frame_path = cache / "0.png"
    image.save(frame_path)
    rgb_hash = __import__("hashlib").sha256(image.tobytes()).hexdigest()
    png_hash = __import__("hashlib").sha256(frame_path.read_bytes()).hexdigest()
    frames = tuple(
        VideoFrameRef(
            index=index,
            png="cache/abc/0.png",
            png_sha256=png_hash,
            rgb24_sha256=rgb_hash,
        )
        for index in range(16)
    )
    example = BenchmarkExample(
        benchmark="fixture-video",
        row_id="row-1",
        group="fixture",
        question="payload question must stay out of summaries",
        options=("left", "right"),
        label_index=1,
        modality="video",
        source_sha256="1" * 64,
        video_frames=frames,
        width=2,
        height=1,
    )

    prepared = build_benchmark_processor_inputs(FakeForward(), example, video_cache_root=tmp_path)

    assert prepared.media_summary["frame_count"] == 16
    assert prepared.media_summary["canonical_png_verified"] is True
    assert "payload question" not in json.dumps(prepared.processor_summary)


def test_real_mvbench_manifest_rejects_source_binding_tamper(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = root / "artifacts/data/research-eval/video-eval-source-metadata"
    config = root / "configs/video-eval-assets.json"
    manifest = root / "artifacts/public-video-protocol-freeze-001/canonical-frame-manifest.json"
    if not (metadata.is_dir() and config.is_file() and manifest.is_file()):
        pytest.skip("public MVBench artifacts are unavailable")
    payload = json.loads(manifest.read_text())
    payload["rows"][0]["source_sha256"] = "f" * 64
    tampered = tmp_path / "canonical-frame-manifest-tampered.json"
    tampered.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="mvbench_manifest_source_binding_mismatch"):
        load_mvbench_examples(frame_manifest=tampered, metadata_dir=metadata, config_path=config)


def test_two_pass_records_are_resumable_and_payload_light(tmp_path: Path) -> None:
    example = BenchmarkExample(
        benchmark="fixture",
        row_id="row-1",
        group="group",
        question="secret prompt text",
        options=("wrong", "right"),
        label_index=1,
        modality="audio",
        source_sha256="a" * 64,
        media_bytes=b"not-real-audio",
        media_bytes_sha256="b" * 64,
    )

    def fake_build(*_args, **_kwargs):
        return SimpleNamespace(
            prepared=SimpleNamespace(modality="audio", option_count=2),
            inputs={"input_ids": FakeTensor()},
            processor_summary={"contains_payload_values": False},
            media_summary={"prepared_pcm_adapter_used": True},
        )

    import eval.teacher_benchmarks as module

    original = module.build_benchmark_processor_inputs
    module.build_benchmark_processor_inputs = fake_build
    try:
        summary = run_two_pass_forward(
            forward=FakeForward(),
            examples=[example],
            video_cache_root=None,
            records_path=tmp_path / "records.jsonl",
            binding={"fixture": True},
        )
    finally:
        module.build_benchmark_processor_inputs = original

    assert summary["two_pass_identical"] is True
    assert summary["benchmarks"]["fixture"]["micro_score"] == 1.0
    assert summary["release_or_stage_acceptance"] is False
    text = (tmp_path / "records.jsonl").read_text()
    assert "secret prompt text" not in text
    assert "right" not in text
    assert "wrong" not in text


def test_compare_two_pass_keeps_failures_in_fixed_denominator() -> None:
    result = compare_two_pass_records(
        [
            {
                "benchmark": "b",
                "row_id": "1",
                "pass": "first",
                "row_hash": "h1",
                "prediction_index": 0,
                "probabilities": [1.0, 0.0],
                "correct": True,
                "group": "g",
            }
        ],
        expected_rows=2,
    )
    assert result["two_pass_identical"] is False
    assert result["micro_total"] == 2
    assert result["fixed_denominator_preserved"] is True


def test_mmau_loader_validates_public_denominator(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    context_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows = []
    for index in range(1000):
        rows.append(
            {
                "context": {"bytes": b"RIFF" + index.to_bytes(2, "little"), "path": None},
                "instruction": f"question {index}",
                "choices": ["(A) no", "(B) yes"],
                "answer": "(B) yes",
                "other_attributes": json.dumps(
                    {"id": f"id-{index}", "split": "test-mini", "task": "sound"}
                ),
            }
        )
    table = pa.table(
        {
            "context": pa.array([row["context"] for row in rows], type=context_type),
            "instruction": [row["instruction"] for row in rows],
            "choices": pa.array([row["choices"] for row in rows], type=pa.list_(pa.string())),
            "answer": [row["answer"] for row in rows],
            "other_attributes": [row["other_attributes"] for row in rows],
        }
    )
    parquet = tmp_path / "mmau.parquet"
    pq.write_table(table, parquet, row_group_size=100)

    examples = load_mmau_examples(
        parquet,
        {
            "bytes": parquet.stat().st_size,
            "sha256": __import__("hashlib").sha256(parquet.read_bytes()).hexdigest(),
        },
    )

    assert len(examples) == 1000
    assert examples[0].label_index == 1
    assert examples[0].modality == "audio"


def test_records_require_explicit_resume_and_validate_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = BenchmarkExample(
        benchmark="fixture",
        row_id="row-1",
        group="group",
        question="payload must not persist",
        options=("wrong", "right"),
        label_index=1,
        modality="audio",
        source_sha256="a" * 64,
        media_bytes=b"not-real-audio",
        media_bytes_sha256="b" * 64,
    )

    def fake_build(*_args, **_kwargs):
        return SimpleNamespace(
            prepared=SimpleNamespace(modality="audio", option_count=2),
            inputs={"input_ids": FakeTensor()},
            processor_summary={
                "contains_payload_values": False,
                "thinking_disabled": True,
            },
            media_summary={"prepared_pcm_adapter_used": True},
        )

    import eval.teacher_benchmarks as module

    monkeypatch.setattr(module, "build_benchmark_processor_inputs", fake_build)
    records = tmp_path / "records.jsonl"
    run_two_pass_forward(
        forward=FakeForward(),
        examples=[example],
        video_cache_root=None,
        records_path=records,
        binding={"fixture": True},
    )
    with pytest.raises(ValueError, match="records_path_exists_without_explicit_resume"):
        run_two_pass_forward(
            forward=FakeForward(),
            examples=[example],
            video_cache_root=None,
            records_path=records,
            binding={"fixture": True},
        )

    lines = [json.loads(line) for line in records.read_text().splitlines()]
    lines[0]["probabilities"] = [float("nan"), 1.0]
    records.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    with pytest.raises(ValueError, match="teacher_probabilities_not_finite"):
        run_two_pass_forward(
            forward=FakeForward(),
            examples=[example],
            video_cache_root=None,
            records_path=records,
            binding={"fixture": True},
            resume=True,
        )


def test_real_forward_limit_guard_writes_static_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.s0_teacher_eval as runner

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mmau": {"file": {}}, "mvbench": {"tasks": []}}))
    parquet = tmp_path / "mmau.parquet"
    parquet.write_bytes(b"placeholder")
    output = tmp_path / "report.json"
    monkeypatch.setattr(runner, "_load_examples", lambda _args, _config: {})

    exit_code = runner.main(
        [
            "--mode",
            "real-forward",
            "--benchmark",
            "all",
            "--config",
            str(config),
            "--mmau-parquet",
            str(parquet),
            "--limit-mmau",
            "1",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["error_code"] == "validation_failed"
    assert "reason" not in report
    assert report["release_or_stage_acceptance"] is False


def test_cpu_preflight_can_use_official_processor_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.s0_teacher_eval as runner

    example = BenchmarkExample(
        benchmark="fixture",
        row_id="row-1",
        group="group",
        question="payload must not persist",
        options=("wrong", "right"),
        label_index=1,
        modality="audio",
        source_sha256="a" * 64,
        media_bytes=b"not-real-audio",
        media_bytes_sha256="b" * 64,
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"mmau": {"file": {}}, "mvbench": {"tasks": []}}))
    parquet = tmp_path / "mmau.parquet"
    parquet.write_bytes(b"placeholder")
    source_dir = tmp_path / "processor-source"
    source_dir.mkdir()
    output = tmp_path / "report.json"

    monkeypatch.setattr(runner, "_load_examples", lambda _args, _config: {"mmau": [example]})
    monkeypatch.setattr(
        runner,
        "validate_teacher_source_tree",
        lambda _source: {"files": {"tokenizer.json": {"sha256": "c" * 64}}},
    )
    monkeypatch.setattr(runner, "load_teacher_processor", lambda _source: RecordingProcessor())

    def fake_build(*_args, **_kwargs):
        return SimpleNamespace(
            processor_summary={
                "contains_payload_values": False,
                "thinking_disabled": True,
                "keys": {"input_ids": {"kind": "tensor", "shape": [1, 3], "dtype": "fake.int64"}},
            },
            media_summary={"prepared_pcm_adapter_used": True},
        )

    monkeypatch.setattr(runner, "build_benchmark_processor_inputs", fake_build)

    exit_code = runner.main(
        [
            "--mode",
            "cpu-preflight",
            "--benchmark",
            "mmau",
            "--config",
            str(config),
            "--mmau-parquet",
            str(parquet),
            "--processor-source-dir",
            str(source_dir),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert exit_code == 0
    assert report["status"] == "passed_cpu_preflight_official_processor_no_model_forward"
    assert report["media_preflight"]["fake_processor_used"] is True
    official = report["official_processor_preflight"]
    assert official["official_processor_used"] is True
    assert official["fake_processor_used"] is False
    assert official["model_forward_run"] is False
    assert official["trust_remote_code"] is False
    assert "payload must not persist" not in output.read_text()


def test_resume_rejects_json_type_coercion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    example = BenchmarkExample(
        benchmark="fixture",
        row_id="row-1",
        group="group",
        question="payload must not persist",
        options=("wrong", "right"),
        label_index=1,
        modality="audio",
        source_sha256="a" * 64,
        media_bytes=b"not-real-audio",
        media_bytes_sha256="b" * 64,
    )

    def fake_build(*_args, **_kwargs):
        return SimpleNamespace(
            prepared=SimpleNamespace(modality="audio", option_count=2),
            inputs={"input_ids": FakeTensor()},
            processor_summary={"contains_payload_values": False, "thinking_disabled": True},
            media_summary={"prepared_pcm_adapter_used": True},
        )

    import eval.teacher_benchmarks as module

    monkeypatch.setattr(module, "build_benchmark_processor_inputs", fake_build)
    records = tmp_path / "records.jsonl"
    run_two_pass_forward(
        forward=FakeForward(),
        examples=[example],
        video_cache_root=None,
        records_path=records,
        binding={"fixture": True},
    )
    lines = [json.loads(line) for line in records.read_text().splitlines()]
    lines[0]["correct"] = 1
    lines[0]["prediction_index"] = "1"
    lines[0]["option_count"] = "2"
    lines[0]["probabilities"] = ["0.25", "0.75"]
    records.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(ValueError):
        run_two_pass_forward(
            forward=FakeForward(),
            examples=[example],
            video_cache_root=None,
            records_path=records,
            binding={"fixture": True},
            resume=True,
        )


def test_runner_code_hashes_are_project_root_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.s0_teacher_eval as runner

    monkeypatch.chdir(tmp_path)
    hashes = runner._code_hashes()

    assert hashes["runner"] != "missing"
    assert hashes["eval_teacher_benchmarks"] != "missing"


def test_missing_code_hash_source_writes_static_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.s0_teacher_eval as runner

    output = tmp_path / "report.json"
    monkeypatch.setattr(runner, "REQUIRED_CODE_HASH_PATHS", {"missing": Path("missing.py")})

    exit_code = runner.main(
        [
            "--mode",
            "cpu-preflight",
            "--benchmark",
            "mmau",
            "--config",
            str(tmp_path / "unused.json"),
            "--mmau-parquet",
            str(tmp_path / "unused.parquet"),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text())
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["error_code"] == "validation_failed"
    assert report["error_type"] == "ValueError"
    assert "reason" not in report
