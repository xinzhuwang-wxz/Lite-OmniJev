import json
from pathlib import Path

import pytest

from data.schema import HardTarget, Sample
from scripts.admit_clinc import (
    EXPECTED_LABELS,
    EXPECTED_TRAIN_SHA256,
    build_samples,
    label_names_from_hf_metadata,
    main,
    option_group,
    require_token_probe,
    run,
)


def labels() -> list[str]:
    values = [f"intent_{index:03d}" for index in range(EXPECTED_LABELS)]
    values[42] = "oos"
    return values


def test_label_metadata_must_have_all_class_names() -> None:
    payload = {"info": {"features": {"intent": {"names": labels()}}}}
    assert label_names_from_hf_metadata({b"huggingface": json.dumps(payload).encode()})[42] == "oos"
    with pytest.raises(ValueError, match="151 labels"):
        label_names_from_hf_metadata(
            {b"huggingface": json.dumps({"info": {"features": {"intent": {"names": []}}}}).encode()}
        )


def test_option_groups_are_contiguous_and_within_choice_limit() -> None:
    all_labels = labels()
    assert option_group(0, all_labels) == [f"intent {index:03d}" for index in range(20)]
    assert option_group(150, all_labels) == [f"intent {index:03d}" for index in range(140, 151)]
    assert 2 <= len(option_group(150, all_labels)) <= 20


def test_build_samples_preserves_human_label_and_keeps_oos_as_option() -> None:
    all_labels = labels()
    rows = [
        {"text": "transfer money please", "intent": 1},
        {"text": "this is out of scope", "intent": 42},
        {"text": "transfer   money please", "intent": 1},
    ]
    samples, report = build_samples(rows, all_labels, source_sha256="a" * 64)
    assert report["samples_written"] == 2
    assert report["input_rows"] == 3
    assert report["duplicate_exact_texts_dropped"] == 1
    assert report["oos_treatment"]["not_mapped_to"] == "U"
    assert report["oos_treatment"]["mapped_to"] == "ordinary out-of-scope intent choice option"
    assert report["raw_labels_original"] == all_labels
    assert report["semantic_review"]["human_semantic_review_performed"] is False
    oos_question = samples[1].questions[0]
    assert oos_question.options is not None
    assert isinstance(oos_question.target, HardTarget)
    assert "oos" in oos_question.options
    assert oos_question.target.value == "C"
    for sample in samples:
        assert Sample.model_validate_json(sample.model_dump_json()) == sample
        assert isinstance(sample.questions[0].target, HardTarget)
        assert sample.questions[0].target.source == "clinc_human_label"
        assert sample.meta.pool == "train"


def test_build_samples_rejects_duplicate_text_with_conflicting_intent() -> None:
    with pytest.raises(ValueError, match="conflicting intents"):
        build_samples(
            [
                {"text": "same utterance", "intent": 1},
                {"text": "same   utterance", "intent": 2},
            ],
            labels(),
            source_sha256="a" * 64,
        )


def test_build_samples_validates_bad_duplicate_intent_before_dedup() -> None:
    with pytest.raises(ValueError, match="row 1 intent id out of range"):
        build_samples(
            [
                {"text": "same utterance", "intent": 1},
                {"text": "same   utterance", "intent": 999},
            ],
            labels(),
            source_sha256="a" * 64,
        )


def test_build_samples_counts_generator_rows_during_iteration() -> None:
    row_iter = ({"text": f"utterance {index}", "intent": index} for index in range(3))
    samples, report = build_samples(row_iter, labels(), source_sha256="a" * 64)
    assert len(samples) == 3
    assert report["input_rows"] == 3


def test_bad_intent_and_missing_file_block_without_admission(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_samples([{"text": "bad", "intent": 999}], labels(), source_sha256="a" * 64)

    report = tmp_path / "report.json"
    code = main(["--input", str(tmp_path / "missing.parquet"), "--report", str(report)])
    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["training_admitted"] is False
    assert payload["expected_input_sha256"] == EXPECTED_TRAIN_SHA256
    assert payload["exit_code"] == 1


def test_token_probe_is_required_before_training_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "admitted.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_bytes(b"not actually parquet")

    monkeypatch.setattr("scripts.admit_clinc.sha256_file", lambda _path: EXPECTED_TRAIN_SHA256)
    monkeypatch.setattr(
        "scripts.admit_clinc.read_clinc_parquet",
        lambda _path: ([{"text": "transfer money", "intent": 1}], labels()),
    )

    code = run(
        input_path,
        output_path,
        report_path,
        model_dir=None,
        command=["admit_clinc", "--input", str(input_path)],
    )

    payload = json.loads(report_path.read_text())
    assert code == 1
    assert not output_path.exists()
    assert payload["status"] == "blocked"
    assert payload["training_admitted"] is False
    assert "Missing required --model-dir" in payload["blocker"]
    assert payload["token_probe"]["required"] is True
    assert payload["output"] is None


def test_failed_token_probe_blocks_training_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "admitted.jsonl"
    report_path = tmp_path / "report.json"
    model_dir = tmp_path / "model"
    input_path.write_bytes(b"not actually parquet")
    model_dir.mkdir()

    monkeypatch.setattr("scripts.admit_clinc.sha256_file", lambda _path: EXPECTED_TRAIN_SHA256)
    monkeypatch.setattr(
        "scripts.admit_clinc.read_clinc_parquet",
        lambda _path: ([{"text": "transfer money", "intent": 1}], labels()),
    )
    monkeypatch.setattr(
        "scripts.admit_clinc.probe_answer_tokens",
        lambda _model_dir: {"status": "failed", "exit_code": 1, "failures": ["A boundary failed"]},
    )

    code = run(
        input_path,
        output_path,
        report_path,
        model_dir=model_dir,
        command=["admit_clinc", "--model-dir", str(model_dir)],
    )

    payload = json.loads(report_path.read_text())
    assert code == 1
    assert not output_path.exists()
    assert payload["training_admitted"] is False
    assert "official tokenizer answer-token probe failed" in payload["blocker"]
    assert payload["token_probe"]["status"] is None


def test_token_probe_success_records_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    token_report = {
        "status": "passed",
        "exit_code": 0,
        "model_dir": "/fixed/e4b",
        "template_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "answer_count": 33,
        "scope": "CPU text tokenizer/template only",
    }

    monkeypatch.setattr("scripts.admit_clinc.probe_answer_tokens", lambda _model_dir: token_report)
    assert require_token_probe(Path("/fixed/e4b")) == token_report
