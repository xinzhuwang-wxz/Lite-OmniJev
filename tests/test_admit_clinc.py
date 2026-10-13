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
    assert report["duplicate_exact_texts_dropped"] == 1
    assert report["oos_treatment"]["not_mapped_to"] == "U"
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
