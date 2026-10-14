import json
from pathlib import Path

import pytest

from data.schema import Sample
from scripts.s1_dev_eval import load_samples, metrics_from_records
from scripts.train_s1 import load_ledger_pools, verify_data_is_train_pool
from train.trainer import LoopConfig, smoke_samples


def sample_row(index: int, *, pool: str = "train", qtype: str = "choice") -> dict:
    question: dict = {
        "qid": "intent",
        "type": qtype,
        "instruction": "Which intent best matches the user's utterance?",
    }
    if qtype == "choice":
        question["options"] = [f"intent {i:02d}" for i in range(20)]
        question["target"] = {"kind": "hard", "value": "B", "source": "clinc_human_label"}
    else:
        question["score_levels"] = 10
        question["target"] = {"kind": "hard", "value": 5, "source": "clinc_human_label"}
    return {
        "id": f"clinc-oos-train-{index:05d}",
        "prefix": {"state": f"utterance number {index}", "media": []},
        "questions": [question],
        "meta": {
            "modality": ["text"],
            "source_dataset": "clinc/clinc_oos",
            "source_revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
            "source_sha256": "a" * 64,
            "license": "CC-BY-3.0",
            "usage": "commercial",
            "pool": pool,
            "family_id": json.dumps(index),
            "data_version": "clinc-oos-plus-train-v1-s1-pools-v1",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_loop_config_rejects_midway_beyond_steps() -> None:
    config = LoopConfig(sample_count=2000, steps=300, accumulation=10, midway_step=300)
    assert config.midway_step == 300
    with pytest.raises(ValueError, match="midway_step"):
        LoopConfig(sample_count=2000, steps=300, accumulation=10, midway_step=301)
    with pytest.raises(ValueError, match="learning_rate"):
        LoopConfig(sample_count=2000, steps=300, accumulation=10, learning_rate=float("nan"))


def test_smoke_samples_enforces_frozen_data_version(tmp_path: Path) -> None:
    rows = [sample_row(i) for i in range(4)]
    data = write_jsonl(tmp_path / "data.jsonl", rows)
    config = LoopConfig(sample_count=4, steps=2, accumulation=2)
    samples = smoke_samples(
        config, data, expected_data_version="clinc-oos-plus-train-v1-s1-pools-v1"
    )
    assert [s.id for s in samples] == [row["id"] for row in rows]
    tampered = [dict(row, meta=dict(row["meta"], data_version="other-v9")) for row in rows]
    with pytest.raises(ValueError, match="data_version"):
        smoke_samples(
            config,
            write_jsonl(tmp_path / "bad.jsonl", tampered),
            expected_data_version="clinc-oos-plus-train-v1-s1-pools-v1",
        )


def test_ledger_verification_rejects_reserved_and_unlisted(tmp_path: Path) -> None:
    ledger = write_jsonl(
        tmp_path / "ledger.jsonl",
        [
            {"item_id": "clinc-oos-train-00000", "pool": "train"},
            {"item_id": "clinc-oos-train-00001", "pool": "holdout"},
            {"item_id": "clinc-oos-train-00002", "pool": "calibration"},
            {"item_id": "clinc-oos-train-00003", "pool": "dev"},
        ],
    )
    assert load_ledger_pools(ledger)["clinc-oos-train-00001"] == "holdout"
    clean = write_jsonl(tmp_path / "clean.jsonl", [sample_row(0)])
    assert verify_data_is_train_pool(clean, ledger)["verified_train"] == 1
    for index, pattern in ((1, "reserved"), (2, "reserved"), (3, "dev")):
        bad = write_jsonl(tmp_path / f"bad-{index}.jsonl", [sample_row(index)])
        with pytest.raises(ValueError, match=pattern):
            verify_data_is_train_pool(bad, ledger)
    unlisted = write_jsonl(tmp_path / "unlisted.jsonl", [sample_row(99)])
    with pytest.raises(ValueError, match="unlisted"):
        verify_data_is_train_pool(unlisted, ledger)


def test_dev_eval_load_samples_rejects_non_choice(tmp_path: Path) -> None:
    data = write_jsonl(tmp_path / "dev.jsonl", [sample_row(0)])
    samples = load_samples(data)
    assert len(samples) == 1
    score_rows = [sample_row(1, qtype="score")]
    with pytest.raises(ValueError, match="Choice"):
        load_samples(write_jsonl(tmp_path / "score.jsonl", score_rows))


def test_metrics_from_records_matches_product_semantics() -> None:
    def record(index: int, p_a: float, target: str, prediction: str) -> dict:
        p_b = 1.0 - p_a
        return {
            "probabilities": {"A": p_a, "B": p_b},
            "prediction": prediction,
            "target": target,
            "p_insufficient": 0.01 * index,
        }

    records = [
        record(0, 0.9, "A", "A"),
        record(1, 0.6, "A", "A"),
        record(2, 0.4, "B", "B"),
        record(3, 0.8, "B", "A"),
    ]
    metrics = metrics_from_records(records)
    assert metrics["product_accuracy"] == 0.75
    assert metrics["accuracy"] == 0.75
    assert metrics["count"] == 4
    assert metrics["mean_p_insufficient"] == pytest.approx(0.015)
    assert metrics["max_p_insufficient"] == pytest.approx(0.03)
    assert metrics["predicted_insufficient"] == 0
    assert 0.0 <= metrics["ece15"] <= 1.0
    assert 0.0 <= metrics["coverage_at_error"] <= 1.0


def test_sample_schema_accepts_ledger_style_family(tmp_path: Path) -> None:
    rows = [sample_row(0)]
    samples = [Sample.model_validate_json(json.dumps(row)) for row in rows]
    assert samples[0].meta.pool == "train"


def test_repeat_identical_handles_single_pass() -> None:
    from scripts.s1_dev_eval import repeat_identical_over

    one: list[list[dict]] = [[{"id": "a"}]]
    assert repeat_identical_over(one) is None
    assert (
        repeat_identical_over(
            one,
        )
        is None
    )
    assert repeat_identical_over([]) is None
    two_same: list[list[dict]] = [[{"id": "a", "p": 0.5}], [{"id": "a", "p": 0.5}]]
    assert repeat_identical_over(two_same) is True
    two_diff: list[list[dict]] = [[{"id": "a", "p": 0.5}], [{"id": "a", "p": 0.4}]]
    assert repeat_identical_over(two_diff) is False
