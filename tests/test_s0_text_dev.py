import hashlib
import json
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts import s0_text_dev as text_dev


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sample(index: int, *, family_id: str | None = None, sample_id: str | None = None) -> Sample:
    option = "A" if index % 2 == 0 else "B"
    return Sample.model_validate(
        {
            "id": sample_id or f"clinc-oos-train-{index:05d}",
            "prefix": {"state": f"utterance {index}"},
            "questions": [
                {
                    "qid": "intent",
                    "type": "choice",
                    "instruction": "Which intent best matches the user's utterance?",
                    "options": ["banking", "travel"],
                    "target": {
                        "kind": "hard",
                        "value": option,
                        "source": "clinc_human_label",
                    },
                }
            ],
            "meta": {
                "modality": ["text"],
                "source_dataset": text_dev.REPO_ID,
                "source_revision": text_dev.REVISION,
                "source_sha256": text_dev.EXPECTED_TRAIN_SHA256,
                "license": text_dev.LICENSE,
                "usage": "commercial",
                "pool": "train",
                "family_id": family_id or digest(f"family-{index}"),
                "data_version": text_dev.DATA_VERSION,
            },
        }
    )


def write_source(path: Path, samples: list[Sample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(row.model_dump_json() + "\n" for row in samples)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def run_builder(
    tmp_path: Path, rows: list[Sample], *, dev_count: int = 3
) -> tuple[int, dict[str, Any]]:
    source = tmp_path / "admitted.jsonl"
    source_sha = write_source(source, rows)
    code = text_dev.main(
        [
            "--source",
            str(source),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--train-out",
            str(tmp_path / "train.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(tmp_path / "allocation.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-source-sha256",
            source_sha,
            "--expected-count",
            str(len(rows)),
            "--expected-raw-source-sha256",
            text_dev.EXPECTED_TRAIN_SHA256,
            "--dev-count",
            str(dev_count),
        ]
    )
    return code, json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_builds_deterministic_dev_train_and_ledgers(tmp_path: Path) -> None:
    rows = [sample(index) for index in range(8)]
    code, report = run_builder(tmp_path, rows)
    assert code == 0
    assert report["status"] == "passed"
    assert report["training_admitted"] is False
    assert report["quality_gate"] == "not_run"
    assert report["gpu_used"] is False
    assert report["holdout_accessed"] is False
    assert report["cost"] is None
    assert report["primitive_scope"] == {
        "choice_only": True,
        "score_covered": False,
        "noul_covered": False,
    }
    assert report["split"]["dev_items"] == 3
    assert report["split"]["train_items"] == 5
    assert report["supersedes"] == {
        "source_data_version": text_dev.DATA_VERSION,
        "superseded_for_training": True,
    }

    dev_rows = read_jsonl(tmp_path / "dev.jsonl")
    train_rows = read_jsonl(tmp_path / "train.jsonl")
    dev_ids = {row["id"] for row in dev_rows}
    train_ids = {row["id"] for row in train_rows}
    assert len(dev_rows) == 3
    assert len(train_rows) == 5
    assert dev_ids.isdisjoint(train_ids)
    assert {row["meta"]["family_id"] for row in dev_rows}.isdisjoint(
        {row["meta"]["family_id"] for row in train_rows}
    )

    source_by_id = {row.id: row.model_dump(mode="json") for row in rows}
    for derived in [*dev_rows, *train_rows]:
        original = source_by_id[derived["id"]]
        expected_pool = "dev" if derived["id"] in dev_ids else "train"
        expected_version = (
            text_dev.TEXT_DEV_VERSION if expected_pool == "dev" else text_dev.TEXT_TRAIN_VERSION
        )
        original["meta"]["pool"] = expected_pool
        original["meta"]["data_version"] = expected_version
        assert derived == original

    dev_ledger = read_jsonl(tmp_path / "dev-ledger.jsonl")
    allocation = read_jsonl(tmp_path / "allocation.jsonl")
    assert len(dev_ledger) == 3
    assert len(allocation) == 8
    encoded_ledgers = json.dumps({"dev": dev_ledger, "allocation": allocation})
    for forbidden in ["utterance", "target", "options", "prompt", "question"]:
        assert forbidden not in encoded_ledgers
    for row in dev_ledger:
        assert "label" not in row
    assert {row["modality"] for row in dev_ledger} == {"text"}
    assert {row["label_origin"] for row in dev_ledger} == {"hard_gold"}
    assert {row["source_dataset"] for row in dev_ledger} == {text_dev.REPO_ID}

    second = tmp_path / "second"
    second.mkdir()
    code2, report2 = run_builder(second, rows)
    assert code2 == 0
    assert report2["outputs"] == report["outputs"]
    assert (second / "dev.jsonl").read_bytes() == (tmp_path / "dev.jsonl").read_bytes()
    assert (second / "train.jsonl").read_bytes() == (tmp_path / "train.jsonl").read_bytes()
    assert (second / "dev-ledger.jsonl").read_bytes() == (
        tmp_path / "dev-ledger.jsonl"
    ).read_bytes()
    assert (second / "allocation.jsonl").read_bytes() == (
        tmp_path / "allocation.jsonl"
    ).read_bytes()


def test_rejects_source_sha_tamper_before_outputs(tmp_path: Path) -> None:
    source = tmp_path / "admitted.jsonl"
    write_source(source, [sample(index) for index in range(5)])
    code = text_dev.main(
        [
            "--source",
            str(source),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--train-out",
            str(tmp_path / "train.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(tmp_path / "allocation.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-source-sha256",
            "0" * 64,
            "--expected-count",
            "5",
            "--dev-count",
            "3",
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["status"] == "blocked"
    assert report["blocker"] == "unexpected admitted JSONL sha256"
    assert not (tmp_path / "dev.jsonl").exists()


def test_rejects_duplicate_sample_id(tmp_path: Path) -> None:
    rows = [sample(0), sample(1, sample_id="clinc-oos-train-00000"), sample(2)]
    code, report = run_builder(tmp_path, rows, dev_count=2)
    assert code == 1
    assert report["status"] == "blocked"
    assert not (tmp_path / "train.jsonl").exists()


def test_rejects_duplicate_family_in_fixed_source(tmp_path: Path) -> None:
    shared_family = digest("shared-family")
    rows = [sample(0, family_id=shared_family), sample(1, family_id=shared_family), sample(2)]
    code, report = run_builder(tmp_path, rows, dev_count=2)
    assert code == 1
    assert report["status"] == "blocked"


def test_rejects_unexpected_pool_revision_license_and_source_hash(tmp_path: Path) -> None:
    bad = sample(0).model_dump(mode="json")
    bad["meta"]["pool"] = "dev"
    with_pool = Sample.model_validate(bad)
    code, _report = run_builder(tmp_path / "pool", [with_pool, sample(1), sample(2)], dev_count=2)
    assert code == 1

    bad = sample(0).model_dump(mode="json")
    bad["meta"]["source_revision"] = "other"
    with_revision = Sample.model_validate(bad)
    code, _report = run_builder(
        tmp_path / "revision", [with_revision, sample(1), sample(2)], dev_count=2
    )
    assert code == 1

    bad = sample(0).model_dump(mode="json")
    bad["meta"]["source_sha256"] = "f" * 64
    with_source_hash = Sample.model_validate(bad)
    code, _report = run_builder(
        tmp_path / "source", [with_source_hash, sample(1), sample(2)], dev_count=2
    )
    assert code == 1


def test_rejects_insufficient_families_and_existing_outputs(tmp_path: Path) -> None:
    code, report = run_builder(tmp_path / "small", [sample(0), sample(1)], dev_count=3)
    assert code == 1
    assert report["status"] == "blocked"

    out = tmp_path / "overwrite"
    out.mkdir()
    (out / "dev.jsonl").write_text("exists", encoding="utf-8")
    source = out / "admitted.jsonl"
    source_sha = write_source(source, [sample(index) for index in range(4)])
    code = text_dev.main(
        [
            "--source",
            str(source),
            "--dev-out",
            str(out / "dev.jsonl"),
            "--train-out",
            str(out / "train.jsonl"),
            "--dev-ledger-out",
            str(out / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(out / "allocation.jsonl"),
            "--report",
            str(out / "report.json"),
            "--expected-source-sha256",
            source_sha,
            "--expected-count",
            "4",
            "--dev-count",
            "2",
        ]
    )
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["status"] == "blocked"
    assert (out / "dev.jsonl").read_text(encoding="utf-8") == "exists"


def test_reports_real_intent_display_coverage_by_pool(tmp_path: Path) -> None:
    rows = [sample(index) for index in range(10)]
    code, report = run_builder(tmp_path, rows, dev_count=4)
    assert code == 0
    assert report["source"]["intent_display_counts"] == {"banking": 5, "travel": 5}
    assert report["source"]["intent_display_coverage"] == 2
    by_pool = report["split"]["intent_display_counts_by_pool"]
    assert sum(by_pool["dev"].values()) == 4
    assert sum(by_pool["train"].values()) == 6
    assert set(by_pool["dev"]) | set(by_pool["train"]) == {"banking", "travel"}


def test_rejects_output_path_alias_without_writing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "admitted.jsonl"
    source_sha = write_source(source, [sample(index) for index in range(4)])
    same = tmp_path / "same.jsonl"
    code = text_dev.main(
        [
            "--source",
            str(source),
            "--dev-out",
            str(same),
            "--train-out",
            str(same),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(tmp_path / "allocation.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-source-sha256",
            source_sha,
            "--expected-count",
            "4",
            "--dev-count",
            "2",
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["blocker"] == "output paths must be unique"
    assert not same.exists()


def test_report_path_existing_or_alias_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "admitted.jsonl"
    source_sha = write_source(source, [sample(index) for index in range(4)])
    report_path = tmp_path / "report.json"
    report_path.write_text("existing evidence", encoding="utf-8")
    code = text_dev.main(
        [
            "--source",
            str(source),
            "--dev-out",
            str(tmp_path / "dev.jsonl"),
            "--train-out",
            str(tmp_path / "train.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(tmp_path / "allocation.jsonl"),
            "--report",
            str(report_path),
            "--expected-source-sha256",
            source_sha,
            "--expected-count",
            "4",
            "--dev-count",
            "2",
        ]
    )
    assert code == 1
    assert report_path.read_text(encoding="utf-8") == "existing evidence"
    assert not (tmp_path / "dev.jsonl").exists()

    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias_source = alias_dir / "admitted.jsonl"
    alias_sha = write_source(alias_source, [sample(index) for index in range(4)])
    aliased_report = alias_dir / "dev.jsonl"
    code = text_dev.main(
        [
            "--source",
            str(alias_source),
            "--dev-out",
            str(aliased_report),
            "--train-out",
            str(alias_dir / "train.jsonl"),
            "--dev-ledger-out",
            str(alias_dir / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(alias_dir / "allocation.jsonl"),
            "--report",
            str(aliased_report),
            "--expected-source-sha256",
            alias_sha,
            "--expected-count",
            "4",
            "--dev-count",
            "2",
        ]
    )
    assert code == 1
    assert not aliased_report.exists()


def test_rejects_dangling_output_symlink_and_bad_dev_count(tmp_path: Path) -> None:
    source = tmp_path / "admitted.jsonl"
    source_sha = write_source(source, [sample(index) for index in range(4)])
    dangling = tmp_path / "dangling.jsonl"
    dangling.symlink_to(tmp_path / "missing-target.jsonl")
    code = text_dev.main(
        [
            "--source",
            str(source),
            "--dev-out",
            str(dangling),
            "--train-out",
            str(tmp_path / "train.jsonl"),
            "--dev-ledger-out",
            str(tmp_path / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(tmp_path / "allocation.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--expected-source-sha256",
            source_sha,
            "--expected-count",
            "4",
            "--dev-count",
            "2",
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["blocker"] == "refusing to overwrite existing outputs"
    assert dangling.is_symlink()

    bad = tmp_path / "bad-count"
    bad.mkdir()
    bad_source = bad / "admitted.jsonl"
    bad_sha = write_source(bad_source, [sample(index) for index in range(4)])
    code = text_dev.main(
        [
            "--source",
            str(bad_source),
            "--dev-out",
            str(bad / "dev.jsonl"),
            "--train-out",
            str(bad / "train.jsonl"),
            "--dev-ledger-out",
            str(bad / "dev-ledger.jsonl"),
            "--allocation-ledger-out",
            str(bad / "allocation.jsonl"),
            "--report",
            str(bad / "report.json"),
            "--expected-source-sha256",
            bad_sha,
            "--expected-count",
            "4",
            "--dev-count",
            "4",
        ]
    )
    report = json.loads((bad / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    assert report["blocker"] == "dev_count must be between 1 and expected_count - 1"
    assert not (bad / "dev.jsonl").exists()
