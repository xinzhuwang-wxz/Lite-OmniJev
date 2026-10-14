import json
from pathlib import Path

import pytest

from data.schema import Sample
from scripts.s1_text_pools import (
    SOURCE_DATA_VERSION,
    canonical_family_hash,
    main,
    sha256_bytes,
)

POOLS_VERSION = "clinc-oos-plus-train-v1-s1-pools-v1"


def sample_row(index: int, *, family_id: str | None = None) -> dict:
    return {
        "id": f"clinc-oos-train-{index:05d}",
        "prefix": {"state": f"utterance number {index}", "media": []},
        "questions": [
            {
                "qid": "intent",
                "type": "choice",
                "instruction": "Which intent best matches the user's utterance?",
                "options": [f"intent {i:02d}" for i in range(20)],
                "target": {"kind": "hard", "value": "B", "source": "clinc_human_label"},
            }
        ],
        "meta": {
            "modality": ["text"],
            "source_dataset": "clinc/clinc_oos",
            "source_revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
            "source_sha256": "a" * 64,
            "license": "CC-BY-3.0",
            "usage": "commercial",
            "pool": "train",
            "family_id": family_id or sha256_bytes(f"family-{index}".encode()),
            "data_version": SOURCE_DATA_VERSION,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode())


def make_dev(tmp_path: Path, families: list[str], monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [sample_row(9000 + i, family_id=family) for i, family in enumerate(families)]
    dev = tmp_path / "dev.jsonl"
    digest = write_jsonl(dev, rows)
    monkeypatch.setattr("scripts.s1_text_pools.EXPECTED_DEV_SHA256", digest)
    return dev


def run_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_rows: list[dict],
    expected_source_sha256: str | None = None,
    dev_families: list[str] | None = None,
    tag: str = "a",
    holdout_count: int = 2,
    calibration_count: int = 3,
) -> tuple[int, Path]:
    source = tmp_path / f"source-{tag}.jsonl"
    digest = write_jsonl(source, source_rows)
    dev = make_dev(tmp_path, dev_families or ["d" * 64], monkeypatch)
    out = tmp_path / f"out-{tag}"
    argv = [
        "--source",
        str(source),
        "--dev",
        str(dev),
        "--train-out",
        str(out / "train.jsonl"),
        "--calibration-out",
        str(out / "calibration.jsonl"),
        "--holdout-out",
        str(out / "holdout.jsonl"),
        "--allocation-ledger-out",
        str(out / "ledger.jsonl"),
        "--report",
        str(out / "report.json"),
        "--expected-source-sha256",
        expected_source_sha256 or digest,
        "--expected-count",
        str(len(source_rows)),
        "--holdout-count",
        str(holdout_count),
        "--calibration-count",
        str(calibration_count),
    ]
    return main(argv), out / "report.json"


def test_split_is_deterministic_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [sample_row(i) for i in range(10)]
    code, report_path = run_split(tmp_path, monkeypatch, source_rows=rows, tag="a")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0 and report["status"] == "passed"
    assert report["split"] == {
        "train_items": 5,
        "calibration_reserve_items": 3,
        "holdout_reserve_items": 2,
        "dev_family_overlap": 0,
        "exclusion_family_overlap": None,
    }
    out = report_path.parent
    pools: dict[str, set[str]] = {}
    for name, count in (
        ("train", 5),
        ("calibration", 3),
        ("holdout", 2),
    ):
        lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == count
        samples = [Sample.model_validate_json(line) for line in lines]
        pools[name] = {s.meta.family_id for s in samples}
        for sample in samples:
            assert sample.meta.pool == name if name != "train" else sample.meta.pool == "train"
            assert sample.meta.data_version == POOLS_VERSION
            assert sample.meta.license == "CC-BY-3.0"
    assert not (pools["train"] & pools["calibration"])
    assert not (pools["train"] & pools["holdout"])
    assert not (pools["calibration"] & pools["holdout"])
    ledger = [
        json.loads(line) for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger) == 10
    assert {row["pool"] for row in ledger} == {"train", "calibration", "holdout"}
    for row in ledger:
        assert row["allocation_key_sha256"] == sha256_bytes(
            f"s1-text-pools-v1:42:{row['family_id']}".encode()
        )
    code_b, report_b = run_split(tmp_path, monkeypatch, source_rows=rows, tag="b")
    assert code_b == 0
    out_b = report_b.parent
    for name in ("train", "calibration", "holdout", "ledger"):
        assert (out / f"{name}.jsonl").read_bytes() == (out_b / f"{name}.jsonl").read_bytes()


def test_reserved_pools_keep_content_but_change_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [sample_row(i) for i in range(6)]
    code, report_path = run_split(
        tmp_path, monkeypatch, source_rows=rows, tag="reserve", holdout_count=1, calibration_count=1
    )
    assert code == 0
    out = report_path.parent
    held_line = (out / "holdout.jsonl").read_text(encoding="utf-8").splitlines()[0]
    held = Sample.model_validate_json(held_line)
    assert held.meta.pool == "holdout"
    expected = json.loads(held.model_dump_json())
    expected["meta"]["pool"] = "train"
    expected["meta"]["data_version"] = SOURCE_DATA_VERSION
    expected_sample = Sample.model_validate(expected)
    original = next(Sample.model_validate(row) for row in rows if row["id"] == held.id)
    assert expected_sample == original
    assert held.questions == original.questions
    assert held.prefix == original.prefix


def test_dev_family_overlap_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev_family = "e" * 64
    rows = [sample_row(i) for i in range(10)]
    rows[4] = sample_row(4, family_id=dev_family)
    code, report_path = run_split(
        tmp_path, monkeypatch, source_rows=rows, dev_families=[dev_family], tag="overlap"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 1 and report["status"] == "blocked"
    assert "dev500" in str(report["blocker"])
    assert not report_path.parent.joinpath("train.jsonl").exists()


def test_source_tamper_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [sample_row(i) for i in range(6)]
    code, report_path = run_split(
        tmp_path, monkeypatch, source_rows=rows, expected_source_sha256="0" * 64, tag="tamper"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 1 and report["status"] == "blocked"
    assert "sha256" in str(report["blocker"])
    assert not report_path.parent.joinpath("train.jsonl").exists()


def test_existing_outputs_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [sample_row(i) for i in range(6)]
    code, report_path = run_split(tmp_path, monkeypatch, source_rows=rows, tag="first")
    assert code == 0
    out = report_path.parent
    (out / "train.jsonl").write_text("sentinel", encoding="utf-8")
    source = tmp_path / "source-first.jsonl"
    digest = sha256_bytes(source.read_bytes())
    argv = [
        "--source",
        str(source),
        "--dev",
        str(tmp_path / "dev.jsonl"),
        "--train-out",
        str(out / "train.jsonl"),
        "--calibration-out",
        str(out / "calibration.jsonl"),
        "--holdout-out",
        str(out / "holdout.jsonl"),
        "--allocation-ledger-out",
        str(out / "ledger.jsonl"),
        "--report",
        str(out / "report-retry.json"),
        "--expected-source-sha256",
        digest,
        "--expected-count",
        "6",
        "--holdout-count",
        "2",
        "--calibration-count",
        "3",
    ]
    assert main(argv) == 1
    blocked = json.loads((out / "report-retry.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert "overwrite" in str(blocked["blocker"])
    assert (out / "train.jsonl").read_text(encoding="utf-8") == "sentinel"


def test_canonical_family_hash_passthrough() -> None:
    assert canonical_family_hash("a" * 64) == "a" * 64
    assert canonical_family_hash("short") == sha256_bytes(b"short")
