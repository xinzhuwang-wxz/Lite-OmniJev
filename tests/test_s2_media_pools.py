import json
from pathlib import Path

from data.schema import Sample
from scripts.s2_media_pools import main, sha256_bytes

IMAGE_POOLS_VERSION = "aokvqa-train-image-choice-s2-pools-v1"
SCREENSHOT_POOLS_VERSION = "gui-v2-screenshot-s2-pools-v1"
SCREENSHOT_SOURCE_DATA_VERSION = "gui-v2-train-screenshot-s2-candidates-v1"

COHORTS: dict[str, dict[str, str]] = {
    "image": {
        "id_prefix": "aokvqa-train",
        "family_prefix": "coco",
        "media_dir": "train",
        "source_dataset": "A-OKVQA",
        "source_revision": "42e8c44ff40d13cf272776a3197365f582137772",
        "license": "QA:Apache-2.0;media:CC-BY-2.0",
        "data_version": "aokvqa-train-image-choice-dev-v1",
        "state": "Answer the multiple-choice question using the attached image.",
        "target_source": "aokvqa_correct_choice_idx",
    },
    "screenshot": {
        "id_prefix": "gui-odyssey-train",
        "family_prefix": "gui",
        "media_dir": "screenshots",
        "source_dataset": "hflqf88888/GUIOdyssey",
        "source_revision": "61632d0f3f4d51d7e9561ce4f84347dd06b2019d",
        "license": "CC-BY-4.0",
        "data_version": SCREENSHOT_SOURCE_DATA_VERSION,
        "state": "Select the screenshot region that matches the atomic GUI subtask.",
        "target_source": "gui_odyssey_sam2_bbox_program_truth",
    },
}


def sample_row(
    cohort: str,
    index: int,
    *,
    media_uri: str,
    media_sha256: str,
    family_id: str | None = None,
    target: str = "B",
) -> dict:
    spec = COHORTS[cohort]
    return {
        "id": f"{spec['id_prefix']}-{index:05d}",
        "prefix": {
            "state": spec["state"],
            "media": [{"type": "image", "uri": media_uri, "sha256": media_sha256}],
        },
        "questions": [
            {
                "qid": "choice",
                "type": "choice",
                "instruction": f"question number {index}",
                "options": [f"option {i:02d}" for i in range(4)],
                "target": {"kind": "hard", "value": target, "source": spec["target_source"]},
            }
        ],
        "meta": {
            "modality": ["image"],
            "source_dataset": spec["source_dataset"],
            "source_revision": spec["source_revision"],
            "source_sha256": "a" * 64,
            "license": spec["license"],
            "usage": "commercial",
            "pool": "train",
            "family_id": family_id or f"{spec['family_prefix']}:{index}",
            "data_version": spec["data_version"],
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode())


def make_media(media_root: Path, cohort: str, index: int) -> tuple[str, str]:
    name = f"{COHORTS[cohort]['media_dir']}/{index:05d}.png"
    path = media_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{cohort}-media-{index}".encode()
    path.write_bytes(payload)
    return name, sha256_bytes(payload)


def cohort_rows(media_root: Path, cohort: str, count: int, *, start: int = 0) -> list[dict]:
    rows: list[dict] = []
    for index in range(start, start + count):
        uri, digest = make_media(media_root, cohort, index)
        rows.append(sample_row(cohort, index, media_uri=uri, media_sha256=digest))
    return rows


def run_split(
    tmp_path: Path,
    *,
    cohort: str,
    media_root: Path,
    source_rows: list[dict],
    tag: str,
    dev_rows: list[dict] | None = None,
    holdout_count: int = 2,
    calibration_count: int = 3,
) -> tuple[int, Path]:
    source = tmp_path / f"source-{tag}.jsonl"
    write_jsonl(source, source_rows)
    dev = tmp_path / f"dev-{tag}.jsonl"
    dev_digest = write_jsonl(dev, dev_rows or [dev_row(cohort, 0)])
    out = tmp_path / f"out-{tag}"
    argv = [
        "--cohort",
        cohort,
        "--source",
        str(source),
        "--media-root",
        str(media_root),
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
        "--holdout-count",
        str(holdout_count),
        "--calibration-count",
        str(calibration_count),
        "--expected-dev-sha256",
        dev_digest,
        "--expected-source-data-version",
        COHORTS[cohort]["data_version"],
    ]
    return main(argv), out / "report.json"


def dev_row(cohort: str, index: int) -> dict:
    return sample_row(
        cohort,
        90000 + index,
        family_id=f"{COHORTS[cohort]['family_prefix']}-dev-{index}",
        media_uri="dev/unused.png",
        media_sha256="0" * 64,
    )


def test_image_split_counts_determinism_and_isolation(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 10)
    code, report_path = run_split(
        tmp_path, cohort="image", media_root=media_root, source_rows=rows, tag="a"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0 and report["status"] == "passed"
    assert report["split"] == {
        "train_items": 5,
        "calibration_reserve_items": 3,
        "holdout_reserve_items": 2,
        "train_families": 5,
        "calibration_reserve_families": 3,
        "holdout_reserve_families": 2,
        "dev_family_overlap": 0,
    }
    out = report_path.parent
    pools: dict[str, set[str]] = {}
    for name, count in (("train", 5), ("calibration", 3), ("holdout", 2)):
        lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == count
        samples = [Sample.model_validate_json(line) for line in lines]
        pools[name] = {sample.meta.family_id for sample in samples}
        for sample in samples:
            assert sample.meta.pool == name
            assert sample.meta.data_version == IMAGE_POOLS_VERSION
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
            f"s2-image-pools-v1:42:{row['family_id']}".encode()
        )
    code_b, report_b = run_split(
        tmp_path, cohort="image", media_root=media_root, source_rows=rows, tag="b"
    )
    assert code_b == 0
    out_b = report_b.parent
    for name in ("train", "calibration", "holdout", "ledger"):
        assert (out / f"{name}.jsonl").read_bytes() == (out_b / f"{name}.jsonl").read_bytes()


def test_reserved_pools_keep_content_but_change_pool(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 6)
    code, report_path = run_split(
        tmp_path,
        cohort="image",
        media_root=media_root,
        source_rows=rows,
        tag="reserve",
        holdout_count=1,
        calibration_count=1,
    )
    assert code == 0
    out = report_path.parent
    held_line = (out / "holdout.jsonl").read_text(encoding="utf-8").splitlines()[0]
    held = Sample.model_validate_json(held_line)
    assert held.meta.pool == "holdout"
    assert held.meta.data_version == IMAGE_POOLS_VERSION
    expected = json.loads(held.model_dump_json())
    expected["meta"]["pool"] = "train"
    expected["meta"]["data_version"] = COHORTS["image"]["data_version"]
    expected_sample = Sample.model_validate(expected)
    original = next(Sample.model_validate(row) for row in rows if row["id"] == held.id)
    assert expected_sample == original
    assert held.questions == original.questions
    assert held.prefix == original.prefix


def test_media_missing_is_blocked_with_sample_id(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 6)
    missing = rows[3]
    (media_root / missing["prefix"]["media"][0]["uri"]).unlink()
    code, report_path = run_split(
        tmp_path, cohort="image", media_root=media_root, source_rows=rows, tag="missing"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 1 and report["status"] == "blocked"
    assert missing["id"] in str(report["blocker"])
    assert "missing" in str(report["blocker"])
    assert not report_path.parent.joinpath("train.jsonl").exists()


def test_media_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 6)
    tampered = rows[3]
    (media_root / tampered["prefix"]["media"][0]["uri"]).write_bytes(b"tampered")
    code, report_path = run_split(
        tmp_path, cohort="image", media_root=media_root, source_rows=rows, tag="tamper"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 1 and report["status"] == "blocked"
    assert tampered["id"] in str(report["blocker"])
    assert "sha256" in str(report["blocker"])
    assert not report_path.parent.joinpath("train.jsonl").exists()


def test_dev_family_overlap_is_rejected(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 10)
    overlap_family = rows[4]["meta"]["family_id"]
    dev = [
        sample_row(
            "image",
            95000,
            family_id=overlap_family,
            media_uri="dev/unused.png",
            media_sha256="0" * 64,
        )
    ]
    code, report_path = run_split(
        tmp_path,
        cohort="image",
        media_root=media_root,
        source_rows=rows,
        tag="overlap",
        dev_rows=dev,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 1 and report["status"] == "blocked"
    assert "dev" in str(report["blocker"])
    assert not report_path.parent.joinpath("train.jsonl").exists()


def test_screenshot_split_with_overridden_counts_and_shared_family(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "screenshot", 6)
    shared_family = "gui-shared-episode"
    for index in (6, 7):
        uri, digest = make_media(media_root, "screenshot", index)
        rows.append(
            sample_row(
                "screenshot", index, family_id=shared_family, media_uri=uri, media_sha256=digest
            )
        )
    code, report_path = run_split(
        tmp_path,
        cohort="screenshot",
        media_root=media_root,
        source_rows=rows,
        tag="shot",
        holdout_count=3,
        calibration_count=2,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0 and report["status"] == "passed"
    split = report["split"]
    assert split["holdout_reserve_families"] == 3
    assert split["calibration_reserve_families"] == 2
    assert split["train_families"] == 2
    assert (
        split["train_items"] + split["calibration_reserve_items"] + split["holdout_reserve_items"]
        == 8
    )
    assert split["dev_family_overlap"] == 0
    out = report_path.parent
    line_counts = {
        name: len((out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines())
        for name in ("train", "calibration", "holdout")
    }
    assert sum(line_counts.values()) == 8
    ledger = [
        json.loads(line) for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger) == 8
    shared = [row for row in ledger if row["family_id"] == shared_family]
    assert len(shared) == 2
    assert len({row["pool"] for row in shared}) == 1
    for row in ledger:
        assert row["allocation_key_sha256"] == sha256_bytes(
            f"s2-screenshot-pools-v1:42:{row['family_id']}".encode()
        )
    holdout_samples = [
        Sample.model_validate_json(line)
        for line in (out / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for sample in holdout_samples:
        assert sample.meta.pool == "holdout"
        assert sample.meta.data_version == SCREENSHOT_POOLS_VERSION
    code_b, report_b = run_split(
        tmp_path,
        cohort="screenshot",
        media_root=media_root,
        source_rows=rows,
        tag="shot-b",
        holdout_count=3,
        calibration_count=2,
    )
    assert code_b == 0
    for name in ("train", "calibration", "holdout", "ledger"):
        assert (out / f"{name}.jsonl").read_bytes() == (
            report_b.parent / f"{name}.jsonl"
        ).read_bytes()


def test_family_target_conflict_is_rejected(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "screenshot", 4)
    uri, digest = make_media(media_root, "screenshot", 4)
    # 同 family 同题面、不同 target —— 真冲突。
    conflicting = sample_row(
        "screenshot",
        4,
        family_id=rows[0]["meta"]["family_id"],
        media_uri=uri,
        media_sha256=digest,
        target="A",
    )
    conflicting["questions"][0]["instruction"] = rows[0]["questions"][0]["instruction"]
    conflicting["questions"][0]["options"] = rows[0]["questions"][0]["options"]
    rows.append(conflicting)
    code, report_path = run_split(
        tmp_path, cohort="screenshot", media_root=media_root, source_rows=rows, tag="conflict"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 1 and report["status"] == "blocked"
    assert rows[0]["meta"]["family_id"] in str(report["blocker"])
    assert not report_path.parent.joinpath("train.jsonl").exists()


def test_existing_outputs_are_refused(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 6)
    code, report_path = run_split(
        tmp_path, cohort="image", media_root=media_root, source_rows=rows, tag="first"
    )
    assert code == 0
    out = report_path.parent
    (out / "train.jsonl").write_text("sentinel", encoding="utf-8")
    source = tmp_path / "source-first.jsonl"
    dev = tmp_path / "dev-first.jsonl"
    argv = [
        "--cohort",
        "image",
        "--source",
        str(source),
        "--media-root",
        str(media_root),
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
        str(out / "report-retry.json"),
        "--holdout-count",
        "2",
        "--calibration-count",
        "3",
        "--expected-dev-sha256",
        sha256_bytes(dev.read_bytes()),
        "--expected-source-data-version",
        COHORTS["image"]["data_version"],
    ]
    assert main(argv) == 1
    blocked = json.loads((out / "report-retry.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert "overwrite" in str(blocked["blocker"])
    assert (out / "train.jsonl").read_text(encoding="utf-8") == "sentinel"


def test_multi_question_family_same_image_is_accepted(tmp_path: Path) -> None:
    """同图不同问题（不同 instruction/target）合法，必须整 family 进同一池。"""
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "image", 4)
    uri, digest = make_media(media_root, "image", 4)
    extra = sample_row(
        "image",
        4,
        family_id=rows[0]["meta"]["family_id"],
        media_uri=uri,
        media_sha256=digest,
        target="A",
    )
    rows.append(extra)
    code, report_path = run_split(
        tmp_path,
        cohort="image",
        media_root=media_root,
        source_rows=rows,
        tag="multiq",
        holdout_count=1,
        calibration_count=1,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0 and report["status"] == "passed"
    out = report_path.parent
    family = rows[0]["meta"]["family_id"]
    locations = {
        name: [
            json.loads(line)["meta"]["family_id"]
            for line in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train", "calibration", "holdout")
    }
    homes = [name for name, fams in locations.items() if family in fams]
    assert len(homes) == 1
    assert sum(fams.count(family) for fams in locations.values()) == 2
