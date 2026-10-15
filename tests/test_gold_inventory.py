"""Synthetic inventory and private-output boundary tests; no sealed inputs."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.gold_inventory import audit_manifest, digest, main

SECRET = "PRIVATE_SENTINEL_PROMPT_LABEL_ID"


def sample(name: str = SECRET, pool: str = "dev", modality: str = "text") -> dict[str, Any]:
    types = [] if modality == "text" else modality.split("+")
    return {
        "id": name,
        "prefix": {
            "state": SECRET,
            "media": [
                {
                    "type": kind,
                    "uri": f"/never/read/{SECRET}/{kind}",
                    "sha256": digest(kind.encode()),
                }
                for kind in types
            ],
        },
        "questions": [
            {
                "qid": SECRET,
                "type": "noul",
                "instruction": SECRET,
                "target": {"kind": "hard", "value": "yes", "source": SECRET},
            }
        ],
        "meta": {
            "modality": types or ["text"],
            "source_dataset": SECRET,
            "source_revision": SECRET,
            "source_sha256": digest(b"source"),
            "license": "MIT",
            "usage": "commercial",
            "pool": pool,
            "family_id": "family-" + name,
            "data_version": modality,
        },
    }


def artifact(tmp_path: Path, rows: list[dict[str, Any]], name: str = "samples") -> dict[str, Any]:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def manifest(
    tmp_path: Path, specs: list[dict[str, Any]], ledgers: list[dict[str, Any]] | None = None
) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "gold-inventory-manifest-v1",
                "samples": specs,
                "component_ledgers": ledgers or [],
            }
        )
    )
    return path


def spec(tmp_path: Path, rows: list[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    return {
        **artifact(tmp_path, rows, str(len(list(tmp_path.iterdir())))),
        "cohort": "text",
        "exposure": "current",
        "expected_pool": "dev",
        **changes,
    }


def test_cli_success_is_aggregate_and_does_not_read_media(tmp_path: Path) -> None:
    path = manifest(tmp_path, [spec(tmp_path, [sample(modality="image")], cohort="image")])
    report = tmp_path / "receipt.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.gold_inventory",
            "--manifest",
            str(path),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert process.stderr == ""
    assert process.stdout == report.read_text()
    assert SECRET not in process.stdout
    assert str(tmp_path) not in process.stdout
    result = json.loads(process.stdout)
    assert result["scope"] == "supplied_inventory_only"
    assert result["inventory_completeness"] == "not_validated"
    assert result["lineage_completeness"] == "not_validated"
    assert result["private_job_permissions"] == "not_validated"
    assert result["gold_provenance"] == "not_validated"
    assert result["isolation"]["unique_items"] == 1


def test_historical_repeats_and_stable_id_detect_pool_move(tmp_path: Path) -> None:
    historical = spec(
        tmp_path,
        [sample(pool="train"), sample(pool="train")],
        exposure="historical",
        expected_pool="train",
    )
    result = audit_manifest(manifest(tmp_path, [historical]))
    assert result.outcome == "isolated"
    assert result.isolation is not None and result.isolation.historical_rows == 2
    current = spec(tmp_path, [sample(pool="holdout")], expected_pool="holdout")
    result = audit_manifest(manifest(tmp_path, [historical, current]))
    assert result.categories == ["pool_overlap"]


def test_family_hash_is_not_double_hashed(tmp_path: Path) -> None:
    first, second = sample("one"), sample("two", pool="holdout")
    second["meta"]["family_id"] = digest(first["meta"]["family_id"].encode()).upper()
    result = audit_manifest(
        manifest(
            tmp_path, [spec(tmp_path, [first]), spec(tmp_path, [second], expected_pool="holdout")]
        )
    )
    assert result.categories == ["pool_overlap"]


@pytest.mark.parametrize("mutation", ["payload", "family", "duplicate"])
def test_conflicting_ids_rejected(tmp_path: Path, mutation: str) -> None:
    first, second = sample(), sample()
    if mutation == "payload":
        second["prefix"]["state"] += "changed"
    elif mutation == "family":
        second["meta"]["family_id"] += "changed"
    result = audit_manifest(manifest(tmp_path, [spec(tmp_path, [first, second])]))
    assert result.categories == ["conflicting_identity"]


def test_mixed_cohort_requires_explicit_version_mapping(tmp_path: Path) -> None:
    value = spec(tmp_path, [sample("text"), sample("image", modality="image")])
    del value["cohort"]
    value["cohorts_by_data_version"] = {"text": "text", "image": "screenshot"}
    path = manifest(tmp_path, [value])
    assert audit_manifest(path).outcome == "isolated"
    value["cohorts_by_data_version"].pop("image")
    assert audit_manifest(manifest(tmp_path, [value])).categories == ["invalid_sample"]
    value["cohort"] = "text"
    assert audit_manifest(manifest(tmp_path, [value])).categories == ["invalid_manifest"]


@pytest.mark.parametrize(
    "mutation,category",
    [
        ("hash", "hash_mismatch"),
        ("json", "invalid_sample"),
        ("validation", "invalid_sample"),
        ("io", "input_io"),
        ("pool", "pool_mismatch"),
        ("cohort", "modality_mismatch"),
        ("missing_media", "modality_mismatch"),
        ("declared_modality", "modality_mismatch"),
    ],
)
def test_private_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mutation: str, category: str
) -> None:
    row = sample(modality="image")
    if mutation == "validation":
        row["questions"][0]["target"]["value"] = SECRET
    if mutation == "missing_media":
        row["prefix"]["media"] = []
    if mutation == "declared_modality":
        row["meta"]["modality"].append("audio")
    value = spec(tmp_path, [row], cohort="image")
    if mutation == "hash":
        value["sha256"] = "0" * 64
    if mutation == "json":
        Path(value["path"]).write_text('{"' + SECRET)
        value["sha256"] = digest(Path(value["path"]).read_bytes())
    if mutation == "io":
        value["path"] = str(tmp_path / SECRET)
    if mutation == "pool":
        value["expected_pool"] = "holdout"
    if mutation == "cohort":
        value["cohort"] = "audio"
    path = manifest(tmp_path, [value])
    report = tmp_path / "receipt"
    assert main(["--manifest", str(path), "--report", str(report)]) == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert SECRET not in output.out + report.read_text()
    assert str(tmp_path) not in output.out
    assert json.loads(output.out)["categories"] == [category]


def crossmodal(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image, audio = sample("image", modality="image"), sample("audio", modality="audio")
    composite = sample("crossmodal", modality="image+audio")
    record: dict[str, Any] = {
        "item_id": composite["id"],
        "crossmodal_family_id": composite["meta"]["family_id"],
        "crossmodal_family_hash": digest(composite["meta"]["family_id"].encode()),
        "pool": "dev",
        "proposal_rule": SECRET,
        "program_truth": "yes",
        "component_family_separate_ledger": True,
    }
    for kind, row in (("image", image), ("audio", audio)):
        record[kind] = {
            "item_id": row["id"],
            "family_id": row["meta"]["family_id"],
            "family_hash": digest(row["meta"]["family_id"].encode()),
            "source_hash": row["meta"]["source_sha256"],
            "media_sha256": row["prefix"]["media"][0]["sha256"],
            "proposal": SECRET,
        }
    return [
        spec(tmp_path, [image], cohort="image"),
        spec(tmp_path, [audio], cohort="audio"),
        spec(tmp_path, [composite], cohort="crossmodal"),
    ], record


@pytest.mark.parametrize(
    "mutation",
    ["none", "missing_ledger", "missing_ref", "family", "media", "pool", "source", "hash", "extra"],
)
def test_hash_bound_crossmodal_join(tmp_path: Path, mutation: str) -> None:
    specs, record = crossmodal(tmp_path)
    if mutation == "missing_ref":
        record["audio"]["item_id"] = SECRET
    if mutation == "family":
        record["image"]["family_hash"] = "0" * 64
    if mutation == "media":
        record["image"]["media_sha256"] = "0" * 64
    if mutation == "source":
        record["image"]["source_hash"] = "0" * 64
    if mutation == "pool":
        record["pool"] = "holdout"
    if mutation == "extra":
        record[SECRET] = SECRET
    ledger = artifact(tmp_path, [record], "components")
    if mutation == "hash":
        ledger["sha256"] = "0" * 64
    result = audit_manifest(
        manifest(tmp_path, specs, [] if mutation == "missing_ledger" else [ledger])
    )
    assert result.outcome == ("isolated" if mutation == "none" else "rejected")
    assert SECRET not in result.model_dump_json()
    if mutation == "none":
        assert result.isolation is not None and result.isolation.connected_components == 1


def test_report_no_overwrite_and_argument_errors_are_private(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = manifest(tmp_path, [spec(tmp_path, [sample()])])
    report = tmp_path / SECRET
    report.write_text(SECRET)
    assert main(["--manifest", str(path), "--report", str(report)]) == 1
    assert report.read_text() == SECRET
    assert list(tmp_path.glob("tmp*")) == []
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert json.loads(output.out)["categories"] == ["report_io"]
    assert main(["--" + SECRET]) == 1
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert json.loads(output.out)["categories"] == ["invalid_arguments"]


def test_manifest_validation_never_echoes_input(tmp_path: Path) -> None:
    path = tmp_path / SECRET
    path.write_text(json.dumps({SECRET: SECRET}))
    result = audit_manifest(path)
    assert result.categories == ["invalid_manifest"]
    assert SECRET not in result.model_dump_json()


def test_unexpected_failure_is_fixed_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.gold_inventory as inventory

    def fail(path: Path) -> bytes:
        raise RuntimeError(SECRET)

    monkeypatch.setattr(inventory, "read_bytes", fail)
    result = inventory.audit_manifest(tmp_path / SECRET)
    assert result.categories == ["job_failed"]
    assert SECRET not in result.model_dump_json()


def test_failed_atomic_publication_leaves_no_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = manifest(tmp_path, [spec(tmp_path, [sample()])])
    before = set(tmp_path.iterdir())

    def fail(source: str, destination: Path) -> None:
        raise OSError(SECRET)

    monkeypatch.setattr(os, "link", fail)
    assert main(["--manifest", str(path), "--report", str(tmp_path / "receipt")]) == 1
    assert set(tmp_path.iterdir()) == before
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert json.loads(output.out)["categories"] == ["report_io"]


@pytest.mark.parametrize("content", ['{"' + SECRET, '{"item_id":"' + SECRET + '"}'])
def test_component_parse_failures_are_private(tmp_path: Path, content: str) -> None:
    specs, _ = crossmodal(tmp_path)
    path = tmp_path / "component"
    path.write_text(content)
    result = audit_manifest(
        manifest(
            tmp_path,
            specs,
            [
                {"path": str(path), "sha256": digest(path.read_bytes())},
            ],
        )
    )
    assert result.categories == ["invalid_component"]
    assert SECRET not in result.model_dump_json()
