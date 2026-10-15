"""Synthetic TRAIN-only pilot fixtures; no sealed inputs."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_gold_inventory import manifest, sample, spec

from data.schema import Sample
from scripts.gold_inventory import audit_manifest, digest
from scripts.gold_media_bundle import Rejected
from scripts.s1_text_pools import canonical_sample_sha
from scripts.s4_crossmodal_train import VERSION, build, main
from train.trainer import LoopConfig, smoke_samples


def fixture(tmp_path: Path, count: int = 5) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    artifacts, sources, rows = [], [], []
    for modality in ("image", "audio"):
        values = []
        for index in range(count):
            row = sample(f"{modality}-{index}", pool="train", modality=modality)
            media = tmp_path / f"{modality}-{index}.bin"
            media.write_bytes(f"{modality}-{index}".encode())
            row["prefix"]["media"][0].update(uri=media.name, sha256=digest(media.read_bytes()))
            row["meta"]["source_sha256"] = digest(f"source-{modality}-{index}".encode())
            q = row["questions"][0]
            q["target"]["source"] = (
                "aokvqa_correct_choice_idx"
                if modality == "image"
                else "clotho_original_unanimous_three_answers"
            )
            if modality == "image":
                q.update(type="choice", options=["red", "blue"])
                q["target"]["value"] = "A"
            values.append(row)
        artifact = spec(tmp_path, values, cohort=modality, expected_pool="train")
        artifacts.append(artifact)
        sources.append(
            {k: artifact[k] for k in ("path", "sha256")}
            | {"modality": modality, "media_root": str(tmp_path)}
        )
        rows.extend(values)
    inventory = manifest(tmp_path, artifacts)
    config = {
        "inventory": {"path": str(inventory), "sha256": digest(inventory.read_bytes())},
        "sources": sources,
    }
    path = tmp_path / "build.json"
    path.write_text(json.dumps(config))
    return path, config, rows


def rewrite(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for source in config["sources"]:
        values = [r for r in rows if r["meta"]["modality"] == [source["modality"]]]
        target = Path(source["path"])
        target.write_text("".join(json.dumps(r) + "\n" for r in values))
        source["sha256"] = digest(target.read_bytes())
    inv = Path(config["inventory"]["path"])
    inventory = json.loads(inv.read_bytes())
    for artifact, source in zip(inventory["samples"], config["sources"], strict=True):
        artifact["sha256"] = source["sha256"]
    inv.write_text(json.dumps(inventory))
    config["inventory"]["sha256"] = digest(inv.read_bytes())
    path.write_text(json.dumps(config))


def test_pilot_determinism_labels_sources_and_permissions(tmp_path: Path) -> None:
    path, config, _ = fixture(tmp_path)
    before = {s["path"]: Path(s["path"]).read_bytes() for s in config["sources"]}
    result = build(path, tmp_path / "one", count=4)
    build(path, tmp_path / "two", count=4)
    assert (tmp_path / "one/train.jsonl").read_bytes() == (
        tmp_path / "two/train.jsonl"
    ).read_bytes()
    assert result["rules"] == {
        "both_true": 2,
        "image_false_audio_true": 1,
        "image_true_audio_false": 1,
    }
    samples = [
        Sample.model_validate_json(line)
        for line in (tmp_path / "one/train.jsonl").read_text().splitlines()
    ]
    assert [s.questions[0].target.value for s in samples] == ["yes", "yes", "no", "no"]
    assert all(
        s.meta.pool == "train"
        and s.meta.data_version == VERSION
        and s.id.startswith("crossmodal-train-")
        for s in samples
    )
    assert audit_manifest(tmp_path / "one/inventory-manifest.json").outcome == "isolated"
    for source, content in before.items():
        assert Path(source).read_bytes() == content
    assert (tmp_path / "one").stat().st_mode & 0o777 == 0o700
    assert all(
        p.stat().st_mode & 0o777 == 0o600 for p in (tmp_path / "one").rglob("*") if p.is_file()
    )
    ledger = [
        json.loads(line) for line in (tmp_path / "one/allocation.jsonl").read_text().splitlines()
    ]
    assert {r["item_id"]: r["derived_sample_sha256"] for r in ledger} == {
        s.id: canonical_sample_sha(s) for s in samples
    }


@pytest.mark.parametrize("change", ["pool", "teacher", "target", "soft", "path", "sha", "capacity"])
def test_reject_source_contract(tmp_path: Path, change: str) -> None:
    path, config, rows = fixture(tmp_path)
    if change == "pool":
        rows[0]["meta"]["pool"] = "dev"
    elif change == "teacher":
        rows[0]["questions"][0]["target"]["source"] = "teacher"
    elif change == "target":
        rows[0]["questions"][0]["target"]["value"] = "U"
    elif change == "soft":
        rows[0]["questions"][0]["target"] = {
            "kind": "soft",
            "distribution": {"A": 1.0},
            "source": "teacher",
        }
    elif change == "path":
        rows[0]["prefix"]["media"][0]["uri"] = "../escape"
    elif change == "sha":
        rows[0]["prefix"]["media"][0]["sha256"] = "f" * 64
    else:
        for row in rows[:5]:
            row["meta"]["family_id"] = "one-family"
    rewrite(path, config, rows)
    with pytest.raises((ValueError, Rejected)):
        build(path, tmp_path / "out", count=4)
    assert not (tmp_path / "out").exists()


def test_cohort_mapping_and_crosspool_audit(tmp_path: Path) -> None:
    path, config, _ = fixture(tmp_path)
    inv = Path(config["inventory"]["path"])
    content = json.loads(inv.read_bytes())
    for artifact in content["samples"]:
        cohort = artifact.pop("cohort")
        artifact["cohorts_by_data_version"] = {cohort: cohort}
    inv.write_text(json.dumps(content))
    config["inventory"]["sha256"] = digest(inv.read_bytes())
    path.write_text(json.dumps(config))
    build(path, tmp_path / "out", count=4)
    final_path = tmp_path / "out/inventory-manifest.json"
    final = json.loads(final_path.read_bytes())
    train = tmp_path / "out/train.jsonl"
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    for row in rows:
        row["meta"]["pool"] = "dev"
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    final["samples"][-1].update(sha256=digest(train.read_bytes()), expected_pool="dev")
    final_path.write_text(json.dumps(final))
    assert audit_manifest(final_path).outcome == "rejected"


def test_cli_private_failure_and_no_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _, _ = fixture(tmp_path)
    args = [
        "--manifest",
        str(path),
        "--output-dir",
        str(tmp_path / "out"),
        "--report",
        str(tmp_path / "report.json"),
        "--count",
        "4",
    ]
    assert main(args) == 0
    before = (tmp_path / "report.json").read_bytes()
    assert main(args) == 1
    assert (tmp_path / "report.json").read_bytes() == before
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("count", [0, 3, 6, 204])
def test_quota_rejected(tmp_path: Path, count: int) -> None:
    path, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="quota"):
        build(path, tmp_path / "out", count=count)


def test_trainer_mixed_cpu_loading(tmp_path: Path) -> None:
    path, _, _ = fixture(tmp_path)
    build(path, tmp_path / "out", count=4)
    config = LoopConfig(
        sample_count=4,
        steps=1,
        accumulation=1,
        modality="mixed",
        media_root=str(tmp_path / "out"),
        expected_data_versions=[VERSION],
    )
    samples = smoke_samples(config, tmp_path / "out/train.jsonl", expected_data_versions=[VERSION])
    assert len(samples) == 4


@pytest.mark.parametrize(
    "change", ["inventory_sha", "source_sha", "unregistered", "root", "symlink"]
)
def test_bound_inputs_rejected(tmp_path: Path, change: str) -> None:
    path, config, _ = fixture(tmp_path)
    if change == "inventory_sha":
        config["inventory"]["sha256"] = "f" * 64
    elif change == "source_sha":
        Path(config["sources"][0]["path"]).write_text("modified")
    elif change == "unregistered":
        config["sources"][0]["path"] = str(tmp_path / "other")
    elif change == "root":
        config["sources"][0]["media_root"] = "relative"
    else:
        original = Path(config["sources"][0]["path"])
        renamed = original.with_suffix(".original")
        original.rename(renamed)
        original.symlink_to(renamed)
    path.write_text(json.dumps(config))
    with pytest.raises((ValueError, Rejected)):
        build(path, tmp_path / "out", count=4)


def test_default_200_exact_unique_mix(tmp_path: Path) -> None:
    path, _, _ = fixture(tmp_path, count=200)
    result = build(path, tmp_path / "out")
    assert result["rules"] == {
        "both_true": 100,
        "image_false_audio_true": 50,
        "image_true_audio_false": 50,
    }
    components = [
        json.loads(line) for line in (tmp_path / "out/components.jsonl").read_text().splitlines()
    ]
    for modality in ("image", "audio"):
        assert len({row[modality]["family_id"] for row in components}) == 200
        assert len({row[modality]["media_sha256"] for row in components}) == 200


def test_duplicate_media_capacity_rejected(tmp_path: Path) -> None:
    path, config, rows = fixture(tmp_path)
    for row in rows[:5]:
        row["prefix"]["media"] = rows[0]["prefix"]["media"]
    rewrite(path, config, rows)
    with pytest.raises(ValueError, match="unique_capacity"):
        build(path, tmp_path / "out", count=4)
