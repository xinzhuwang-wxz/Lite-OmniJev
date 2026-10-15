"""Synthetic canaries only: no source or sealed payloads are accessed."""

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from test_gold_inventory import SECRET, manifest, sample, spec

from scripts.gold_inventory import digest
from scripts.vqa2_gold_capacity import MEDIA_LICENSE, main, measure


def fixture(tmp_path: Path) -> dict[str, Any]:
    history = sample("old", modality="image")
    history["meta"]["family_id"] = "coco:1"
    inventory = manifest(tmp_path, [spec(tmp_path, [history], cohort="image")])
    config: dict[str, Any] = {
        "inventory": {"path": str(inventory), "sha256": digest(inventory.read_bytes())},
        "media_roots": [str(tmp_path)],
    }
    metadata = {
        "licenses": [{"id": 4, "url": MEDIA_LICENSE}],
        "images": [
            {
                "id": i,
                "license": 4,
                "file_name": f"{i:012d}.jpg",
                "coco_url": f"http://images.cocodataset.org/train2017/{i:012d}.jpg",
            }
            for i in range(1, 7)
        ],
    }
    payloads = {
        "coco_metadata": metadata,
        "aok_train": [{"image_id": "2"}],
        "quarantine": {"families": [digest(b"coco:3")], "media": []},
    }
    for key, value in payloads.items():
        path = tmp_path / key
        path.write_text(json.dumps(value))
        config[key] = {"path": str(path), "sha256": digest(path.read_bytes())}
    annotations, questions = [], []
    for i in range(1, 7):
        for j, answer in enumerate(("yes", "2", "red")):
            qid = i * 10 + j
            questions.append(
                {"image_id": str(i), "question_id": str(qid), "question": f"{SECRET} color {j}?"}
            )
            annotations.append(
                {
                    "image_id": i,
                    "question_id": qid,
                    "multiple_choice_answer": answer,
                    "answers": [{"answer": answer}] * 10,
                }
            )
    for key, member, rows, field in (
        ("train_annotations", "v2_mscoco_train2014_annotations.json", annotations, "annotations"),
        ("train_questions", "v2_OpenEnded_mscoco_train2014_questions.json", questions, "questions"),
    ):
        path = tmp_path / key
        with zipfile.ZipFile(path, "w") as handle:
            handle.writestr(member, json.dumps({field: rows}))
        config[key] = {"path": str(path), "sha256": digest(path.read_bytes())}
    return config


def rewrite(config: dict[str, Any], key: str, update: Any) -> None:
    path = Path(config[key]["path"])
    if key.startswith("train_"):
        with zipfile.ZipFile(path) as handle:
            member = handle.namelist()[0]
            payload = json.loads(handle.read(member))
        update(payload)
        with zipfile.ZipFile(path, "w") as handle:
            handle.writestr(member, json.dumps(payload))
    else:
        payload = json.loads(path.read_text())
        update(payload)
        path.write_text(json.dumps(payload))
    config[key]["sha256"] = digest(path.read_bytes())


def test_all_pools_quarantine_and_category_overlap(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    result = measure(config, synthetic_canary=True)
    assert result["counts"]["eligible_families"] == 3
    assert result["counts"]["eligible_originals"] == 9
    assert result["counts"]["excluded_families"] == 3
    assert sum(v["families"] for v in result["categories"].values()) == 9
    assert result["counts"]["missing_media_families"] == 3
    assert SECRET not in json.dumps(result)


def test_media_history_and_duplicate_media(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    (tmp_path / "000000000004.jpg").write_bytes(b"image")
    for i in (5, 6):
        (tmp_path / f"{i:012d}.jpg").write_bytes(b"new")
    result = measure(config, synthetic_canary=True)
    assert result["counts"]["history_media_collisions"] == 1
    assert result["counts"]["eligible_families"] == 2
    assert result["counts"]["local_unique_media"] == 1


def test_nonunanimous_and_conflicting_normalized_questions(tmp_path: Path) -> None:
    config = fixture(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        payload["annotations"][9]["answers"] = [{"answer": "no"}] * 10

    rewrite(config, "train_annotations", mutate)

    def conflict(payload: dict[str, Any]) -> None:
        payload["questions"][13]["question"] = payload["questions"][12]["question"].upper()

    rewrite(config, "train_questions", conflict)
    result = measure(config, synthetic_canary=True)
    assert result["counts"]["conflicting_or_nonunanimous_questions"] == 2
    assert result["counts"]["eligible_originals"] == 6


@pytest.mark.parametrize("key", ["inventory", "quarantine", "coco_metadata", "train_annotations"])
def test_hash_mutation_is_rejected(tmp_path: Path, key: str) -> None:
    config = fixture(tmp_path)
    with Path(config[key]["path"]).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError):
        measure(config, synthetic_canary=True)


def test_official_license_exact_and_real_pins(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    with pytest.raises(ValueError):
        measure(config)
    rewrite(config, "coco_metadata", lambda p: p["licenses"][0].update(url="https://example.com"))
    with pytest.raises(ValueError):
        measure(config, synthetic_canary=True)


def test_cli_failure_and_success_never_leak(tmp_path: Path, capsys: Any) -> None:
    assert main(["--config", str(tmp_path / SECRET)]) == 1
    output = capsys.readouterr()
    assert SECRET not in output.out and not output.err
    config = fixture(tmp_path)
    path = tmp_path / "config"
    path.write_text(json.dumps(config))
    before = set(tmp_path.iterdir())
    assert main(["--config", str(path), "--synthetic-canary"]) == 0
    output = capsys.readouterr()
    assert not output.err and SECRET not in output.out and str(tmp_path) not in output.out
    assert json.loads(output.out)["synthetic_canary"] is True
    assert set(tmp_path.iterdir()) == before


def test_exclusion_before_answer_fields(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    rewrite(config, "train_annotations", lambda p: p["annotations"][0].pop("answers"))
    assert measure(config, synthetic_canary=True)["counts"]["eligible_originals"] == 9


def test_bound_historical_payload_mutation(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    inventory = json.loads(Path(config["inventory"]["path"]).read_text())
    Path(inventory["samples"][0]["path"]).write_text(SECRET)
    with pytest.raises(ValueError):
        measure(config, synthetic_canary=True)


def test_crossmodal_bound_component_and_missing_ledger(tmp_path: Path) -> None:
    from test_gold_inventory import artifact, crossmodal

    config = fixture(tmp_path)
    specs, record = crossmodal(tmp_path)
    original = json.loads(Path(specs[0]["path"]).read_text())
    original["meta"]["family_id"] = "coco:4"
    Path(specs[0]["path"]).write_text(json.dumps(original) + "\n")
    specs[0]["sha256"] = digest(Path(specs[0]["path"]).read_bytes())
    record["image"].update(family_id="coco:4", family_hash=digest(b"coco:4"))
    path = manifest(tmp_path, specs, [artifact(tmp_path, [record], "components")])
    config["inventory"] = {"path": str(path), "sha256": digest(path.read_bytes())}
    assert measure(config, synthetic_canary=True)["counts"]["eligible_families"] == 3
    rewrite(config, "inventory", lambda p: p.update(component_ledgers=[]))
    with pytest.raises(ValueError):
        measure(config, synthetic_canary=True)


def test_quarantined_media_and_normalized_duplicate_question(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    (tmp_path / "000000000004.jpg").write_bytes(b"quarantined")
    rewrite(config, "quarantine", lambda p: p["media"].append(digest(b"quarantined")))

    def same_gold(payload: dict[str, Any]) -> None:
        payload["annotations"][13]["multiple_choice_answer"] = "yes"
        payload["annotations"][13]["answers"] = [{"answer": "YES"}] * 10

    rewrite(config, "train_annotations", same_gold)

    def same_question(payload: dict[str, Any]) -> None:
        payload["questions"][13]["question"] = payload["questions"][12]["question"].upper()

    rewrite(config, "train_questions", same_question)
    result = measure(config, synthetic_canary=True)
    assert result["counts"]["history_media_collisions"] == 1
    assert result["counts"]["duplicate_questions"] == 1
    assert result["counts"]["eligible_originals"] == 5
