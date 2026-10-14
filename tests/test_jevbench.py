from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.jevbench import (
    EXPECTED_PRIMITIVE_COUNTS,
    JevBenchProtocolError,
    JevBenchTask,
    build_decision_request,
    build_sample,
    corpus_row,
    load_public_tasks,
    logits_to_source_distribution,
    project_distribution_to_source,
    sha256_file,
)
from eval.jevbench import argmax_label, score_predictions, score_task


def _row(
    *,
    row_id: str,
    primitive: str,
    labels: list[str],
    expected: str | int,
    group: str | None,
    family: str = "family_a",
    state: str | dict[str, object] = "plain state",
) -> dict[str, object]:
    if primitive == "choice":
        criteria: dict[str, object] | list[str] = {
            label: f"Meaning for {label}" for label in labels
        }
    elif primitive == "score":
        criteria = [f"Level {label}" for label in labels]
    else:
        criteria = {
            "false": "The statement is not satisfied.",
            "true": "The statement is satisfied.",
        }
    return {
        "expected": expected,
        "family": family,
        "group": group,
        "id": row_id,
        "labels": labels,
        "provenance": {
            "license": "MIT",
            "source": "test fixture",
            "gold_probs": {str(expected): 1.0},
        },
        "question": {
            "criteria": criteria,
            "instructions": f"Decide {row_id}",
            "type": primitive,
        },
        "split": "public",
        "state": state,
    }


def _rows_for_split(split: str, counts: dict[str, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for primitive, count in counts.items():
        for index in range(count):
            row_id = f"{split}-{primitive}-{index:03d}"
            group = None if split == "easy" else f"{split}-group-{len(rows) // 2:03d}"
            if split == "hard":
                group = f"hard-group-{len(rows):03d}"
            if primitive == "choice":
                labels = ["S", "6_credits", "other"] if index == 0 else ["alpha", "beta", "gamma"]
                expected: str | int = labels[index % len(labels)]
            elif primitive == "score":
                labels = ["0", "1", "2", "3", "4"] if index == 0 else ["0", "1", "2", "3"]
                expected = index % len(labels)
            else:
                labels = ["no", "yes"]
                expected = labels[index % 2]
            state: str | dict[str, object] = (
                {"z": 2, "a": {"nested": True}} if row_id == "hard-choice-000" else "plain state"
            )
            rows.append(
                _row(
                    row_id=row_id,
                    primitive=primitive,
                    labels=labels,
                    expected=expected,
                    group=group,
                    family=f"family_{primitive}",
                    state=state,
                )
            )
    return rows


def _write_source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    paths = {
        "LICENSE": "MIT\n",
        "datasets/HARD-TIER.md": "hard tier\n",
        "datasets/manifest.json": "{}\n",
        "jevbench/metrics.py": "# metrics\n",
        "jevbench/scoring.py": "# scoring\n",
    }
    split_rows = {
        "datasets/public/original.jsonl": _rows_for_split(
            "original", {"noul": 24, "choice": 36, "score": 12}
        ),
        "datasets/public/easy.jsonl": _rows_for_split("easy", {"choice": 36, "noul": 12}),
        "datasets/public/hard.jsonl": _rows_for_split(
            "hard", {"choice": 67, "noul": 38, "score": 6}
        ),
    }
    for relative, text in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for relative, rows in split_rows.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    receipts = []
    for relative in [
        "LICENSE",
        "datasets/HARD-TIER.md",
        "datasets/manifest.json",
        "datasets/public/easy.jsonl",
        "datasets/public/hard.jsonl",
        "datasets/public/original.jsonl",
        "jevbench/metrics.py",
        "jevbench/scoring.py",
    ]:
        path = root / relative
        receipts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "revision": "7625ecb1b075cfe4d09d34141d353325825eea1d",
                "files": receipts,
            }
        ),
        encoding="utf-8",
    )
    return root, config


def _load_fixture(tmp_path: Path) -> tuple[tuple[JevBenchTask, ...], Path, Path]:
    root, config_path = _write_source(tmp_path)
    config = json.loads(config_path.read_text())
    return load_public_tasks(root, config), root, config_path


def test_load_public_tasks_requires_locked_hashes_and_all_231_rows(tmp_path: Path) -> None:
    tasks, _, _ = _load_fixture(tmp_path)

    assert len(tasks) == 231
    assert {
        primitive: sum(1 for task in tasks if task.primitive == primitive)
        for primitive in EXPECTED_PRIMITIVE_COUNTS
    } == EXPECTED_PRIMITIVE_COUNTS

    root, config_path = _write_source(tmp_path / "bad")
    config = json.loads(config_path.read_text())
    config["files"][0]["sha256"] = "0" * 64
    with pytest.raises(JevBenchProtocolError, match="sha256 mismatch"):
        load_public_tasks(root, config)


def test_missing_or_duplicate_task_is_rejected_before_scoring(tmp_path: Path) -> None:
    tasks, root, config_path = _load_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    original_path = root / "datasets/public/easy.jsonl"
    lines = original_path.read_text(encoding="utf-8").splitlines()
    original_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    for entry in config["files"]:
        if entry["path"] == "datasets/public/easy.jsonl":
            entry["bytes"] = original_path.stat().st_size
            entry["sha256"] = sha256_file(original_path)
    with pytest.raises(JevBenchProtocolError, match="row counts changed"):
        load_public_tasks(root, config)

    predictions = {
        task.id: {"probabilities": {label: 1.0 / len(task.labels) for label in task.labels}}
        for task in tasks
    }
    predictions.pop(tasks[0].id)
    with pytest.raises(JevBenchProtocolError, match="prediction ids mismatch"):
        score_predictions(tasks, predictions)


def test_adapter_preserves_source_label_order_and_reverses_project_tokens(tmp_path: Path) -> None:
    tasks, _, _ = _load_fixture(tmp_path)
    choice = next(task for task in tasks if task.labels == ("S", "6_credits", "other"))
    noul = next(task for task in tasks if task.primitive == "noul")

    assert choice.project_tokens == ("A", "B", "C")
    assert choice.source_to_project == {"S": "A", "6_credits": "B", "other": "C"}
    assert project_distribution_to_source(choice, {"A": 0.1, "B": 0.7, "C": 0.2}) == {
        "S": 0.1,
        "6_credits": 0.7,
        "other": 0.2,
    }

    assert noul.labels == ("no", "yes")
    assert noul.project_tokens == ("yes", "no")
    assert project_distribution_to_source(noul, {"yes": 0.8, "no": 0.2}) == {
        "no": 0.2,
        "yes": 0.8,
    }
    probs, p_u = logits_to_source_distribution(noul, {"yes": 4.0, "no": 1.0, "U": 0.0})
    assert probs["yes"] > probs["no"]
    assert 0.0 < p_u < 1.0


def test_model_payload_excludes_gold_and_provenance_and_state_is_deterministic(
    tmp_path: Path,
) -> None:
    tasks, _, _ = _load_fixture(tmp_path)
    task = next(item for item in tasks if isinstance(item.state, dict))

    row = corpus_row(task)
    payload_text = json.dumps(row["model_payload"], ensure_ascii=False, sort_keys=True)
    assert "expected" not in payload_text
    assert "provenance" not in payload_text
    assert "gold_probs" not in payload_text
    assert build_sample(task).prefix.state == '{"a":{"nested":true},"z":2}'
    DecisionRequest = build_decision_request(task)
    assert DecisionRequest.state == {"a": {"nested": True}, "z": 2}


def test_score_uses_argmax_not_rounded_expected_value_for_score_tasks(tmp_path: Path) -> None:
    tasks, _, _ = _load_fixture(tmp_path)
    task = next(item for item in tasks if item.primitive == "score" and item.expected == 2)
    result = score_task(task, {"0": 0.4, "1": 0.0, "2": 0.2, "3": 0.4})

    assert result.predicted == "0"
    assert result.correct is False
    assert result.ordinal_ev == pytest.approx(1.6)


def test_tie_break_invalid_probs_and_rounded_renormalization(tmp_path: Path) -> None:
    tasks, _, _ = _load_fixture(tmp_path)
    choice = next(item for item in tasks if item.labels == ("alpha", "beta", "gamma"))

    assert argmax_label({"beta": 0.5, "alpha": 0.5}) == "alpha"
    invalid = score_task(choice, {"alpha": 0.9, "beta": 0.2, "gamma": 0.0})
    assert invalid.valid is False

    rounded = score_task(choice, {"alpha": 0.34, "beta": 0.33, "gamma": 0.335})
    assert rounded.valid is True
    assert rounded.strict_valid is False
    assert rounded.renormalized is True


def test_score_predictions_ignores_side_pu_but_rejects_u_inside_official_probs(
    tmp_path: Path,
) -> None:
    tasks, _, _ = _load_fixture(tmp_path)
    predictions: list[dict[str, Any]] = []
    for task in tasks:
        probs = {label: 0.0 for label in task.labels}
        probs[str(task.expected)] = 1.0
        predictions.append({"id": task.id, "probabilities": probs, "p_insufficient": 0.99})

    scored = score_predictions(tasks, predictions)
    assert scored["accuracy"] == 1.0
    assert scored["p_insufficient_in_official_scorer"] is False

    first_probs = predictions[0]["probabilities"]
    assert isinstance(first_probs, dict)
    predictions[0]["probabilities"] = {**first_probs, "U": 0.0}
    scored = score_predictions(tasks, predictions)
    assert scored["results"][tasks[0].id]["valid"] is False
