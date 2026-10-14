from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from data.jevbench import load_public_tasks, sha256_file
from scripts import s0_public_baselines as baseline


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


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    import hashlib

    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _write_model_assets(tmp_path: Path, *, include_weights: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "models"
    manifest_models: list[dict[str, Any]] = []
    for model_id, repo, revision in [
        ("gemma4-e4b-it", "google/gemma-4-E4B-it", "e" * 40),
        ("gemma4-e2b-it", "google/gemma-4-E2B-it", "d" * 40),
    ]:
        directory = root / repo.replace("/", "__") / revision
        directory.mkdir(parents=True)
        files = {
            "tokenizer.json": "{}\n",
            "tokenizer_config.json": json.dumps({"chat_template": "{{ messages[0]['content'] }}"})
            + "\n",
            "chat_template.jinja": "{{ messages[0]['content'] }}\n",
            "config.json": json.dumps(
                {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}
            )
            + "\n",
        }
        if include_weights:
            files["model.safetensors"] = "not real safetensors, hash contract only\n"
        entries = []
        for name, text in files.items():
            path = directory / name
            path.write_text(text, encoding="utf-8")
            entries.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "required": True,
                    "kind": "weights" if name.endswith(".safetensors") else "metadata",
                    "lfs_sha256": sha256_file(path) if name.endswith(".safetensors") else None,
                    "git_blob_sha1": None if name.endswith(".safetensors") else _git_blob(path),
                }
            )
        manifest_models.append(
            {
                "id": model_id,
                "display_name": model_id,
                "repo_id": repo,
                "revision": revision,
                "license": "apache-2.0",
                "role": "test",
                "files": entries,
            }
        )
    manifest = tmp_path / "project-model-assets.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "models": manifest_models}), encoding="utf-8"
    )
    return root, manifest


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        assert enable_thinking is False
        assert add_generation_prompt is True
        content = cast(str, messages[0]["content"])
        text = "<chat>" + content + "<gen>"
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        answer_tokens: dict[str, int] = {
            "yes": 2001,
            "no": 2002,
            **{ch: 1000 + ord(ch) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZU0123456789"},
        }
        if text in answer_tokens:
            return [answer_tokens[text]]
        for token in sorted(answer_tokens, key=len, reverse=True):
            if text.endswith(token):
                return [ord(ch) % 251 + 1 for ch in text[: -len(token)]] + [answer_tokens[token]]
        return [ord(ch) % 251 + 1 for ch in text]

    def __call__(
        self, text: str, *, return_tensors: str, add_special_tokens: bool = True
    ) -> dict[str, Any]:
        return {"input_ids": [self.encode(text, add_special_tokens=add_special_tokens)]}


def _fake_forward(item: baseline.PreparedQuestion) -> dict[int, float]:
    ids = list(item.answer_token_ids.values())
    weights = [math.exp(-(index + 1)) for index, _ in enumerate(ids)]
    total = math.fsum(weights)
    return {
        token_id: math.log(weight / total) for token_id, weight in zip(ids, weights, strict=True)
    }


def _fake_factory(
    model_path: Path, tokenizer: baseline.ChatTokenizer, dtype: str
) -> tuple[baseline.ForwardFn, Callable[[], None]]:
    return _fake_forward, lambda: None


def test_preflight_preserves_noul_rubric_and_keeps_gold_out_of_prompt(tmp_path: Path) -> None:
    source_root, source_config = _write_source(tmp_path)
    model_root, manifest = _write_model_assets(tmp_path)
    report = baseline.preflight_report(
        source_root=source_root,
        source_config=source_config,
        asset_manifest=manifest,
        model_root=model_root,
        model_id="gemma4-e4b-it",
        tokenizer_factory=lambda _: FakeTokenizer(),
    )

    assert report["status"] == "passed"
    assert report["acceptance"]["release_or_stage_acceptance"] is False
    noul_task = next(
        task
        for task in load_public_tasks(source_root, json.loads(source_config.read_text()))
        if task.primitive == "noul"
    )
    prepared = baseline.build_prepared_question(noul_task, FakeTokenizer())
    assert "Answer meanings" in prepared.payload
    assert "The statement is satisfied." in prepared.payload
    assert "The statement is not satisfied." in prepared.payload
    assert "gold_probs" not in prepared.payload
    assert "provenance" not in prepared.payload


def test_run_fake_forward_scores_full_231_but_never_counts_as_acceptance(tmp_path: Path) -> None:
    source_root, source_config = _write_source(tmp_path)
    model_root, manifest = _write_model_assets(tmp_path, include_weights=True)

    sidecar = tmp_path / "records.jsonl"
    report = baseline.run_baseline_report(
        source_root=source_root,
        source_config=source_config,
        asset_manifest=manifest,
        model_root=model_root,
        model_id="gemma4-e2b-it",
        tokenizer_factory=lambda _: FakeTokenizer(),
        forward_factory=_fake_factory,
        fake_forward=True,
        records_sidecar=sidecar,
    )

    assert report["status"] == "passed"
    assert len(report["records"]) == 231
    assert report["repeatability"]["passed"] is True
    assert report["score_pass_1"]["public_decisions"] == 231
    assert report["acceptance"]["release_or_stage_acceptance"] is False
    assert report["acceptance"]["fake_forward_control_only"] is True
    assert report["records_sidecar"]["records"] == 462
    lines = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert lines[0]["pass_index"] == 1
    assert lines[-1]["pass_index"] == 2


def test_repeatability_failure_reports_failed_without_shrinking_denominator(tmp_path: Path) -> None:
    source_root, source_config = _write_source(tmp_path)
    model_root, manifest = _write_model_assets(tmp_path, include_weights=True)
    calls = {"n": 0}

    def drifting_forward(item: baseline.PreparedQuestion) -> dict[int, float]:
        calls["n"] += 1
        raw = dict(_fake_forward(item))
        if calls["n"] > 231:
            first = next(iter(raw))
            raw[first] -= 0.001
        return raw

    def drifting_factory(
        model_path: Path, tokenizer: baseline.ChatTokenizer, dtype: str
    ) -> tuple[baseline.ForwardFn, Callable[[], None]]:
        return drifting_forward, lambda: None

    report = baseline.run_baseline_report(
        source_root=source_root,
        source_config=source_config,
        asset_manifest=manifest,
        model_root=model_root,
        model_id="gemma4-e4b-it",
        tokenizer_factory=lambda _: FakeTokenizer(),
        forward_factory=drifting_factory,
        fake_forward=True,
    )

    assert report["status"] == "failed"
    assert len(report["records"]) == 231
    assert report["repeatability"]["passed"] is False
    assert report["score_pass_1"]["public_decisions"] == 231


def test_unsupported_model_is_notimplemented(tmp_path: Path) -> None:
    _, manifest = _write_model_assets(tmp_path)
    data = baseline.load_asset_manifest(manifest)
    with pytest.raises(NotImplementedError, match="supports only"):
        baseline.select_supported_model(data, "valen-preview-0923")


def test_run_requires_weight_hashes_before_model_load(tmp_path: Path) -> None:
    source_root, source_config = _write_source(tmp_path)
    model_root, manifest = _write_model_assets(tmp_path, include_weights=False)
    with pytest.raises(baseline.PublicBaselineError, match="safetensors inventory"):
        baseline.run_baseline_report(
            source_root=source_root,
            source_config=source_config,
            asset_manifest=manifest,
            model_root=model_root,
            model_id="gemma4-e4b-it",
            tokenizer_factory=lambda _: FakeTokenizer(),
            forward_factory=_fake_factory,
            fake_forward=True,
        )


def test_prepared_model_inputs_use_verified_ids_without_tokenizer_call(tmp_path: Path) -> None:
    source_root, source_config = _write_source(tmp_path)
    task = load_public_tasks(source_root, json.loads(source_config.read_text()))[0]
    prepared = baseline.build_prepared_question(task, FakeTokenizer())

    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(value: list[list[int]], *, device: str, dtype: str) -> dict[str, object]:
            return {"value": value, "device": device, "dtype": dtype}

        @staticmethod
        def ones_like(value: object) -> dict[str, object]:
            return {"ones_like": value}

    inputs = baseline.prepared_model_inputs(prepared, FakeTorch())

    assert inputs["input_ids"] == {
        "value": [prepared.input_ids],
        "device": "cuda",
        "dtype": "long",
    }
    assert inputs["attention_mask"] == {"ones_like": inputs["input_ids"]}


def test_sidecar_keeps_completed_records_when_forward_raises(tmp_path: Path) -> None:
    source_root, source_config = _write_source(tmp_path)
    model_root, manifest = _write_model_assets(tmp_path, include_weights=True)
    sidecar = tmp_path / "partial.jsonl"
    calls = {"n": 0}

    def failing_forward(item: baseline.PreparedQuestion) -> dict[int, float]:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return _fake_forward(item)

    def failing_factory(
        model_path: Path, tokenizer: baseline.ChatTokenizer, dtype: str
    ) -> tuple[baseline.ForwardFn, Callable[[], None]]:
        return failing_forward, lambda: None

    with pytest.raises(RuntimeError, match="boom"):
        baseline.run_baseline_report(
            source_root=source_root,
            source_config=source_config,
            asset_manifest=manifest,
            model_root=model_root,
            model_id="gemma4-e4b-it",
            tokenizer_factory=lambda _: FakeTokenizer(),
            forward_factory=failing_factory,
            fake_forward=True,
            records_sidecar=sidecar,
        )

    lines = sidecar.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["pass_index"] for line in lines] == [1, 1]
