from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import s0_development_baselines as baseline


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample(
    index: int, *, pool: str = "dev", target: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "id": f"text-{index:03d}",
        "prefix": {"state": f"state {index}", "media": []},
        "questions": [
            {
                "qid": "q0",
                "type": "choice",
                "instruction": f"Choose for {index}",
                "options": ["alpha", "beta", "gamma"],
                "target": target or {"kind": "hard", "value": "A", "source": "hard_gold"},
            }
        ],
        "meta": {
            "modality": ["text"],
            "source_dataset": "fixture",
            "source_revision": "r1",
            "source_sha256": "1" * 64,
            "license": "MIT",
            "usage": "research",
            "pool": pool,
            "family_id": f"family-{index:03d}",
            "data_version": "fixture-v1",
        },
    }


def _audio_sample(tmp_path: Path) -> dict[str, Any]:
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake audio")
    row = _sample(0)
    row["prefix"]["media"] = [
        {"type": "audio", "uri": "clip.wav", "sha256": hashlib.sha256(b"fake audio").hexdigest()}
    ]
    row["meta"]["modality"] = ["audio"]
    return row


def _write_dev(tmp_path: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    dev = tmp_path / "dev.jsonl"
    dev.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = tmp_path / "build-report.json"
    report.write_text(
        json.dumps({"status": "passed", "outputs": {"dev_jsonl_sha256": _sha(dev)}}),
        encoding="utf-8",
    )
    return dev, report


def _asset_entry(path: Path, relative: str, *, weight: bool = False) -> dict[str, Any]:
    data = (path / relative).read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "required": True,
        "kind": "weights" if weight else "metadata",
        "lfs_sha256": hashlib.sha256(data).hexdigest() if weight else None,
        "git_blob_sha1": None if weight else _git_blob(data),
    }


def _write_assets(tmp_path: Path, model_id: str = "gemma4-e4b-it") -> tuple[Path, Path]:
    root = tmp_path / "models"
    repo = "google/gemma-4-E4B-it" if model_id.startswith("gemma") else "OmniJev/OneJev-4B"
    revision = "e" * 40 if model_id.startswith("gemma") else "b" * 40
    model_type = "gemma4" if model_id.startswith("gemma") else "qwen3_5"
    architecture = (
        "Gemma4ForConditionalGeneration"
        if model_id.startswith("gemma")
        else "Qwen3_5ForConditionalGeneration"
    )
    directory = root / repo.replace("/", "__") / revision
    directory.mkdir(parents=True)
    files = {
        "config.json": json.dumps({"model_type": model_type, "architectures": [architecture]})
        + "\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "chat_template.jinja": "{{ messages[0]['content'] }}\n",
        "processor_config.json": "{}\n",
        "model.safetensors": "fixture weights\n",
    }
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": model_id,
                        "display_name": model_id,
                        "repo_id": repo,
                        "revision": revision,
                        "license": "apache-2.0",
                        "role": "test",
                        "files": [
                            _asset_entry(directory, "config.json"),
                            _asset_entry(directory, "tokenizer.json"),
                            _asset_entry(directory, "tokenizer_config.json"),
                            _asset_entry(directory, "chat_template.jinja"),
                            _asset_entry(directory, "processor_config.json"),
                            _asset_entry(directory, "model.safetensors", weight=True),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, manifest


class FakeArray:
    shape = (1, 3)

    def __init__(self, data: bytes = b"abc") -> None:
        self._data = data
        self.dtype = type("DType", (), {"kind": "i"})()

    def tobytes(self) -> bytes:
        return self._data


class FakeTensor:
    dtype = "torch.int64"

    def __init__(self, ids: list[int], data: bytes | None = None) -> None:
        self._ids = ids
        self._data = data

    def __getitem__(self, index: int) -> FakeTensor:
        assert index == 0
        return self

    def tolist(self) -> list[int]:
        return self._ids

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> FakeArray:
        return FakeArray(self._data or bytes(self._ids[:8]))


class FakeTokenizer:
    def __init__(self) -> None:
        self.ids = {
            "A": 11,
            "B": 12,
            "C": 13,
            "U": 14,
            "yes": 15,
            "no": 16,
            "0": 17,
            "1": 18,
            "2": 19,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text in self.ids:
            return [self.ids[text]]
        for token in sorted(self.ids, key=len, reverse=True):
            if text.endswith(token):
                return [ord(char) % 251 + 100 for char in text[: -len(token)]] + [self.ids[token]]
        return [ord(char) % 251 + 100 for char in text]


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]:
        assert enable_thinking is False
        text = "<chat>" + json.dumps(messages[0]["content"], sort_keys=True) + "<gen>"
        return self.tokenizer.encode(text, add_special_tokens=False) if tokenize else text

    def __call__(self, **kwargs: Any) -> dict[str, FakeTensor]:
        text = cast(list[str], kwargs["text"])[0]
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        return {"input_ids": FakeTensor(self.tokenizer.encode(text, add_special_tokens=False))}


class OverBudgetProcessor(FakeProcessor):
    def __call__(self, **kwargs: Any) -> dict[str, FakeTensor]:
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": FakeTensor([101] * 8193)}


class DriftingInputProcessor(FakeProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, **kwargs: Any) -> dict[str, FakeTensor]:
        self.calls += 1
        text = cast(list[str], kwargs["text"])[0]
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return {"input_ids": FakeTensor(ids, bytes([self.calls % 251]))}


def _fake_forward(item: baseline.PreparedSample) -> dict[int, float]:
    ids = list(item.answer_token_ids.values())
    weights = [math.exp(-(index + 1)) for index, _ in enumerate(ids)]
    total = math.fsum(weights)
    return {
        token_id: math.log(weight / total) for token_id, weight in zip(ids, weights, strict=True)
    }


def _fake_factory(
    model_path: Path, model_id: str, dtype: str
) -> tuple[baseline.ForwardFn, Callable[[], None]]:
    return _fake_forward, lambda: None


def test_run_scores_500_dev_rows_and_preserves_records(tmp_path: Path) -> None:
    dev, build = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    model_root, manifest = _write_assets(tmp_path)
    sidecar = tmp_path / "records.jsonl"

    report = baseline.run_development_report(
        build_report=build,
        dev_jsonl=dev,
        media_root=tmp_path,
        cohort="text",
        asset_manifest=manifest,
        model_root=model_root,
        model_id="gemma4-e4b-it",
        processor_factory=lambda _: FakeProcessor(),
        forward_factory=_fake_factory,
        fake_forward=True,
        records_sidecar=sidecar,
    )

    assert report["status"] == "passed"
    assert report["dataset"]["rows"] == 500
    assert report["metrics_pass_1"]["count"] == 500
    assert report["repeatability"]["passed"] is True
    assert report["records_sidecar"]["records"] == 1000
    assert len(report["records"]) == 500
    first = report["records"][0]
    assert set(first["joint_k_plus_u"]) == {"A", "B", "C", "U"}
    assert math.isclose(sum(first["conditional_without_u"].values()), 1.0)
    assert "input_summary" in first


def test_hash_binding_refuses_wrong_dev_jsonl(tmp_path: Path) -> None:
    dev, build = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    dev.write_text(dev.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(baseline.DevelopmentBaselineError, match="dev_jsonl_sha256"):
        baseline.bind_build_report(build, dev)


def test_schema_guard_requires_exact_500_unique_dev_hard_rows(tmp_path: Path) -> None:
    rows = [_sample(index) for index in range(499)]
    samples = baseline.load_samples(_write_dev(tmp_path, rows)[0])
    with pytest.raises(baseline.DevelopmentBaselineError, match="expected 500"):
        baseline.validate_samples(samples, cohort="text")

    rows = [_sample(index) for index in range(500)]
    rows[0]["meta"]["pool"] = "train"
    samples = baseline.load_samples(_write_dev(tmp_path, rows)[0])
    with pytest.raises(baseline.DevelopmentBaselineError, match="pool=dev"):
        baseline.validate_samples(samples, cohort="text")

    rows = [_sample(index) for index in range(500)]
    rows[0]["questions"][0]["target"] = {"kind": "soft", "probs": [0.7, 0.2, 0.1], "source": "t"}
    samples = baseline.load_samples(_write_dev(tmp_path, rows)[0])
    with pytest.raises(baseline.DevelopmentBaselineError, match="soft gold"):
        baseline.validate_samples(samples, cohort="text")


def test_onejev_audio_is_unsupported_before_gpu_or_assets(tmp_path: Path) -> None:
    with pytest.raises(baseline.UnsupportedDevelopmentBaseline, match="OneJev audio"):
        baseline.run_development_report(
            build_report=tmp_path / "missing.json",
            dev_jsonl=tmp_path / "missing.jsonl",
            media_root=tmp_path,
            cohort="audio",
            asset_manifest=tmp_path / "missing-assets.json",
            model_root=tmp_path / "models",
            model_id="onejev-4b",
        )


def test_nonfinite_forward_is_rejected_with_item_context(tmp_path: Path) -> None:
    dev, build = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    model_root, manifest = _write_assets(tmp_path)

    def bad_forward(item: baseline.PreparedSample) -> dict[int, float]:
        raw = dict(_fake_forward(item))
        raw[next(iter(raw))] = float("nan")
        return raw

    def bad_factory(
        model_path: Path, model_id: str, dtype: str
    ) -> tuple[baseline.ForwardFn, Callable[[], None]]:
        return bad_forward, lambda: None

    with pytest.raises(baseline.DevelopmentBaselineError, match="item_id=text-000 phase=forward"):
        baseline.run_development_report(
            build_report=build,
            dev_jsonl=dev,
            media_root=tmp_path,
            cohort="text",
            asset_manifest=manifest,
            model_root=model_root,
            model_id="gemma4-e4b-it",
            processor_factory=lambda _: FakeProcessor(),
            forward_factory=bad_factory,
            fake_forward=True,
        )


def test_completed_records_survive_forward_failure(tmp_path: Path) -> None:
    dev, build = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    model_root, manifest = _write_assets(tmp_path)
    sidecar = tmp_path / "partial.jsonl"
    calls = {"count": 0}

    def failing_forward(item: baseline.PreparedSample) -> dict[int, float]:
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("boom")
        return _fake_forward(item)

    def failing_factory(
        model_path: Path, model_id: str, dtype: str
    ) -> tuple[baseline.ForwardFn, Callable[[], None]]:
        return failing_forward, lambda: None

    with pytest.raises(baseline.DevelopmentBaselineError, match="boom"):
        baseline.run_development_report(
            build_report=build,
            dev_jsonl=dev,
            media_root=tmp_path,
            cohort="text",
            asset_manifest=manifest,
            model_root=model_root,
            model_id="gemma4-e4b-it",
            processor_factory=lambda _: FakeProcessor(),
            forward_factory=failing_factory,
            fake_forward=True,
            records_sidecar=sidecar,
        )

    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["id"] for line in lines] == ["text-000", "text-001"]


def test_overbudget_processor_ids_fail_with_item_context_and_preserve_records(
    tmp_path: Path,
) -> None:
    dev, build = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    model_root, manifest = _write_assets(tmp_path)
    sidecar = tmp_path / "partial.jsonl"

    with pytest.raises(
        baseline.DevelopmentBaselineError,
        match="item_id=text-000 phase=prepare pass=1: item_id=text-000 prompt token budget",
    ):
        baseline.run_development_report(
            build_report=build,
            dev_jsonl=dev,
            media_root=tmp_path,
            cohort="text",
            asset_manifest=manifest,
            model_root=model_root,
            model_id="gemma4-e4b-it",
            processor_factory=lambda _: OverBudgetProcessor(),
            forward_factory=_fake_factory,
            fake_forward=True,
            records_sidecar=sidecar,
        )

    assert not sidecar.exists()


def test_repeatability_binds_processor_tensor_summary(tmp_path: Path) -> None:
    dev, build = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    model_root, manifest = _write_assets(tmp_path)
    processor = DriftingInputProcessor()

    report = baseline.run_development_report(
        build_report=build,
        dev_jsonl=dev,
        media_root=tmp_path,
        cohort="text",
        asset_manifest=manifest,
        model_root=model_root,
        model_id="gemma4-e4b-it",
        processor_factory=lambda _: processor,
        forward_factory=_fake_factory,
        fake_forward=True,
    )

    assert report["status"] == "failed"
    assert report["repeatability"]["passed"] is False
    assert report["repeatability"]["first_mismatches"][0]["kind"] == "input_summary_sha256"


def test_rubric_options_are_rendered_in_processor_prompt(tmp_path: Path) -> None:
    dev, _ = _write_dev(tmp_path, [_sample(index) for index in range(500)])
    sample = baseline.load_samples(dev)[0]
    messages, kwargs, identity = baseline.build_messages_and_media(
        sample, sample.questions[0], tmp_path
    )

    assert kwargs["add_special_tokens"] is False
    assert kwargs["truncation"] is False
    assert messages[0]["content"][-1]["type"] == "text"
    payload = messages[0]["content"][-1]["text"]
    assert '"A":"alpha"' in payload
    assert '"B":"beta"' in payload
    assert '"U":"Insufficient evidence to decide"' in payload
    assert identity["payload"] == payload


def test_audio_uses_canonical_pcm_roundtrip_waveform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = baseline.load_samples(_write_dev(tmp_path, [_audio_sample(tmp_path)])[0])[0]

    class Canonical:
        hf_value = "canonical-waveform"
        summary = {"type": "audio", "canonical_pcm16_sha256": "2" * 64}

    monkeypatch.setattr(baseline, "load_media", lambda *args, **kwargs: "raw-waveform")
    monkeypatch.setattr(baseline, "canonicalize_media", lambda *args: Canonical())

    messages, kwargs, identity = baseline.build_messages_and_media(
        sample, sample.questions[0], tmp_path
    )

    assert kwargs["audio"] == ["canonical-waveform"]
    assert kwargs["sampling_rate"] == 16000
    assert identity["canonical_media"] == [Canonical.summary]
    assert messages[0]["content"][0] == {"type": "audio"}
