"""CPU bounded preparation behavior; not real-model/GPU parity evidence."""

import json
import random
from types import SimpleNamespace

import pytest

from data.schema import Media
from train.monitoring import SwanLabConfig
from train.trainer import BoundedPreparedExamples, LoopConfig, SmokeConfig, run, smoke_samples


def test_cache_bounds_revisits_and_oversize_preserve_numerical_inputs():
    torch = pytest.importorskip("torch")
    samples = smoke_samples(SmokeConfig())[:6]
    calls = []

    def prepare(sample):
        index = samples.index(sample)
        calls.append(index)
        # A tiny view retains the entire backing storage; accounting must include it.
        tensor = torch.arange(40 if index == 5 else 8, dtype=torch.float32)[:3]
        return {"input_ids": [index, 2], "pixel_values": tensor}

    report = {}
    examples = BoundedPreparedExamples(samples, prepare, 64, report)
    order = list(range(len(samples)))
    random.Random(42).shuffle(order)
    for index in order * 2:
        actual = examples[index]
        expected = prepare(samples[index])
        assert actual["input_ids"] == expected["input_ids"]
        assert torch.equal(actual["pixel_values"], expected["pixel_values"])
        assert report["preparation"]["cache_retained_bytes"] <= 64
    assert report["preparation"]["cache_peak_bytes"] == 64
    before = len(calls)
    examples[5]
    examples[5]
    assert len(calls) == before + 2  # oversized example is never retained
    assert report["preparation"]["prepared_examples"] == 6
    assert report["preparation"]["observed_max_prompt_tokens"] == 2
    with pytest.raises(IndexError):
        examples[len(samples)]


def test_lazy_preflight_partial_and_deferred_failure_preserves_run_evidence(monkeypatch, tmp_path):
    samples = smoke_samples(SmokeConfig())[:3]
    monkeypatch.setattr("train.trainer.smoke_samples", lambda *args, **kwargs: samples)
    monkeypatch.setattr("train.trainer.verified_assets", lambda *args: {})
    monkeypatch.setattr("train.trainer.runtime_requirements", lambda: {})
    monkeypatch.setattr("train.trainer.OfficialTokenizerAdapter", lambda *args: object())
    import sys

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
        ),
    )
    visited = []

    def prepare(sample, *args, **kwargs):
        visited.append(sample.id)
        if sample is samples[2]:
            raise ValueError("empty or over-length multimodal prompt; truncation is forbidden")
        return {"input_ids": [1, 2, 3]}

    monkeypatch.setattr("train.trainer.prepare_mixed_example", prepare)
    config = LoopConfig(
        sample_count=3,
        steps=2,
        accumulation=1,
        modality="mixed",
        media_root=str(tmp_path),
        expected_data_versions=["v1"],
        preparation_mode="lazy",
        preparation_cache_bytes=0,
    )
    output = tmp_path / "run"
    kwargs = dict(
        model_dir=tmp_path / ("a" * 40),
        manifest_path=tmp_path / "manifest",
        output_dir=output,
        monitoring=SwanLabConfig(mode="disabled", failure_policy="fail", logdir=tmp_path),
    )
    report = run(config, execute=False, **kwargs)
    assert report["status"] == "ready", report
    assert visited == [samples[0].id]
    assert report["examples"] == 3
    assert "max_prompt_tokens" not in report
    assert report["preparation"]["prepared_examples"] == 1
    assert report["preparation"]["media_validation"].startswith("partial")
    assert report["config"]["preparation_mode"] == "lazy"

    def train(config, model_dir, examples, output_dir, report, telemetry, **kwargs):
        # Model an existing completed checkpoint before accessing a corrupt later row.
        checkpoint = output_dir / "checkpoint-proof"
        checkpoint.write_text("completed step")
        report["checkpoint"] = str(checkpoint)
        report["steps_completed"] = 1
        examples[2]

    monkeypatch.setattr("train.trainer.train_smoke", train)
    report = run(config, execute=True, **kwargs)
    assert report["status"] == "failed", report
    assert samples[2].id in report["error"]
    assert "over-length" in report["error"]
    assert report["steps_completed"] == 1
    assert (output / "checkpoint-proof").read_text() == "completed step"
    assert json.loads((output / "run.json").read_text())["preparation"]["failed_sample_index"] == 2


def test_lazy_mode_rejects_text_only_config():
    with pytest.raises(ValueError, match="reserved for media"):
        LoopConfig(sample_count=2, steps=1, accumulation=1, preparation_mode="lazy")


def test_representatives_visit_each_modality_once_and_cache_hits_do_not_reprepare():
    samples = smoke_samples(SmokeConfig())[:8]
    for sample, kind in zip(samples, ["text", "image", "audio", "video"] * 2, strict=True):
        if kind != "text":
            sample.prefix.media = [Media(type=kind, uri="fixture", sha256="a" * 64)]
    calls = []

    def prepare(sample):
        calls.append(sample.id)
        return {"input_ids": [1, 2]}

    report = {}
    examples = BoundedPreparedExamples(samples, prepare, 64, report)
    examples.validate_representatives()
    assert calls == [sample.id for sample in samples[:4]]
    examples[0]
    assert len(calls) == 4
    assert report["preparation"]["preflight_prepared_examples"] == 4
