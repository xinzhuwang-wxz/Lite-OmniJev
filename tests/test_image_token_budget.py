"""Official image budget routing; CPU fixtures do not prove native GPU parity."""

import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scripts import s1_dev_eval
from serve.gateway.app import Settings, create_app
from train import trainer
from train.monitoring import SwanLabConfig
from train.multimodal import configure_image_processor, native_processor_kwargs


@pytest.mark.parametrize("budget", [70, 140, 280, 560, 1120])
def test_official_image_only_setting(budget):
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(max_soft_tokens=280),
        video_processor=SimpleNamespace(max_soft_tokens=70),
        audio_processor=object(),
    )
    assert configure_image_processor(processor, None) == 280
    assert configure_image_processor(processor, budget) == budget
    assert processor.image_processor.max_soft_tokens == budget
    assert processor.video_processor.max_soft_tokens == 70
    assert native_processor_kwargs({}, budget) == {"images_kwargs": {"max_soft_tokens": budget}}
    with pytest.raises(ValueError, match="video"):
        native_processor_kwargs({"video": {"fps": 2}}, budget)


@pytest.mark.parametrize("budget", [0, 281, 1121, "1120", True])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        trainer.LoopConfig(sample_count=2, steps=1, accumulation=1, image_max_soft_tokens=budget)
    with pytest.raises(ValueError):
        configure_image_processor(object(), budget)
    with pytest.raises(ValueError):
        Settings(image_max_soft_tokens=budget)


@pytest.mark.parametrize(
    "modality,mode", [("image", "eager"), ("mixed", "eager"), ("mixed", "lazy")]
)
@pytest.mark.parametrize("budget", [None, 1120])
def test_training_loaders_apply_budget_before_preparation(
    monkeypatch, tmp_path, modality, mode, budget
):
    samples = trainer.smoke_samples(trainer.SmokeConfig())[:2]
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_soft_tokens=280))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor)
        ),
    )
    monkeypatch.setattr(trainer, "OfficialTokenizerAdapter", lambda *args: object())
    monkeypatch.setattr(trainer, "smoke_samples", lambda *args, **kwargs: samples)
    monkeypatch.setattr(trainer, "verified_assets", lambda *args: {})
    monkeypatch.setattr(trainer, "runtime_requirements", lambda: {})
    observed = []

    def prepare(*args, **kwargs):
        observed.append(processor.image_processor.max_soft_tokens)
        return {"input_ids": [1, 2]}

    monkeypatch.setattr(trainer, "prepare_media_example", prepare)
    monkeypatch.setattr(trainer, "prepare_mixed_example", prepare)
    config = trainer.LoopConfig(
        sample_count=2,
        steps=1,
        accumulation=1,
        modality=modality,
        media_root=str(tmp_path),
        preparation_mode=mode,
        expected_data_versions=["v1"] if modality == "mixed" else None,
        image_max_soft_tokens=budget,
    )
    report = trainer.run(
        config,
        tmp_path / ("a" * 40),
        tmp_path / "manifest",
        tmp_path / "output",
        execute=False,
        monitoring=SwanLabConfig(mode="disabled", failure_policy="fail", logdir=tmp_path),
    )
    assert report["status"] == "ready", report
    assert observed and set(observed) == {budget or 280}
    assert report["config"].get("image_max_soft_tokens") == budget
    if budget is None:
        assert "image_max_soft_tokens" not in report["config"]
        assert "image_max_soft_tokens" not in config.model_dump()


def test_gateway_env_and_startup_processor(monkeypatch, tmp_path):
    monkeypatch.setenv("LITE_IMAGE_MAX_SOFT_TOKENS", "1120")
    assert Settings.from_env().image_max_soft_tokens == 1120
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_soft_tokens=280))
    with TestClient(
        create_app(Settings(media_root=tmp_path, image_max_soft_tokens=1120), processor=processor)
    ):
        assert processor.image_processor.max_soft_tokens == 1120
    with pytest.raises(ValueError, match="canonical media processor"):
        with TestClient(create_app(Settings(image_max_soft_tokens=1120))):
            pass
    monkeypatch.setenv("LITE_IMAGE_MAX_SOFT_TOKENS", "300")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_eval_invalid_profile_leaves_requested_failure_evidence(tmp_path):
    report = tmp_path / "report.json"
    assert (
        s1_dev_eval.run(
            model_dir=tmp_path,
            adapter_dir=None,
            data_path=tmp_path / "data",
            output_dir=tmp_path / "out",
            report_out=report,
            max_length=2048,
            passes=1,
            command=["not-a-source-of-configuration"],
            image_max_soft_tokens=300,
        )
        == 1
    )
    evidence = json.loads(report.read_text())
    assert evidence["image_max_soft_tokens"] == {"requested": 300, "effective": None}


@pytest.mark.parametrize("budget,expected", [(None, 280), (1120, 1120)])
def test_eval_reports_actual_loaded_processor_budget(monkeypatch, tmp_path, budget, expected):
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_soft_tokens=280))
    monkeypatch.setattr(
        s1_dev_eval,
        "load_samples",
        lambda *args, **kwargs: [SimpleNamespace(prefix=SimpleNamespace(media=[object()]))],
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(use_deterministic_algorithms=lambda *args: None, bfloat16="bfloat16"),
    )

    def stop_before_gpu(*args, **kwargs):
        assert processor.image_processor.max_soft_tokens == expected
        raise ValueError("bounded CPU test stops before GPU model loading")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=stop_before_gpu),
        ),
    )
    report = tmp_path / "report.json"
    assert (
        s1_dev_eval.run(
            model_dir=tmp_path,
            adapter_dir=None,
            data_path=tmp_path / "data",
            output_dir=tmp_path / "out",
            report_out=report,
            max_length=2048,
            passes=1,
            command=["--image-max-soft-tokens", "70"],
            image_max_soft_tokens=budget,
        )
        == 1
    )
    assert json.loads(report.read_text())["image_max_soft_tokens"] == {
        "requested": budget,
        "effective": expected,
    }
