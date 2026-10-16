"""Readiness draft configuration checks, not GPU or model quality evidence."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def helper(monkeypatch):
    path = Path(__file__).resolve().parents[1] / ".scratch/s6/e2b-training-readiness.py"
    with monkeypatch.context() as patch:
        for key in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            patch.setenv(key, "test-placeholder")
        spec = importlib.util.spec_from_file_location("e2b_readiness", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def source():
    return {
        "sample_count": 1200,
        "steps": 100,
        "accumulation": 12,
        "modality": "mixed",
        "media_roots": ["/existing/media"],
        "expected_data_versions": ["existing-train-v1"],
        "learning_rate": 2e-5,
        "preparation_mode": "lazy",
        "rank": 16,
        "alpha": 32,
    }


def test_draft_preserves_course_and_zero_experimental_objectives(helper):
    original = source()
    config = helper.draft_config(original)
    assert config.sample_count == 1200 and config.steps * config.accumulation == 1200
    assert config.learning_rate == original["learning_rate"]
    assert config.rank == 16 and config.alpha == 32
    assert config.checkpoint_every == 5 and config.midway_step == 10
    assert config.reference_kl_weight == config.brier_weight == config.video_pair_loss_weight == 0
    assert "checkpoint_every" not in original


@pytest.mark.parametrize(
    "update", [{"sample_count": 500}, {"accumulation": 10}, {"brier_weight": 0.5}]
)
def test_wrong_or_experimental_course_rejected(helper, update):
    with pytest.raises(ValueError):
        helper.draft_config({**source(), **update})
