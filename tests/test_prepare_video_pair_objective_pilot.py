"""CPU contract for deriving the pair arm from the actual matched CE config."""

import importlib.util
from pathlib import Path

import pytest

from train.trainer import LoopConfig


def helper():
    path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-video-pair-objective-pilot.py"
    spec = importlib.util.spec_from_file_location("prepare_video_pair_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return LoopConfig(
        sample_count=1200,
        steps=100,
        accumulation=12,
        modality="mixed",
        rank=64,
        alpha=128,
        learning_rate=2e-5,
        image_max_soft_tokens=1120,
        expected_data_versions=["fixed"],
        media_root="/media",
        midway_step=10,
        checkpoint_every=10,
    ).model_dump()


def test_derivation_adds_only_five_bound_fields_and_preserves_ce_cadence(tmp_path):
    module = helper()
    sidecar, ledger = tmp_path / "sidecar.json", tmp_path / "ledger.jsonl"
    sidecar.write_text("sidecar")
    ledger.write_text("ledger")
    original = config()
    derived = module.derive_config(original, sidecar, ledger)
    assert {key: value for key, value in derived.items() if key in original} == original
    assert set(derived) - set(original) == {
        "video_pair_loss_weight",
        "video_pair_sidecar",
        "video_pair_sidecar_sha256",
        "video_pair_ledger",
        "video_pair_ledger_sha256",
    }
    assert derived["video_pair_sidecar_sha256"] == module.sha(sidecar)
    assert derived["video_pair_ledger_sha256"] == module.sha(ledger)
    assert derived["video_pair_loss_weight"] == 0.1
    assert derived["midway_step"] == derived["checkpoint_every"] == 10


@pytest.mark.parametrize(
    "change",
    [
        {"midway_step": 5},
        {"checkpoint_every": 5},
        {"video_pair_loss_weight": 0.1},
        {"rank": 16, "alpha": 32},
    ],
)
def test_changed_control_cannot_silently_define_pair_comparison(tmp_path, change):
    sidecar, ledger = tmp_path / "sidecar", tmp_path / "ledger"
    sidecar.write_text("sidecar")
    ledger.write_text("ledger")
    with pytest.raises(ValueError):
        helper().derive_config(config() | change, sidecar, ledger)
