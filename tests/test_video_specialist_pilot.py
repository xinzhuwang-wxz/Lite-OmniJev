"""Video-only recipe preserves60families and declares120actual repeated exposures."""

import collections
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from train.trainer import LoopConfig

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-video-specialist-pilot.py"
spec = importlib.util.spec_from_file_location("video_specialist_prepare", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def samples():
    return [
        SimpleNamespace(
            id=str(i), meta=SimpleNamespace(pool="train", modality=["video"], family_id=str(i))
        )
        for i in range(60)
    ]


def test_fixed_shuffle_cycles_twice_without_new_families():
    rows = samples()
    draws = module.actual_draws(rows)
    assert draws[:60] == draws[60:]
    assert draws == module.actual_draws(rows)
    assert collections.Counter(s.id for s in draws) == {str(i): 2 for i in range(60)}
    rows[0].meta.pool = "dev"
    with pytest.raises(ValueError, match="TRAIN"):
        module.actual_draws(rows)


def test_only_sampling_config_fields_change():
    original = LoopConfig(
        sample_count=120,
        steps=10,
        accumulation=12,
        modality="mixed",
        media_roots=["/fixture/media"],
        midway_step=5,
        expected_data_versions=["video-v1", "old-v1"],
    ).model_dump(mode="json")
    changed = module.specialist_config(original, "video-v1")
    difference = {k for k in changed if changed[k] != original[k]}
    assert difference == {
        "sample_count",
        "modality",
        "expected_data_version",
        "expected_data_versions",
    }
    assert (changed["steps"], changed["accumulation"]) == (10, 12)
