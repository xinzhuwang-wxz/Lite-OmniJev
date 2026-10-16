"""Only explicitly bounded decoder LoRA capacity settings are admitted."""

import pytest
from pydantic import ValidationError

from train.trainer import LoopConfig, SmokeConfig


@pytest.mark.parametrize("rank,alpha", [(16, 32), (64, 128)])
def test_loop_admits_bounded_rank_with_unchanged_scaling(rank, alpha):
    config = LoopConfig(sample_count=1200, steps=100, accumulation=12, rank=rank, alpha=alpha)
    assert (config.rank, config.alpha) == (rank, alpha)


@pytest.mark.parametrize("rank,alpha", [(16, 128), (64, 32), (32, 64), (8, 16)])
def test_loop_rejects_other_capacity_or_scaling(rank, alpha):
    with pytest.raises(ValidationError):
        LoopConfig(sample_count=1200, steps=100, accumulation=12, rank=rank, alpha=alpha)


def test_legacy_defaults_and_p0_scope_remain_fixed():
    config = LoopConfig(sample_count=100, steps=10, accumulation=10)
    assert (config.rank, config.alpha) == (16, 32)
    with pytest.raises(ValidationError):
        SmokeConfig(rank=64, alpha=128)
