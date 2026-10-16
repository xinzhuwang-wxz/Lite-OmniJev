"""Controlled comparisons distinguish historical drift from within-process rank differences."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / ".scratch/s4/probe-rank-runtime-control.py"
spec = importlib.util.spec_from_file_location("rank_control", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows(yes=0.6, logit=1.0):
    return {
        str(i): {
            "raw": {"yes": logit, "no": 0.0, "U": -10.0},
            "joint": {"yes": yes, "no": 0.99 - yes, "U": 0.01},
            "conditional": {"yes": yes / 0.99, "no": (0.99 - yes) / 0.99},
            "prediction": "yes",
        }
        for i in range(12)
    }


def test_pairwise_rank_comparisons_separate_drift_and_repeat():
    historical = rows()
    fresh = rows(0.62, 1.125)
    repeat = rows(0.62, 1.125)
    assert module.compare_rows(historical, fresh)["max_legal_plus_u_logit_difference"] == 0.125
    assert module.compare_rows(fresh, repeat)["max_joint_probability_difference"] == 0
    with pytest.raises(ValueError, match="twelve"):
        module.compare_rows(historical, {})


def test_zero_new_a_nonzero_b_is_algebraic_option_not_numeric_proof():
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(16, 5, generator=generator, dtype=torch.float64)
    b = torch.randn(3, 16, generator=generator, dtype=torch.float64)
    aa = torch.cat([a, torch.zeros(48, 5, dtype=torch.float64)]).requires_grad_()
    bb = torch.cat(
        [b, torch.randn(3, 48, generator=generator, dtype=torch.float64)], dim=1
    ).requires_grad_()
    assert torch.allclose(bb @ aa, b @ a, atol=1e-12, rtol=1e-12)
    (bb @ aa).sum().backward()
    assert torch.count_nonzero(aa.grad[16:]) > 0
    assert torch.count_nonzero(bb.grad[:, 16:]) == 0
