"""Zero added B preserves the mathematical decoder update, including old tensor bits."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / ".scratch/s4/expand-decision-lora-rank.py"
spec = importlib.util.spec_from_file_location("rank_expand", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TARGET = "model.language_model.layers.0.self_attn.q_proj"


def tensors():
    generator = torch.Generator().manual_seed(8)
    return {
        f"base_model.model.{TARGET}.lora_A.weight": torch.randn(16, 5, generator=generator),
        f"base_model.model.{TARGET}.lora_B.weight": torch.randn(3, 16, generator=generator),
    }


def test_delta_prefix_zero_and_deterministic_new_directions():
    source = tensors()
    expanded = module.expand_tensors(source, [TARGET])
    again = module.expand_tensors(source, [TARGET])
    a, b = source.values()
    aa, bb = expanded.values()
    assert module.bit_equal(aa[:16], a) and module.bit_equal(bb[:, :16], b)
    assert torch.count_nonzero(bb[:, 16:]) == 0
    assert torch.count_nonzero(aa[16:]) > 0
    assert torch.allclose(
        2 * bb.double() @ aa.double(), 2 * b.double() @ a.double(), atol=1e-12, rtol=1e-12
    )
    assert all(module.bit_equal(expanded[k], again[k]) for k in expanded)


def test_reject_extra_tensor_nonfinite_or_wrong_rank():
    source = tensors()
    source["vision.lora_A.weight"] = torch.zeros(16, 5)
    with pytest.raises(ValueError, match="exact decoder"):
        module.expand_tensors(source, [TARGET])
    source = tensors()
    next(iter(source.values()))[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module.expand_tensors(source, [TARGET])


def test_nonstandard_scaling_or_rank_patterns_rejected():
    config = {
        "r": 16,
        "lora_alpha": 32,
        "peft_type": "LORA",
        "bias": "none",
        "lora_bias": False,
        "use_rslora": False,
        "use_dora": False,
        "use_qalora": False,
        "fan_in_fan_out": False,
        "lora_dropout": 0.0,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "init_lora_weights": True,
        "inference_mode": True,
        "target_modules": [TARGET],
    }
    assert module.validate_config(config) == [TARGET]
    for key, value in [("use_rslora", True), ("rank_pattern", {"x": 8}), ("bias", "all")]:
        with pytest.raises(ValueError, match="unsupported"):
            module.validate_config({**config, key: value})


def test_parity_checks_all_actions_including_u():
    probe_path = path.with_name("probe-expanded-lora-parity.py")
    probe_spec = importlib.util.spec_from_file_location("rank_probe", probe_path)
    assert probe_spec and probe_spec.loader
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    assert probe.max_probability_difference(
        {"yes": 0.4, "no": 0.5, "U": 0.1}, {"yes": 0.4, "no": 0.49, "U": 0.11}
    ) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="U"):
        probe.max_probability_difference({"yes": 0.4, "no": 0.6}, {"yes": 0.4, "no": 0.6})
