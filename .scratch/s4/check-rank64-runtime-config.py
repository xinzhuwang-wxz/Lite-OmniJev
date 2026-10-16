"""CPU-only generic rank64 checks; no Gemma forward or runtime parity claim."""

import argparse
import importlib.metadata
import json


def peft_check():
    import torch
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = torch.nn.Sequential(torch.nn.Linear(96, 96, bias=False))
    model = get_peft_model(
        base, LoraConfig(r=64, lora_alpha=128, target_modules=["0"], bias="none")
    )
    layer = model.base_model.model[0]
    a = torch.randn(16, 96)
    b = torch.randn(96, 16)
    with torch.no_grad():
        layer.lora_A["default"].weight[:16].copy_(a)
        layer.lora_B["default"].weight.zero_()
        layer.lora_B["default"].weight[:, :16].copy_(b)
    x = torch.randn(2, 96)
    delta = layer.lora_B["default"](layer.lora_A["default"](x)) * layer.scaling["default"]
    error = (delta - ((x @ a.T) @ b.T) * 2).abs().max().item()
    assert error < 1e-4 and layer.scaling["default"] == 2
    delta.square().mean().backward()
    assert layer.lora_A["default"].weight.grad[16:].abs().max().item() == 0
    assert layer.lora_B["default"].weight.grad[:, 16:].abs().max().item() > 0
    return {
        "generic_peft_rank64": True,
        "scale": layer.scaling["default"],
        "toy_fp32_max_abs_error": error,
        "new_A_initial_gradient_zero": True,
        "new_B_initial_gradient_nonzero": True,
        "gemma_forward_tested": False,
    }


def vllm_check():
    from vllm.config.lora import LoRAConfig
    from vllm.lora.peft_helper import PEFTHelper

    helper = PEFTHelper(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    helper.validate_legal(LoRAConfig(max_lora_rank=64, max_loras=1, lora_dtype="bfloat16"))
    try:
        helper.validate_legal(LoRAConfig())
    except ValueError:
        pass
    else:
        raise AssertionError("rank64 accepted by default16")
    return {
        "rank64_config_and_peft_helper": True,
        "default16_rejects_rank64": True,
        "scale": helper.vllm_lora_scaling_factor,
        "kernel_or_gemma_forward_tested": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["peft", "vllm"])
    a = p.parse_args()
    result = peft_check() if a.mode == "peft" else vllm_check()
    result.update(
        {
            "versions": {
                n: importlib.metadata.version(n) for n in ["torch", "peft", "transformers"]
            },
            "cuda_requested": False,
            "gpu_forward": False,
            "cost": None,
        }
    )
    if a.mode == "vllm":
        result["versions"]["vllm"] = importlib.metadata.version("vllm")
    print(json.dumps(result))
