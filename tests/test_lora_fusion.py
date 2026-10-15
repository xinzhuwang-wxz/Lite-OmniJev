import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.lora_fusion import approximate, fuse, sha256

KEY = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
BKEY = KEY.replace("lora_A", "lora_B")


def adapter(path: Path, *, tensors=None, config=None):
    path.mkdir()
    save_file(
        tensors or {KEY: torch.randn(16, 40), BKEY: torch.randn(36, 16)},
        str(path / "adapter_model.safetensors"),
    )
    (path / "adapter_config.json").write_text(
        json.dumps(config or {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none"})
    )
    return sha256(path / "adapter_model.safetensors")


def test_dense_optimal_error_and_not_factor_average():
    torch.manual_seed(9)
    a1, a2 = torch.randn(16, 40), torch.randn(16, 40)
    b1, b2 = torch.randn(36, 16), torch.randn(36, 16)
    a, b, metrics = approximate(a1, b1, a2, b2, 0.25)
    dense = 0.75 * b1.double() @ a1.double() + 0.25 * b2.double() @ a2.double()
    expected = torch.linalg.svdvals(dense)[16:].square().sum().sqrt() / torch.linalg.norm(dense)
    observed = torch.linalg.norm(b.double() @ a.double() - dense) / torch.linalg.norm(dense)
    assert float(observed) == pytest.approx(float(expected), abs=1e-7)
    assert metrics["relative_frobenius_error"] == pytest.approx(float(observed), abs=1e-7)


def test_identical_adapter_effective_delta_and_hashes(tmp_path):
    source = tmp_path / "source"
    digest = adapter(source)
    report = fuse(source, source, digest, digest, tmp_path / "out", 0.5)
    old = load_file(str(source / "adapter_model.safetensors"))
    new = load_file(str(tmp_path / "out/adapter_model.safetensors"))
    torch.testing.assert_close(new[BKEY] @ new[KEY], old[BKEY] @ old[KEY], atol=2e-5, rtol=1e-5)
    assert report["promotion_allowed"]
    assert report["output"]["weights_sha256"] == sha256(tmp_path / "out/adapter_model.safetensors")
    assert (tmp_path / "out").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("problem", ["sha", "config", "keys", "nonfinite", "overwrite"])
def test_rejections(tmp_path, problem):
    first, second = tmp_path / "first", tmp_path / "second"
    one = adapter(first)
    tensors = {KEY: torch.randn(16, 40), BKEY: torch.randn(36, 16)}
    if problem == "keys":
        tensors = {k.replace("layers.0", "layers.1"): v for k, v in tensors.items()}
    if problem == "nonfinite":
        tensors[KEY][0, 0] = float("nan")
    config = (
        {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none", "lora_dropout": 0.1}
        if problem == "config"
        else None
    )
    two = adapter(second, tensors=tensors, config=config)
    output = tmp_path / "out"
    if problem == "overwrite":
        output.mkdir()
        (output / "sentinel").write_text("keep")
    with pytest.raises((ValueError, FileExistsError)):
        fuse(first, second, "0" * 64 if problem == "sha" else one, two, output, 0.5)
    if problem == "overwrite":
        assert (output / "sentinel").read_text() == "keep"
    else:
        report = json.loads((output / "fusion_report.json").read_text())
        assert not report["promotion_allowed"]
        assert not (output / "adapter_model.safetensors").exists()


def test_lossy_mix_cannot_promote(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    one, two = adapter(first), adapter(second)
    report = fuse(first, second, one, two, tmp_path / "out", 0.25)
    assert report["status"] == "rejected_residual"
    assert not report["promotion_allowed"]
