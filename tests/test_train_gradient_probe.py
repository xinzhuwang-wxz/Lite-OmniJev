"""CPU evidence for bounded, weight-preserving gold gradient diagnostics."""

import json
from types import SimpleNamespace

import pytest

from scripts.train_gradient_probe import GROUPS, group, main, metrics, probe
from train.trainer import LoopConfig


def test_group_uses_prepared_inputs_and_rejects_mixed_video():
    assert group({"observed_modalities": ["video"]}) == "other"
    assert group({"pixel_values_videos": 1}) == "video"
    assert group({"pixel_values": 1, "input_features": 1}) == "crossmodal"
    with pytest.raises(ValueError, match="mixed video"):
        group({"pixel_values_videos": 1, "pixel_values": 1})


def test_vectors_keep_zero_norm_null_and_detect_conflict():
    torch = pytest.importorskip("torch")
    vectors = {
        "video": torch.tensor([1.0, 0.0]),
        "crossmodal": torch.tensor([-2.0, 0.0]),
        "other": torch.zeros(2),
    }
    result = metrics(vectors, dict.fromkeys(GROUPS, 0), dict.fromkeys(GROUPS, 0.0))
    assert result["pairs"]["video:crossmodal"]["cosine"] == -1
    assert result["pairs"]["total:crossmodal"]["dot"] == 2
    assert result["pairs"]["video:other"]["cosine"] is None
    assert result["mean_gold_ce"]["other"] is None
    vectors["video"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        metrics(vectors, dict.fromkeys(GROUPS, 0), dict.fromkeys(GROUPS, 0.0))


def test_probe_is_bounded_and_does_not_update_weights():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))
            self.dropout = torch.nn.Dropout(0.9)

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.dropout(self.lora_weight).reshape(1, 1, 3))

    model = Model()
    original = model.lora_weight.detach().clone()
    config = LoopConfig(sample_count=36, steps=30, accumulation=12)
    example = {"input_ids": [1], "token_ids": [0, 1, 2], "target": [1.0, 0.0, 0.0]}
    result = probe(config, [example] * 36, model, "cpu")
    assert result["draws"] == 36
    assert len(result["steps"]) == 3
    assert result["aggregate"]["counts"] == {"video": 0, "crossmodal": 0, "other": 36}
    assert len(result["draw_bindings"]) == 36
    expected_gradient = original.softmax(-1) - torch.tensor([1.0, 0.0, 0.0])
    assert result["steps"][0]["gradient_norm"]["other"] == pytest.approx(
        float(expected_gradient.norm()), rel=1e-6
    )
    assert result["denominator"] == 12
    assert result["optimizer_updates"] == 0
    assert torch.equal(model.lora_weight, original)
    assert model.lora_weight.grad is None
    assert not model.training
    assert result == probe(config, [example] * 36, model, "cpu")


def test_identity_failure_writes_receipt_without_cuda(tmp_path):
    output = tmp_path / "receipt.json"
    args = []
    for name in ("config", "data", "ledger", "model-dir", "asset-manifest", "init-adapter"):
        args += ["--" + name, str(tmp_path / name)]
    for name in ("config", "data", "ledger", "asset-manifest", "adapter", "adapter-config"):
        args += ["--expected-" + name + "-sha256", "0" * 64]
    args += ["--expected-code-revision", "wrong", "--output", str(output)]
    assert main(args) == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert receipt["gpu_requested"] is False
    assert "code revision mismatch" in receipt["error"]
    with pytest.raises(FileExistsError):
        main(args)
