"""Observable training telemetry, without CUDA or raw input uploads."""

from train.trainer import SmokeConfig, log_training_metrics


class Recorder:
    def __init__(self):
        self.records = []

    def log(self, metrics, step=None):
        self.records.append((step, metrics))

    def finish(self, report):
        pass


def test_step_metrics_are_shared_by_report_and_monitoring_with_bounded_history():
    recorder = Recorder()
    report = {}
    for step in range(1, 301):
        report["steps_completed"] = step
        log_training_metrics(
            recorder,
            SmokeConfig(),
            report,
            mean_loss=0.5,
            grad_norm=0.25,
            step_metrics={
                "step_wall_seconds": 2.0,
                "cpu_prepare_wait_seconds": 0.2,
                "invocation_samples_seen": float(step * 10),
                "input_tokens": 123.0,
                "peak_cuda_bytes": 4096.0,
                "image/samples": 2.0,
                "audio/samples": 2.0,
                "crossmodal/samples": 2.0,
                "crossmodal/loss": 0.5,
            },
        )
    assert len(report["step_metrics"]) == 256
    assert report["step_metrics"][0]["step"] == 45
    latest = report["step_metrics"][-1]
    assert latest["step"] == 300
    assert latest["invocation_samples_seen"] == 3000
    step, uploaded = recorder.records[-1]
    assert step == 300
    for key, value in latest.items():
        if key != "step":
            assert uploaded[f"train/{key}"] == value
    assert all(isinstance(value, (int, float)) for value in uploaded.values())
    assert uploaded["train/loss"] == 0.5


def test_training_loop_reports_actual_draws_and_declared_crossmodal_loss(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    import pytest

    from train.trainer import LoopConfig, train_smoke

    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def to(self, device):
            return self

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.weight.expand(1, 1, 3))

        def save_pretrained(self, directory, **kwargs):
            from pathlib import Path

            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.safetensors").write_bytes(b"cpu-test")

    model = Model()
    original_tensor = torch.tensor
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: original_tensor(*args, **{**kwargs, "device": "cpu"}),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU test double")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 8192)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: model)
        ),
    )
    monkeypatch.setattr("train.trainer.attach_lora", lambda *args: model)
    monkeypatch.setattr("train.trainer.decision_loss", lambda logits, *a, **k: logits[0] ** 2)
    monkeypatch.setattr("train.trainer.save_checkpoint", lambda *a, **k: tmp_path / "checkpoint")
    config = LoopConfig(sample_count=2, steps=2, accumulation=2)
    examples = [
        {
            "input_ids": [1, 2],
            "token_ids": [0, 1],
            "target": [1, 0],
            "observed_modalities": ["image", "audio"],
            "id": "private-path/question",
        },
        {
            "input_ids": [3, 4, 5],
            "token_ids": [0, 1],
            "target": [1, 0],
            "observed_modalities": ["text"],
        },
    ]
    report = {
        "examples": 2,
        "max_prompt_tokens": 3,
        "config": {},
        "model_revision": "test",
        "data_sha256": "test",
        "asset_hashes": {},
    }
    recorder = Recorder()
    train_smoke(config, tmp_path, examples, tmp_path, report, recorder)
    assert report["steps_completed"] == 2
    for index, metrics in enumerate(report["step_metrics"], 1):
        assert metrics["invocation_samples_seen"] == index * 2
        assert metrics["step_samples"] == 2
        assert metrics["input_tokens"] == 5
        assert metrics["image/samples"] == metrics["audio/samples"] == 1
        assert metrics["crossmodal/samples"] == 1
        assert metrics["text/samples"] == 1
        assert metrics["image/loss"] == metrics["audio/loss"] == metrics["crossmodal/loss"]
        assert metrics["peak_cuda_bytes"] == 8192
        assert metrics["step_wall_seconds"] >= metrics["cpu_prepare_wait_seconds"] >= 0
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["step_metrics"] == report["step_metrics"]
    assert "private-path/question" not in json.dumps(recorder.records)
    assert "private-path/question" not in json.dumps(saved["step_metrics"])
