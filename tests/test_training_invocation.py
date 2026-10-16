"""Real CPU AdamW/dropout through the training loop; no Gemma/CUDA quality claims."""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from train import trainer
from train.checkpoint import load_checkpoint
from train.monitoring import SwanLabConfig

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


@pytest.fixture
def cpu(monkeypatch, tmp_path):
    draws = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)
            self.dropout = torch.nn.Dropout(0.25)

        def to(self, device):
            return self

        def forward(self, input_ids, **kwargs):
            draws.append(int(input_ids[0, 0]))
            x = input_ids.float()
            features = torch.stack([x / 8, x.square() / 64 + random.random() / 10], dim=-1)
            return SimpleNamespace(logits=self.linear(self.dropout(features)))

        def save_pretrained(self, directory, **kwargs):
            from safetensors.torch import save_file

            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            save_file(self.state_dict(), str(path / "adapter_model.safetensors"))

    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "CPU test double")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: Model())
        ),
    )
    monkeypatch.setattr(trainer, "attach_lora", lambda base, *a: base)
    monkeypatch.setattr(trainer, "verified_assets", lambda *a: {"weights": "immutable"})
    monkeypatch.setattr(trainer, "runtime_requirements", lambda: {"torch": torch.__version__})
    monkeypatch.setattr(trainer, "OfficialTokenizerAdapter", lambda *a: object())
    monkeypatch.setattr(
        trainer,
        "prepare_examples",
        lambda samples, *a: [
            {"input_ids": [i + 1], "token_ids": [0, 1, 2], "target": [1.0, 0.0, 0.0]}
            for i in range(len(samples))
        ],
    )
    config = trainer.LoopConfig(sample_count=8, steps=4, accumulation=2, checkpoint_every=3)

    def run(name, **kwargs):
        return trainer.run(
            config,
            tmp_path / ("a" * 40),
            tmp_path / "manifest.json",
            tmp_path / name,
            execute=True,
            monitoring=SwanLabConfig(
                mode="disabled", failure_policy="warn", logdir=tmp_path / "logs"
            ),
            **kwargs,
        )

    return SimpleNamespace(run=run, draws=draws, config=config, root=tmp_path)


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_pause_forces_nonperiodic_checkpoint_and_exact_adamw_rng_resume(cpu):
    full = cpu.run("full")
    assert full["status"] == "completed" and full["steps_completed"] == 4
    full_draws = list(cpu.draws)
    full_next_rng = (torch.rand(3), random.random())
    cpu.draws.clear()
    paused = cpu.run("split", invocation_steps=2)
    assert paused["status"] == "paused" and paused["exit_code"] == 0
    assert paused["steps_completed"] == 2
    assert Path(paused["checkpoint"]).name == "step-0002"
    assert Path(paused["adapter_dir"]).name == "adapter-invocation-step-0002"
    assert Path(paused["adapter_dir"], "adapter_model.safetensors").is_file()
    assert paused["invocation"]["next_draw"] == 4
    assert "invocation_steps" not in paused["config"]
    first_checkpoint_sha = trainer.file_sha256(Path(paused["checkpoint"]) / "checkpoint.json")
    resumed = cpu.run("split", invocation_steps=2, resume_from=Path(paused["checkpoint"]))
    assert resumed["status"] == "completed" and resumed["exit_code"] == 0
    assert resumed["invocation"]["start_step"] == 2
    assert resumed["invocation"]["steps_completed"] == 2
    assert resumed["invocation"]["next_draw"] == 8
    assert full["config"] == paused["config"] == resumed["config"]
    assert cpu.draws == full_draws and len(set(cpu.draws)) == 8
    assert resumed["losses"] == full["losses"]
    assert (
        trainer.file_sha256(Path(paused["checkpoint"]) / "checkpoint.json") == first_checkpoint_sha
    )
    a, b = load_checkpoint(Path(full["checkpoint"])), load_checkpoint(Path(resumed["checkpoint"]))
    for key in [
        "model_tensors",
        "optimizer_state",
        "torch_cpu_rng",
        "python_random_state",
        "order",
    ]:
        assert_state_equal(a[key], b[key])
    assert torch.equal(torch.rand(3), full_next_rng[0]) and random.random() == full_next_rng[1]


def test_periodic_and_invocation_boundary_do_not_double_save(cpu):
    report = cpu.run("periodic", invocation_steps=3)
    assert report["status"] == "paused" and report["steps_completed"] == 3
    assert [p.name for p in (cpu.root / "periodic/checkpoints").iterdir()] == ["step-0003"]


def test_rejected_old_checkpoint_and_config_preserve_previous_report(cpu):
    first = cpu.run("stale", invocation_steps=1)
    second = cpu.run("stale", invocation_steps=1, resume_from=Path(first["checkpoint"]))
    path = cpu.root / "stale/run.json"
    receipt = path.read_bytes()
    stale = cpu.run("stale", invocation_steps=1, resume_from=Path(first["checkpoint"]))
    assert stale["status"] == "blocked" and "older than" in stale["error"]
    assert path.read_bytes() == receipt
    cpu.config.learning_rate *= 2
    mismatch = cpu.run("stale", invocation_steps=1, resume_from=Path(second["checkpoint"]))
    assert mismatch["status"] == "blocked" and "config does not match" in mismatch["error"]
    assert path.read_bytes() == receipt


def test_data_identity_rejection_preserves_paused_report(cpu):
    paused = cpu.run("data", invocation_steps=1)
    path = cpu.root / "data/run.json"
    receipt = path.read_bytes()
    (cpu.root / "data/data.jsonl").write_text("changed")
    report = cpu.run("data", invocation_steps=1, resume_from=Path(paused["checkpoint"]))
    assert report["status"] == "blocked" and "data.jsonl" in report["error"]
    assert path.read_bytes() == receipt


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
def test_invalid_invocation_limit_rejected_before_creating_output(cpu, value):
    report = cpu.run("invalid", invocation_steps=value)
    assert report["status"] == "blocked" and report["exit_code"] == 1
    assert "positive integer" in report["error"]
    assert not (cpu.root / "invalid").exists()


def test_failure_after_update_keeps_failed_receipt_and_committed_checkpoint(cpu, monkeypatch):
    original = trainer.save_checkpoint

    def fail_after_first(*args, **kwargs):
        if kwargs["step"] == 2:
            raise OSError("synthetic checkpoint write failure")
        return original(*args, **kwargs)

    cpu.config.checkpoint_every = 1
    monkeypatch.setattr(trainer, "save_checkpoint", fail_after_first)
    report = cpu.run("failed", invocation_steps=2)
    assert report["status"] == "failed" and report["exit_code"] == 1
    assert report["steps_completed"] == 2 and Path(report["checkpoint"]).name == "step-0001"
    assert json.loads((cpu.root / "failed/run.json").read_text())["status"] == "failed"


def test_cli_forwards_limit_without_adding_it_to_global_config(monkeypatch, tmp_path):
    from scripts import train_s1

    config = trainer.LoopConfig(sample_count=8, steps=4, accumulation=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    captured = {}

    def fake_run(received, *args, **kwargs):
        captured.update(kwargs)
        assert received.model_dump_json() == config.model_dump_json()
        return {"status": "paused", "exit_code": 0, "steps_completed": 2}

    monkeypatch.setattr(train_s1, "run", fake_run)
    code = train_s1.main(
        [
            "run",
            "--config",
            str(config_path),
            "--model-dir",
            str(tmp_path / "model"),
            "--asset-manifest",
            str(tmp_path / "assets.json"),
            "--data",
            str(tmp_path / "data.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--report",
            str(tmp_path / "report.json"),
            "--invocation-steps",
            "2",
        ]
    )
    assert code == 0 and captured["invocation_steps"] == 2
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "paused"
