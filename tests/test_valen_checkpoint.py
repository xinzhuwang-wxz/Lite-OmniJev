"""Valen checkpoint conversion tests use tiny fixtures, never upstream weights."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import train.valen_checkpoint as valen_checkpoint
from scripts.valen_checkpoint import main
from train.checkpoint import file_sha256
from train.valen_checkpoint import convert_checkpoint, derive_expected_tensors, load_expected_spec


def test_derive_expected_tensors_matches_valen_preview_groups():
    base_config = {
        "text_config": {"hidden_size": 8},
        "vision_config": {"depth": 6},
    }
    valen_config = {
        "stage": "vision_top",
        "method": "rlcd",
        "lora_rank": 32,
        "projection_dim": 256,
    }
    base_shapes = {
        "model.visual.blocks.2.norm.weight": {"shape": [4], "dtype": "torch.bfloat16"},
        "model.visual.blocks.3.norm.weight": {"shape": [4], "dtype": "torch.bfloat16"},
        "model.visual.blocks.4.norm.weight": {"shape": [4], "dtype": "torch.bfloat16"},
        "model.visual.blocks.5.norm.weight": {"shape": [4], "dtype": "torch.bfloat16"},
        "model.visual.merger.linear.weight": {"shape": [8, 4], "dtype": "torch.bfloat16"},
        "model.language_model.layers.0.self_attn.q_proj.weight": {
            "shape": [8, 8],
            "dtype": "torch.bfloat16",
        },
        "model.language_model.layers.0.conv1d.weight": {
            "shape": [8, 8],
            "dtype": "torch.bfloat16",
        },
    }
    metadata = {
        "trainable_parameters": {
            "vision": 16,
            "merger": 32,
            "lora": 512,
            "head": 4096,
        }
    }

    tensors = derive_expected_tensors(
        base_config=base_config,
        valen_config=valen_config,
        metadata=metadata,
        base_shapes=base_shapes,
    )

    assert "backbone.visual.blocks.2.norm.weight" in tensors
    assert "backbone.visual.merger.linear.weight" in tensors
    q_proj_lora_a = (
        "backbone.language_model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    conv_lora_a = "backbone.language_model.base_model.model.layers.0.conv1d.lora_A.default.weight"
    assert tensors[q_proj_lora_a]["shape"] == [32, 8]
    assert conv_lora_a not in tensors
    assert tensors["head.decision.weight"]["shape"] == [256, 8]


def test_derive_expected_tensors_refuses_count_drift():
    with pytest.raises(ValueError, match="lora"):
        derive_expected_tensors(
            base_config={"text_config": {"hidden_size": 8}, "vision_config": {"depth": 4}},
            valen_config={
                "stage": "vision_top",
                "method": "rlcd",
                "lora_rank": 32,
                "projection_dim": 256,
            },
            metadata={
                "trainable_parameters": {
                    "vision": 0,
                    "merger": 0,
                    "lora": 1,
                    "head": 4096,
                }
            },
            base_shapes={},
        )


def test_load_expected_spec_rejects_missing_tensors(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="lacks tensors"):
        load_expected_spec(spec)


def test_convert_cli_requires_isolation(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema_version": 1, "tensors": {"x": {"shape": [1], "dtype": "torch.uint8"}}})
    )
    report = tmp_path / "report.json"

    code = main(
        [
            "convert",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "weights.safetensors"),
            "--expected-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--expected-spec",
            str(spec),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "isolation_required"


def test_isolated_cli_refuses_fifo_output_and_preserves_destination(tmp_path, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("fifo unavailable")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.json"
    metadata = tmp_path / "training_metadata.json"
    config.write_text("{}")
    metadata.write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema_version": 1, "tensors": {"x": {"shape": [1], "dtype": "torch.uint8"}}})
    )
    output = tmp_path / "weights.safetensors"
    output.write_text("previous-good-output")
    report = tmp_path / "report.json"

    monkeypatch.setattr(valen_checkpoint.platform, "system", lambda: "Linux")
    monkeypatch.setattr(valen_checkpoint.os, "geteuid", lambda: 0)

    def fake_run(command, **_kwargs):
        jail = Path(command[command.index("--jail-root") + 1])
        os.mkfifo(jail / "output" / "weights.safetensors")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"schema_version": 1, "status": "converted", "output_sha256": "0" * 64}
            ),
            stderr="",
        )

    monkeypatch.setattr(valen_checkpoint.subprocess, "run", fake_run)
    code = main(
        [
            "convert",
            "--isolate",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--expected-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--expected-spec",
            str(spec),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "isolation_unavailable"
    assert output.read_text() == "previous-good-output"


def test_isolated_cli_preserves_existing_output_on_child_hash_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.json"
    metadata = tmp_path / "training_metadata.json"
    config.write_text("{}")
    metadata.write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema_version": 1, "tensors": {"x": {"shape": [1], "dtype": "torch.uint8"}}})
    )
    output = tmp_path / "weights.safetensors"
    output.write_text("previous-good-output")
    report = tmp_path / "report.json"

    monkeypatch.setattr(valen_checkpoint.platform, "system", lambda: "Linux")
    monkeypatch.setattr(valen_checkpoint.os, "geteuid", lambda: 0)

    def fake_run(command, **_kwargs):
        jail = Path(command[command.index("--jail-root") + 1])
        (jail / "output" / "weights.safetensors").write_bytes(b"not a real safetensors file")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"schema_version": 1, "status": "converted", "output_sha256": "0" * 64}
            ),
            stderr="",
        )

    monkeypatch.setattr(valen_checkpoint.subprocess, "run", fake_run)
    code = main(
        [
            "convert",
            "--isolate",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--expected-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--expected-spec",
            str(spec),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "isolation_unavailable"
    assert output.read_text() == "previous-good-output"


def test_tiny_direct_conversion_round_trips_weights(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "checkpoint.pt"
    weights = {
        "head.decision.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "head.candidate.weight": torch.ones(2, 3, dtype=torch.float32),
    }
    torch.save({"weights": weights, "config": {}, "optimizer": {}}, checkpoint)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in weights.items()
                },
            }
        )
    )
    output = tmp_path / "weights.safetensors"
    payload = convert_checkpoint(
        checkpoint,
        output,
        expected_sha256=file_sha256(checkpoint),
        expected_size=checkpoint.stat().st_size,
        expected_spec=load_expected_spec(spec),
    )

    assert payload["status"] == "converted"
    loaded = safetensors.load_file(str(output), device="cpu")
    assert all(torch.equal(loaded[key], value) for key, value in weights.items())
