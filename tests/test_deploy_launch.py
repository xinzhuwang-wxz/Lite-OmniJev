"""Exercise launch configuration without importing or starting a GPU engine."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "serve/deploy/launch-vllm.sh"


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_published_and_official_metadata_produce_fp32_config(tmp_path: Path, dtype: str) -> None:
    base, model, env = [tmp_path / name for name in ("base", "model", "env")]
    for path in (base, model, env / "bin"):
        path.mkdir(parents=True)
    config = {
        "dtype": dtype,
        **{key: {"dtype": dtype} for key in ("audio_config", "text_config", "vision_config")},
    }
    for path in (base, model):
        (path / "config.json").write_text(json.dumps(config))
    (model / "generation_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture-not-a-real-model")
    (env / "bin/python").symlink_to(sys.executable)
    (env / "bin/vllm").write_text('import sys\nassert sys.argv[1] == "serve"\n')
    import os

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        env={**os.environ, "SERVE_ENV": str(env), "MODEL_DIR": str(model), "BASE_DIR": str(base)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    produced = json.loads((tmp_path / "config-fp32-dir/config.json").read_text())
    assert produced["dtype"] == "float32"
    assert all(
        produced[key]["dtype"] == "float32"
        for key in ("audio_config", "text_config", "vision_config")
    )
    assert json.loads((base / "config.json").read_text()) == config


def test_operator_directory_without_binding_is_preserved(tmp_path: Path) -> None:
    from serve.deploy.prepare_view import prepare

    base, model, view, config_dir = [
        tmp_path / name for name in ("base", "model", "view", "config")
    ]
    for path in (base, model, view):
        path.mkdir()
    cfg = {
        "dtype": "float32",
        **{key: {"dtype": "float32"} for key in ("audio_config", "text_config", "vision_config")},
    }
    for path in (base, model):
        (path / "config.json").write_text(json.dumps(cfg))
    (model / "model.safetensors").write_bytes(b"fixture")
    original = view / "operator-data.txt"
    original.write_text("preserve")
    with pytest.raises(ValueError, match="unbound directory"):
        prepare(model, base, view, config_dir)
    assert original.read_text() == "preserve"
    assert not config_dir.exists()


def test_changed_source_refuses_stale_view_and_keeps_previous_release(tmp_path: Path) -> None:
    from serve.deploy.prepare_view import prepare

    base, first, second, view, config_dir = [
        tmp_path / name for name in ("base", "first", "second", "view", "config")
    ]
    cfg = {
        "dtype": "float32",
        **{key: {"dtype": "float32"} for key in ("audio_config", "text_config", "vision_config")},
    }
    for path in (base, first, second):
        path.mkdir()
        (path / "config.json").write_text(json.dumps(cfg))
    (first / "model.safetensors").write_bytes(b"first release")
    (second / "model.safetensors").write_bytes(b"next release")
    prepare(first, base, view, config_dir)
    prepare(first, base, view, config_dir)  # Identical, intact source can be reused.
    with pytest.raises(ValueError, match="source identity differs"):
        prepare(second, base, view, config_dir)
    assert (view / "model.safetensors").read_bytes() == b"first release"
    (view / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="derived metadata changed"):
        prepare(first, base, view, config_dir)
