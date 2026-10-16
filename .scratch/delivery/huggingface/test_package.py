import importlib.util
import json
import struct
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "package_check", Path(__file__).with_name("check_package.py")
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_templates_remain_incomplete():
    for name in ["e4b-adapter", "e2b-edge"]:
        r = M.check(Path(__file__).parent / name)
        assert r["status"] == "inventory_incomplete" and r["errors"] == []
        assert not r["release_ready"] and "adapter_model.safetensors" in r["missing"]


def test_reject_unknown_and_symlink(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"base")
    (tmp_path / "README.md").symlink_to(tmp_path / "model.safetensors")
    r = M.check(tmp_path)
    assert len(r["errors"]) == 2 and not r["files_sha256"]


def test_reject_credentials_without_echo(tmp_path):
    secret = "hf_" + "x" * 24
    (tmp_path / "README.md").write_text(secret)
    r = M.check(tmp_path)
    assert r["errors"] == ["credential_pattern:README.md"] and secret not in str(r)


def test_reject_base_tensor_in_adapter_filename(tmp_path):
    header = json.dumps({"model.embed_tokens.weight": {"shape": [16, 16], "dtype": "F32"}}).encode()
    (tmp_path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    assert M.check(tmp_path)["errors"] == ["invalid_adapter_header"]
