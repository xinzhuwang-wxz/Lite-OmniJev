"""Exact E2B/E4B admission without weakening training asset verification."""

import hashlib
import json

import pytest

from train.trainer import verified_assets


def fixture(tmp_path, repo_id="google/gemma-4-E2B-it", revision="a" * 40, architecture=None):
    model = tmp_path / revision
    model.mkdir()
    contents = {
        "config.json": json.dumps(
            architecture
            or {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}
        ).encode(),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": b"fixture-only-not-real-weights",
    }
    files = []
    for name, payload in contents.items():
        (model / name).write_bytes(payload)
        files.append(
            {
                "path": name,
                "size": len(payload),
                "kind": "weights" if name.endswith(".safetensors") else "metadata",
                "upstream_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "repo_id": repo_id,
                        "revision": revision,
                        "license": "apache-2.0",
                        "files": files,
                    }
                ],
            }
        )
    )
    return model, manifest


@pytest.mark.parametrize("repo", ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"])
def test_exact_supported_backbones_have_all_files_verified(tmp_path, repo):
    model, manifest = fixture(tmp_path, repo)
    hashes = verified_assets(model, manifest)
    assert (
        hashes["model.safetensors"]
        == hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest()
    )
    assert len(hashes) == 4


@pytest.mark.parametrize(
    "repo", ["google/gemma-4-E2B", "other/gemma-4-E2B-it", "google/gemma-4-12B-it"]
)
def test_other_models_not_admitted(tmp_path, repo):
    with pytest.raises(ValueError, match="pinned E4B or E2B"):
        verified_assets(*fixture(tmp_path, repo))


def test_mutable_e2b_revision_rejected(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        verified_assets(*fixture(tmp_path, revision="main"))


@pytest.mark.parametrize("name", ["model.safetensors", "config.json", "tokenizer.json"])
def test_e2b_tampered_files_rejected(tmp_path, name):
    model, manifest = fixture(tmp_path)
    (model / name).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verified_assets(model, manifest)


def test_e2b_other_architecture_rejected(tmp_path):
    with pytest.raises(ValueError, match="architecture"):
        verified_assets(
            *fixture(tmp_path, architecture={"model_type": "other", "architectures": ["Other"]})
        )


def test_e2b_extra_weight_shard_rejected(tmp_path):
    model, manifest = fixture(tmp_path)
    (model / "model-extra.safetensors").write_bytes(b"not-pinned")
    with pytest.raises(ValueError, match="unverified or missing"):
        verified_assets(model, manifest)


def test_e2b_duplicate_revision_rejected(tmp_path):
    model, manifest = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["models"].append(data["models"][0])
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="pinned E4B or E2B"):
        verified_assets(model, manifest)
