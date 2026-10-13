import hashlib
import json

import pytest

from scripts.stage_models import inventory, stage


def manifest(name="README.md", repo="google/gemma-4-E2B-it"):
    data = b"published source"
    return {
        "schema_version": 1,
        "models": [
            {
                "repo_id": repo,
                "revision": "a" * 40,
                "license": "apache-2.0",
                "files": [
                    {
                        "path": name,
                        "bytes": len(data),
                        "required": True,
                        "lfs_sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("name", ["../README.md", "loader.py", "model.bin", "unexpected.pt"])
def test_unreviewed_or_escaping_files_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        inventory(manifest(name), tmp_path)


def test_prohibited_teacher_denied(tmp_path):
    with pytest.raises(ValueError, match="prohibited"):
        inventory(manifest(repo="anything/JEV-27B"), tmp_path)


def test_reviewed_pickle_checkpoint_quarantined_and_never_loaded(tmp_path):
    entry = inventory(manifest("head.pt", "akhilaaa3/Jev-Omni"), tmp_path)[0]
    assert entry[2].is_relative_to(tmp_path / "quarantine" / "teacher")


def test_verified_file_resumed_without_redownload(tmp_path):
    value = manifest()
    path = inventory(value, tmp_path)[0][2]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"published source")
    calls = []
    result = stage(value, tmp_path, tmp_path / "report.json", lambda **kwargs: calls.append(kwargs))
    assert calls == []
    assert result["exit_code"] == 0
    assert result["weights_loaded"] is False
    assert result["training_admitted"] is False


def test_corrupt_existing_file_preserved_and_signed_error_redacted(tmp_path):
    value = manifest()
    path = inventory(value, tmp_path)[0][2]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt")
    calls = []
    result = stage(value, tmp_path, tmp_path / "report.json", lambda **kwargs: calls.append(kwargs))
    assert result["exit_code"] == 1
    assert path.read_bytes() == b"corrupt"
    assert calls == []
    path.unlink()

    def fail(**kwargs):
        raise RuntimeError("https://private.example/?secret=sentinel")

    result = stage(value, tmp_path, tmp_path / "report.json", fail)
    assert result["exit_code"] == 1
    assert "sentinel" not in json.dumps(result)
