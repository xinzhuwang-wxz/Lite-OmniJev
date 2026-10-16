"""Public proposal and metadata-binding failure checks."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "third_admission", HERE / "decision-index-third-admission.py"
)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = json.loads((HERE / "decision-index-third-admission.observed.json").read_bytes())
    path = tmp_path / "base.json"
    path.write_text(json.dumps({"files": rows}))
    monkeypatch.setattr(mod, "BASE", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_real_metadata_proposal_does_not_admit(base: Path) -> None:
    result = mod.build(
        base,
        HERE / "decision-index-third-admission.observed.json",
        HERE / "decision-index-third-admission.metadata.json",
    )
    assert result["catalog_ids"] == [1, 3, 58]
    assert result["admissions"] == [] and not result["applied"]
    assert len(result["decisions"][0]["matching_publisher_blobs"]) == 16
    assert len(result["decisions"][0]["different_publisher_blobs"]) == 25


def test_changed_input_hash_is_rejected(tmp_path: Path, base: Path) -> None:
    rows = json.loads((HERE / "decision-index-third-admission.observed.json").read_bytes())
    rows[0]["sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="input SHA"):
        mod.build(
            base,
            changed,
            HERE / "decision-index-third-admission.metadata.json",
        )


def test_changed_base_is_rejected(tmp_path: Path) -> None:
    changed = tmp_path / "base.json"
    changed.write_text("{}")
    with pytest.raises(ValueError, match="approved base"):
        mod.build(
            changed,
            HERE / "decision-index-third-admission.observed.json",
            HERE / "decision-index-third-admission.metadata.json",
        )
