"""Public proposal and identity rejection checks, without reading dataset rows."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "admission_next", HERE / "decision-index-admission-next.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
META = HERE / "decision-index-admission-next.metadata.json"
OBS = HERE / "decision-index-admission-next.observed.json"


@pytest.fixture
def base(tmp_path, monkeypatch):
    rows = json.loads(OBS.read_bytes())
    p = tmp_path / "base.json"
    p.write_text(json.dumps({"files": rows, "admissions": [{}] * 16}))
    monkeypatch.setattr(module, "BASE", hashlib.sha256(p.read_bytes()).hexdigest())
    return p


def test_only_requested_evaluation_proposal(base):
    result = module.build(base, OBS, META)
    assert result["catalog_ids"] == [3, 32, 37]
    assert not result["applied"] and not result["source_rows_read"]
    assert {a["catalog_id"]: a["license"] for a in result["admissions"]} == {
        3: "Apache-2.0",
        32: "CC-BY-4.0",
        37: "Apache-2.0",
    }


def test_changed_raw_binding_rejected(base, tmp_path):
    rows = json.loads(OBS.read_bytes())
    rows[0]["sha256"] = "0" * 64
    p = tmp_path / "observed.json"
    p.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="input_SHA"):
        module.build(base, p, META)


def test_modified_publisher_evidence_rejected(base, tmp_path):
    p = tmp_path / "metadata.json"
    p.write_bytes(META.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata_identity"):
        module.build(base, OBS, p)
