"""Public proposal contract tests; metadata only, no benchmark records."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "batch005", HERE / "decision-index-admission-batch005.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def inputs(monkeypatch):
    metadata = json.loads((HERE / "decision-index-admission-batch005-metadata.json").read_bytes())
    observed = [
        {k: v for k, v in f.items() if k != "publisher_blob"}
        for r in metadata["candidates"]
        for f in r["inputs"]
    ]
    manifest = {
        "source_archive": {"path": "/archive.tar.gz", "sha256": m.ARCHIVE},
        "files": [
            {k: f[k] for k in ["path", "sha256", "destination", "catalog_ids"]} for f in observed
        ],
        "admissions": [
            {
                "catalog_id": i,
                "evaluation_admitted": True,
                "license": "prior",
                "evidence": "unchanged",
            }
            for i in sorted(m.PRIOR)
        ],
    }
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    # Synthetic metadata-only manifest tests validators without shipping full remote manifest.
    monkeypatch.setattr(m, "BASE", m.sha(raw))
    return raw, observed, metadata


def test_six_proposals_keep_all_prior_and_not_ready(inputs):
    result = m.build(*inputs)
    assert result["proposed_total_count"] == 27
    assert result["input_files_verified"] == 12
    assert result["input_bytes_verified"] == 57230302
    assert result["ready"] is False and result["applied"] is False
    assert len(result["remaining_catalog_ids"]) == 17
    assert 33 in result["remaining_catalog_ids"]
    assert (
        next(x for x in result["admissions"] if x["catalog_id"] == 28)["license"]
        == "CC-BY (version unspecified by publisher)"
    )


def test_manifest_mutation_rejected(inputs):
    raw, observed, metadata = inputs
    with pytest.raises(ValueError, match="approved_manifest_bytes"):
        m.build(raw + b" ", observed, metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "0" * 64),
        ("path", "/other"),
        ("md5", "0" * 32),
        ("git_blob_sha1", "0" * 40),
        ("bytes", 0),
    ],
)
def test_actual_file_identity_rejected(inputs, field, value):
    raw, observed, metadata = inputs
    observed = copy.deepcopy(observed)
    observed[0][field] = value
    with pytest.raises(ValueError):
        m.build(raw, observed, metadata)


def test_duplicate_observation_rejected(inputs):
    raw, observed, metadata = inputs
    with pytest.raises(ValueError, match="missing_duplicate"):
        m.build(raw, observed + [observed[0]], metadata)


def test_license_snapshot_mutation_rejected(inputs):
    raw, observed, metadata = inputs
    metadata = copy.deepcopy(metadata)
    metadata["candidates"][0]["license"] = "MIT"
    with pytest.raises(ValueError, match="frozen_metadata_identity"):
        m.build(raw, observed, metadata)


def test_prior_scope_mutation_rejected(inputs, monkeypatch):
    raw, observed, metadata = inputs
    obj = json.loads(raw)
    obj["admissions"].pop()
    raw = json.dumps(obj).encode()
    monkeypatch.setattr(m, "BASE", m.sha(raw))
    with pytest.raises(ValueError, match="approved_admissions_identity"):
        m.build(raw, observed, metadata)


def test_existing_output_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "proposal.json"
    output.write_text("original")
    monkeypatch.setattr(
        "sys.argv", ["batch005", "--manifest", str(tmp_path / "absent"), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="output_exists"):
        m.main()
    assert output.read_text() == "original"
