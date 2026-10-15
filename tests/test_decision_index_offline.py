"""Offline preparation behavior; upstream recipes are never executed."""

import ast
import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts import decision_index_offline as offline


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    # Minimal version-pinned source fixture, not an upstream recipe execution.
    scored = (
        "import json\nimport pickle\nALLOWED = set()\n"
        "class TableUnpickler(pickle.Unpickler):\n    pass\n\n"
        "def digest(text):\n    return text\n\n"
        "def routerbench(layout):\n    for shot in (0, 5):\n"
        '        path = layout.raw / f"routerbench/routerbench_{shot}shot.pkl"\n'
        + offline.OLD_READ
        + '        for ix, r in enumerate(table.to_dict("records")):\n            pass\n'
    ).encode()
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(f"decision-index-{offline.REVISION}/{offline.SCORED_PATH}")
        info.size = len(scored)
        bundle.addfile(info, io.BytesIO(scored))
    monkeypatch.setattr(offline, "ARCHIVE_SHA256", offline.digest(archive))
    import hashlib

    monkeypatch.setattr(offline, "SCORED_SHA256", hashlib.sha256(scored).hexdigest())
    raw = tmp_path / "raw.json"
    raw.write_text('{"not_a_benchmark_row":true}')
    config = {
        "source_archive": {"path": str(archive), "sha256": offline.digest(archive)},
        "files": [
            {
                "path": str(raw),
                "sha256": offline.digest(raw),
                "destination": offline.RAW + "clinc150/data_full.json",
                "catalog_ids": [5],
            }
        ],
        "admissions": [
            {
                "catalog_id": 5,
                "evaluation_admitted": True,
                "license": "explicit test license",
                "evidence": "test decision",
            }
        ],
    }
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps(config))
    return manifest, config, tmp_path / "output"


def test_preparation_maps_copies_binds_sources_and_never_claims_ready(preparation):
    manifest, config, output = preparation
    before = offline.digest(Path(config["source_archive"]["path"]))
    report = offline.prepare(manifest, output)
    target = output / "work" / config["files"][0]["destination"]
    assert target.read_bytes() == Path(config["files"][0]["path"]).read_bytes()
    for path in (output, *output.rglob("*")):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
    assert report["status"] == "prepared_not_rebuilt"
    assert report["ready"] is False
    assert report["pending"]["recipe_file_closure_verified"] is False
    assert report["pending"]["full_rebuild_and_official_verify"] is False
    assert 5 not in report["pending"]["evaluation_admissions"]
    assert 6 in report["pending"]["evaluation_admissions"]
    assert report["pending"]["tiktoken_cache"] is True
    assert "--skip-download" in report["next_call"]["rebuild_argv"]
    assert report["next_call"]["cwd"] == str(output / "source")
    assert offline.digest(Path(config["source_archive"]["path"])) == before
    assert json.loads((output / "preparation.json").read_text()) == report
    # Work is an independent copy, never an alias back to the caller's input.
    target.write_bytes(b"modified copy")
    assert offline.digest(Path(config["files"][0]["path"])) == config["files"][0]["sha256"]
    with pytest.raises(FileExistsError):
        offline.prepare(manifest, output)


@pytest.mark.parametrize("case", ["sha", "missing", "traversal", "absolute", "symlink"])
def test_invalid_inputs_reject_without_creating_output(preparation, case):
    manifest, config, output = preparation
    item = config["files"][0]
    if case == "sha":
        item["sha256"] = "0" * 64
    elif case == "missing":
        item["path"] += "-missing"
    elif case == "traversal":
        item["destination"] = offline.RAW + "../../escape"
    elif case == "absolute":
        item["destination"] = "/tmp/escape"
    else:
        link = manifest.parent / "link"
        link.symlink_to(item["path"])
        item["path"] = str(link)
    manifest.write_text(json.dumps(config))
    with pytest.raises((ValueError, OSError)):
        offline.prepare(manifest, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "C:/bad", "a\\bad"])
def test_zip_path_traversal_is_rejected_before_writes(tmp_path, name):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("normal.txt", "normal")
        bundle.writestr(name, "not allowed")
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        offline.safe_zip(archive, out)
    assert not out.exists()


def test_zip_link_rejection_and_selected_members(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "../../target")
    with pytest.raises(ValueError, match="links"):
        offline.safe_zip(archive, tmp_path / "out")
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("release/themeData.mat", "selected")
        bundle.writestr("other.txt", "ignored")
    out = tmp_path / "selected"
    assert offline.safe_zip(archive, out, {"release/themeData.mat"}) == ["release/themeData.mat"]
    assert (out / "release/themeData.mat").read_text() == "selected"
    assert not (out / "other.txt").exists()
    with pytest.raises(ValueError, match="missing"):
        offline.safe_zip(archive, tmp_path / "missing", {"absent.mat"})


def test_source_tar_links_are_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo(f"decision-index-{offline.REVISION}/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/outside"
        bundle.addfile(info)
    with pytest.raises(ValueError, match="links"):
        offline.safe_tar_source(archive, tmp_path / "out")


def test_derived_json_reader_preserves_order_columns_and_values(preparation, tmp_path):
    manifest, _, output = preparation
    offline.prepare(manifest, output)
    text = (output / "source" / offline.SCORED_PATH).read_text()
    assert "pickle" not in text
    assert 'enumerate(table.to_dict("records"))' not in text
    assert "enumerate(rows)" in text
    # Compile only the generated local JSON reader, never upstream functions/imports.
    reader = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "offline_router_records"
    )
    namespace = {"json": json}
    exec(
        compile(ast.Module(body=[reader], type_ignores=[]), "local-json-reader", "exec"), namespace
    )
    records = [
        {"sample_id": "second", "quality": 0.0, "prompt": ["a", "b"]},
        {"sample_id": "first", "quality": 1.0, "prompt": "unicode题"},
    ]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert namespace["offline_router_records"](path) == records
    path.write_text('{"quality": NaN}\n')
    import math

    assert math.isnan(namespace["offline_router_records"](path)[0]["quality"])
    for bad in ['{"a": 1, "a": 2}\n', '{}\n{"b": 1}\n', "[]\n"]:
        path.write_text(bad)
        with pytest.raises(ValueError):
            namespace["offline_router_records"](path)


def test_real_locked_archive_can_be_prepared_without_execution(tmp_path):
    archive = Path(
        "/Users/bamboo/Githubs/Lite-OmniJev/artifacts/source-audit/public-index/decision-index.tar.gz"
    )
    if not archive.is_file():
        pytest.skip("fixed public source archive not present; fixture cases remain portable")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_archive": {
                    "path": str(archive),
                    "sha256": offline.ARCHIVE_SHA256,
                },
                "files": [],
            }
        )
    )
    before = offline.digest(archive)
    report = offline.prepare(manifest, tmp_path / "prepared")
    assert report["ready"] is False
    assert len(report["pending"]["catalog_materials_not_declared"]) == 44
    assert report["routerbench_patch"]["original_sha256"] == offline.SCORED_SHA256
    assert offline.digest(archive) == before
    tokens = [token for item in report["recipe_static_scan"] for token in item["tokens"]]
    assert "pickle" not in tokens
    assert "get_encoding" in tokens
    assert "subprocess" in tokens


def test_router_materials_must_match_audited_conversion_receipt(preparation):
    manifest, config, output = preparation
    config["files"][0]["destination"] = offline.RAW + "routerbench/routerbench_0shot.jsonl"
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="audited conversion"):
        offline.prepare(manifest, output)
    assert not output.exists()
