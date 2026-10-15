"""CPU revision-map boundary tests; fixtures are not real acquisition attestations."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import decision_index_provenance as provenance


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def checkout(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    git(root, "init", "-q")
    (root / "fixture.txt").write_text("test fixture, not an external source")
    git(root, "add", "fixture.txt")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "Test fixture",
    )
    return root, git(root, "rev-parse", "HEAD")


def test_original_requires_its_own_real_git_root_and_exact_head(tmp_path):
    root, head = checkout(tmp_path)
    assert provenance.verify_origin(root, head)["head"] == head
    with pytest.raises(ValueError, match="acquisition pin"):
        provenance.verify_origin(root, "0" * 40)
    nested = root / "nested"
    nested.mkdir()
    assert git(nested, "rev-parse", "HEAD") == head
    with pytest.raises(ValueError, match="metadata missing"):
        provenance.verify_origin(nested, head)


@pytest.fixture
def lookup(tmp_path, monkeypatch):
    source = tmp_path / "layout.py"
    source.write_text("import subprocess\n" + provenance.OLD_FUNCTION)
    monkeypatch.setattr(provenance, "LAYOUT_SHA256", provenance.digest(source))
    repo = tmp_path / "materialized/work/raw/repos/contractnli"
    repo.mkdir(parents=True)
    material = repo / "data.bin"
    material.write_bytes(b"hashed material")
    mapping = {
        "repositories": {
            str(repo): {
                "revision": "a" * 40,
                "materials": [
                    {"materialized_path": str(material), "sha256": provenance.digest(material)}
                ],
            }
        }
    }
    path = tmp_path / "revision-map.json"
    path.write_text(json.dumps(mapping))
    patched = provenance.derived_layout(source, provenance.digest(path))
    function = next(node for node in ast.parse(patched).body if isinstance(node, ast.FunctionDef))
    namespace = {}
    # Execute only our generated local lookup, never upstream module imports or recipe.
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "derived-local-lookup", "exec"),
        namespace,
    )
    monkeypatch.setenv("DECISION_INDEX_PROVENANCE_MAP", str(path))
    return namespace["git_revision"], repo, path, material, mapping, source


def test_lookup_returns_bound_source_revision_without_git_fallback(lookup, monkeypatch):
    function, repo, *_ = lookup
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: pytest.fail("Git fallback executed")
    )
    assert function(repo) == "a" * 40
    with pytest.raises(ValueError, match="unregistered"):
        function(repo.parent / "unregistered")


@pytest.mark.parametrize("change", ["mapping", "material", "missing_environment", "symlink"])
def test_lookup_rejects_tampering(lookup, monkeypatch, change, tmp_path):
    function, repo, path, material, _, _ = lookup
    if change == "mapping":
        path.write_text("{}")
    elif change == "material":
        material.write_bytes(b"mutated")
    elif change == "missing_environment":
        monkeypatch.delenv("DECISION_INDEX_PROVENANCE_MAP")
    else:
        link = tmp_path / "repo-link"
        link.symlink_to(repo, target_is_directory=True)
        repo = link
    with pytest.raises(ValueError):
        function(repo)


def test_layout_patch_rejects_unpinned_or_changed_original(lookup):
    _, _, path, _, _, source = lookup
    before = source.read_bytes()
    patched = provenance.derived_layout(source, provenance.digest(path))
    assert source.read_bytes() == before
    assert "subprocess.check_output" not in patched
    source.write_bytes(before + b"\n# unexpected change\n")
    with pytest.raises(ValueError, match="original layout source SHA"):
        provenance.derived_layout(source, provenance.digest(path))


def test_missing_searchless_directory_needs_no_fabricated_git_metadata(tmp_path, monkeypatch):
    source = tmp_path / "layout.py"
    source.write_text(provenance.OLD_FUNCTION)
    monkeypatch.setattr(provenance, "LAYOUT_SHA256", provenance.digest(source))
    missing_repo = tmp_path / "work/repos/searchless_chess"
    bag = tmp_path / "bag"
    bag.write_bytes(b"bound external bag bytes")
    mapping = tmp_path / "revision-map.json"
    mapping.write_text(
        json.dumps(
            {
                "repositories": {
                    str(missing_repo): {
                        "revision": "b" * 40,
                        "materials": [
                            {"materialized_path": str(bag), "sha256": provenance.digest(bag)}
                        ],
                    }
                }
            }
        )
    )
    patched = provenance.derived_layout(source, provenance.digest(mapping))
    function = next(node for node in ast.parse(patched).body if isinstance(node, ast.FunctionDef))
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "local-lookup", "exec"), namespace)
    monkeypatch.setenv("DECISION_INDEX_PROVENANCE_MAP", str(mapping))
    assert namespace["git_revision"](missing_repo) == "b" * 40
    assert not missing_repo.exists()


def test_bundle_refuses_existing_output_before_touching_sources(tmp_path):
    prepared = tmp_path / "prepared"
    original = tmp_path / "original"
    output = tmp_path / "existing"
    for path in (prepared, original, output):
        path.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    marker = output / "keep"
    marker.write_text("untouched")
    with pytest.raises(ValueError, match="already exists"):
        provenance.bundle(prepared, original, manifest, output)
    assert marker.read_text() == "untouched"


def test_real_locked_layout_patch_seam_when_available(tmp_path):
    source = Path(
        "/Users/bamboo/Githubs/Lite-OmniJev/artifacts/source-audit/public-index/src/decision-index-f8552a13b643d39f61185f5951bb5fed16bedf51/decision_index/suite/build/layout.py"
    )
    if not source.exists():
        pytest.skip("fixed source archive not present")
    before = provenance.digest(source)
    patched = provenance.derived_layout(source, hashlib.sha256(b"mapping").hexdigest())
    assert provenance.digest(source) == before == provenance.LAYOUT_SHA256
    assert hashlib.sha256(patched.encode()).hexdigest() != before


def test_complete_bundle_keeps_source_work_and_real_git_untouched(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    original = tmp_path / "quarantine"
    output = tmp_path / "bundle"
    source = prepared / "source/decision_index/suite/build"
    source.mkdir(parents=True)
    layout = source / "layout.py"
    layout.write_text(provenance.OLD_FUNCTION)
    (source / "acquire.py").write_text("# fixture constants supplied below\n")
    monkeypatch.setattr(provenance, "LAYOUT_SHA256", provenance.digest(layout))
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"fixture archive; not upstream")
    monkeypatch.setattr(provenance, "ARCHIVE_SHA256", provenance.digest(archive))
    acquisition = {"GIT": {}, "HTTP": {}}
    files = []
    before = {}
    for name in provenance.REPOSITORIES:
        repo = original / "repos" / name
        repo.mkdir(parents=True)
        git(repo, "init", "-q")
        data = repo / "fixture.txt"
        data.write_text("fixture " + name)
        git(repo, "add", "fixture.txt")
        git(
            repo,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Fixture",
        )
        head = git(repo, "rev-parse", "HEAD")
        acquisition["GIT"][name] = ("fixture-url", head, None)
        if name == "searchless_chess":
            data = original / "raw/downloads/chessbench-test-action-value.bag"
            data.parent.mkdir(parents=True)
            data.write_bytes(b"external pinned HTTP fixture")
            destination = provenance.RAW + "downloads/chessbench-test-action-value.bag"
            acquisition["HTTP"]["raw/downloads/chessbench-test-action-value.bag"] = (
                "fixture-url",
                provenance.digest(data),
            )
        else:
            destination = provenance.RAW + "repos/" + name + "/fixture.txt"
        materialized = prepared / "work" / destination
        materialized.parent.mkdir(parents=True, exist_ok=True)
        materialized.write_bytes(data.read_bytes())
        item = {
            "path": str(data),
            "sha256": provenance.digest(data),
            "destination": destination,
            "catalog_ids": [],
        }
        files.append(item)
        before[str(data)] = provenance.digest(data)
        before[str(materialized)] = provenance.digest(materialized)
    monkeypatch.setattr(provenance, "literal_acquisition", lambda path: acquisition)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_archive": {"path": str(archive), "sha256": provenance.digest(archive)},
                "files": files,
            }
        )
    )
    (prepared / "preparation.json").write_text(
        json.dumps(
            {
                "status": "prepared_not_rebuilt",
                "source_archive_sha256": provenance.digest(archive),
                "input_manifest_sha256": provenance.digest(manifest),
                "inputs": files,
            }
        )
    )
    original_layout = layout.read_bytes()
    result = provenance.bundle(prepared, original, manifest, output)
    assert result["repositories"] == 8 and result["material_files"] == 8 and not result["ready"]
    assert result["original_source_sha256"] == hashlib.sha256(original_layout).hexdigest()
    assert result["derived_source_sha256"] == provenance.digest(output / "layout.py")
    assert result["revision_map_sha256"] == provenance.digest(output / "revision-map.json")
    assert not result["source_patch_is_original_upstream"]
    assert layout.read_bytes() == original_layout
    assert all(provenance.digest(Path(path)) == expected for path, expected in before.items())
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
    assert not any(
        (prepared / "work" / provenance.RAW / "repos" / name / ".git").exists()
        for name in provenance.REPOSITORIES
    )
