"""Build an explicit revision-map patch bundle for fixed offline Decision Index inputs.

Validates real original Git roots/HEADs and hashes; never creates Git metadata,
executes upstream, loads pickle or modifies the prepared source/work trees.
The bundle's layout.py is explicitly derived, not the original upstream recipe.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.decision_index_offline import (
    ARCHIVE_SHA256,
    RAW,
    REVISION,
    PreparationManifest,
    checked_file,
    digest,
    relative_path,
)

LAYOUT_SHA256 = "21d36d35cf8c5369cadbd2f509e1dc508b6dae3ec3d408447643895ffade38fb"
ACQUIRE_SHA256 = "94bc9997f9c95ae33ed9d2d00e6c715bd1408b69ecb64de9e31e06317b761c16"
REPOSITORIES = (
    "searchless_chess",
    "habermas_machine",
    "pop909cl",
    "isarcasm",
    "sata",
    "contractnli",
    "nli4ct",
    "esci",
)
OLD_FUNCTION = """def git_revision(repo):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
"""


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def plain_path(path: Path, *, exists: bool = True) -> Path:
    require(
        path.is_absolute() and path.absolute() == path.resolve(), "absolute canonical path required"
    )
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), "symlink path forbidden")
    if exists:
        require(path.exists(), "path missing")
    return path


def git_value(root: Path, *args: str) -> str:
    # Remove overrides that could redirect Git away from this original checkout.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_OPTIONAL_LOCKS="0",
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_NO_LAZY_FETCH="1",
        GIT_ALLOW_PROTOCOL="",
    )
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=30,
    )
    return result.stdout.strip()


def verify_origin(root: Path, expected: str) -> dict[str, str]:
    plain_path(root)
    require(
        (root / ".git").is_dir() and not (root / ".git").is_symlink(),
        "original Git metadata missing",
    )
    require(
        git_value(root, "rev-parse", "--show-toplevel") == str(root), "original Git root mismatch"
    )
    head = git_value(root, "rev-parse", "HEAD")
    require(head == expected, "original Git HEAD differs from acquisition pin")
    require(
        git_value(root, "cat-file", "-t", head) == "commit", "original HEAD is not a commit object"
    )
    return {
        "root": str(root),
        "head": head,
        "root_checked": str(root),
        "working_tree_clean": "not_attested; material hashes are bound separately",
    }


def literal_acquisition(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == ACQUIRE_SHA256, "fixed acquisition source SHA mismatch"
    )
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"GIT", "HTTP"}:
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {"GIT", "HTTP"}, "acquisition constants missing")
    return values


def derived_layout(source: Path, mapping_sha: str) -> str:
    raw = source.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == LAYOUT_SHA256,
        "fixed original layout source SHA mismatch",
    )
    require(
        len(mapping_sha) == 64 and all(c in "0123456789abcdef" for c in mapping_sha),
        "invalid mapping SHA",
    )
    text = raw.decode()
    require(text.count(OLD_FUNCTION) == 1, "fixed git_revision seam mismatch")
    function = f'''def git_revision(repo):
    # Derived artifact provenance; no Git fallback and no claim of a local checkout.
    import hashlib
    import json
    import os
    from pathlib import Path
    name = os.environ.get("DECISION_INDEX_PROVENANCE_MAP")
    if not name:
        raise ValueError("explicit artifact provenance mapping required")
    mapping_path = Path(name)
    if not mapping_path.is_absolute() or mapping_path.is_symlink():
        raise ValueError("invalid provenance mapping path")
    raw = mapping_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "{mapping_sha}":
        raise ValueError("artifact provenance mapping SHA mismatch")
    mapping = json.loads(raw)
    path = Path(repo).absolute()
    if path != path.resolve():
        raise ValueError("provenance repo path is not canonical")
    entry = mapping["repositories"].get(str(path))
    if entry is None:
        raise ValueError("unregistered artifact provenance repo")
    for item in entry["materials"]:
        material = Path(item["materialized_path"])
        if not material.is_absolute() or material != material.resolve() or material.is_symlink():
            raise ValueError("invalid artifact material path")
        with material.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError("artifact material SHA mismatch")
    return entry["revision"]
'''
    patched = text.replace(OLD_FUNCTION, function, 1)
    ast.parse(patched)
    return patched


def bundle(prepared: Path, original: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    for path in (prepared, original, manifest_path):
        plain_path(path)
    plain_path(output, exists=False)
    require(not output.exists(), "bundle output already exists")
    receipt_path = prepared / "preparation.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    require(
        receipt["status"] == "prepared_not_rebuilt"
        and receipt["source_archive_sha256"] == ARCHIVE_SHA256,
        "preparation receipt is not the pinned source",
    )
    manifest_bytes = manifest_path.read_bytes()
    require(
        hashlib.sha256(manifest_bytes).hexdigest() == receipt["input_manifest_sha256"],
        "input manifest differs from preparation receipt",
    )
    manifest = PreparationManifest.model_validate_json(manifest_bytes)
    require(manifest.source_archive.sha256 == ARCHIVE_SHA256, "source archive pin mismatch")
    checked_file(manifest.source_archive)
    source = prepared / "source/decision_index/suite/build"
    acquisition = literal_acquisition(source / "acquire.py")
    require(digest(source / "layout.py") == LAYOUT_SHA256, "original layout source was modified")
    evidence = {item["destination"]: item for item in receipt["inputs"]}
    require(len(evidence) == len(receipt["inputs"]), "duplicate preparation destinations")
    destinations = set()
    materials: dict[str, list[dict[str, str]]] = {name: [] for name in REPOSITORIES}
    for item in manifest.files:
        relative = relative_path(item.destination)
        require(item.destination not in destinations, "duplicate manifest destination")
        destinations.add(item.destination)
        name = next(
            (
                name
                for name in REPOSITORIES
                if item.destination.startswith(RAW + "repos/" + name + "/")
            ),
            None,
        )
        if item.destination == RAW + "downloads/chessbench-test-action-value.bag":
            name = "searchless_chess"
        if name is None:
            continue
        require(
            item.destination in evidence
            and evidence[item.destination]["sha256"] == item.sha256
            and evidence[item.destination]["path"] == str(item.path),
            "material preparation evidence mismatch",
        )
        plain_path(item.path)
        require(item.path.is_relative_to(original), "material source outside declared quarantine")
        checked_file(item)
        materialized = plain_path(prepared / "work" / relative)
        require(
            materialized.is_file() and digest(materialized) == item.sha256,
            "materialized input SHA mismatch",
        )
        if name == "searchless_chess":
            source_kind = "pinned_http_material"
            http_key = "raw/downloads/chessbench-test-action-value.bag"
        else:
            suffix = item.destination[len(RAW + "repos/" + name + "/") :]
            git_source = original / "repos" / name / suffix
            source_kind = (
                "verified_git_root_artifact" if item.path == git_source else "pinned_http_material"
            )
            http_key = "raw/repos/" + name + "/" + suffix
        if source_kind == "pinned_http_material":
            require(
                http_key in acquisition["HTTP"] and acquisition["HTTP"][http_key][1] == item.sha256,
                "non-Git material lacks pinned HTTP acquisition",
            )
        materials[name].append(
            {
                "source_path": str(item.path),
                "materialized_path": str(materialized),
                "sha256": item.sha256,
                "source_kind": source_kind,
            }
        )
    repositories = {}
    for name in REPOSITORIES:
        require(bool(materials[name]), "required repo material binding missing: " + name)
        expected = acquisition["GIT"][name][1]
        origin = verify_origin(original / "repos" / name, expected)
        path = plain_path(prepared / "work" / RAW / "repos" / name, exists=False)
        require(not (path / ".git").exists(), "materialized Git metadata must not be used")
        repositories[str(path)] = {
            "revision": expected,
            "original_git": origin,
            "materials": sorted(materials[name], key=lambda item: item["materialized_path"]),
            "provenance_kind": "verified_source_commit_for_hash_bound_artifacts",
        }
    mapping = {
        "schema": "decision-index-artifact-provenance-v1",
        "upstream_revision": REVISION,
        "upstream_archive_sha256": ARCHIVE_SHA256,
        "upstream_source_kind": "archive_not_checkout",
        "preparation_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "repositories": repositories,
    }
    raw = (json.dumps(mapping, sort_keys=True, indent=2) + "\n").encode()
    mapping_sha = hashlib.sha256(raw).hexdigest()
    patched = derived_layout(source / "layout.py", mapping_sha)
    # No source or work mutation; only a new reviewable patch bundle.
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, data in [("revision-map.json", raw), ("layout.py", patched.encode())]:
        with (output / name).open("xb") as sink:
            sink.write(data)
        (output / name).chmod(0o600)
    result = {
        "status": "patch_bundle_prepared",
        "exit_code": 0,
        "cost": None,
        "python": platform.python_version(),
        "duration_seconds": time.monotonic() - started,
        "script_sha256": digest(Path(__file__)),
        "ready": False,
        "upstream_executed": False,
        "original_source_sha256": LAYOUT_SHA256,
        "derived_source_sha256": digest(output / "layout.py"),
        "revision_map_sha256": mapping_sha,
        "repositories": len(repositories),
        "material_files": sum(len(rows) for rows in materials.values()),
        "source_patch_is_original_upstream": False,
        "original_source_path": str(source / "layout.py"),
        "bundle": str(output),
        "lookup_environment": {"DECISION_INDEX_PROVENANCE_MAP": str(output / "revision-map.json")},
        "remaining": [
            "Review/install derived layout patch with hash verification",
            "Exact tiktoken cache and real network isolation",
            "License admissions and complete rebuild/verify",
        ],
        "scope": "Artifact source revision only; no clean-checkout or evaluation admission claim",
    }
    with (output / "receipt.json").open("x") as sink:
        json.dump(result, sink, indent=2)
        sink.write("\n")
    (output / "receipt.json").chmod(0o600)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "original-root", "input-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = bundle(
            args.prepared_root, args.original_root, args.input_manifest, args.output_dir
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "exit_code": 1,
                    "ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
