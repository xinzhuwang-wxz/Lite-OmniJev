"""Prepare hash-bound Decision Index inputs without executing upstream or loading pickle.

Manifest: source_archive {path, sha256}; files [{path, sha256, destination,
catalog_ids}]; admissions [{catalog_id, evaluation_admitted, license, evidence}].
Destinations are relative to the new work root: artifacts/benchmark-suite/raw/...,
data/sources/..., or cache/tiktoken/.... Explicit admissions are caller-supplied
source decisions, not inferred from downloads. This command never claims rebuild readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REVISION = "f8552a13b643d39f61185f5951bb5fed16bedf51"
ARCHIVE_SHA256 = "12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082"
SCORED_SHA256 = "872061974ab9e3061e9a1e795cf5a6e07f5b0359c96b7451313e36288337b736"
CATALOG_IDS = frozenset(
    {
        1,
        2,
        3,
        4,
        5,
        6,
        9,
        10,
        11,
        12,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        48,
        50,
        56,
        57,
        58,
        59,
        61,
        62,
        64,
    }
)
RAW = "artifacts/benchmark-suite/raw/"
ROUTER_INPUT_HASHES = {
    RAW + "routerbench/routerbench_" + name: sha
    for name, sha in {
        "0shot.pkl": "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
        "5shot.pkl": "fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe",
        "0shot.jsonl": "a2e13db1588e7195ae69529bdc8a207f32cfdbf10191f920092008cd524af344",
        "5shot.jsonl": "4e29bb24d8de803daf1f0535460f9f1916e4f92c032e55acebf87d1b05403c67",
    }.items()
}
SCORED_PATH = "decision_index/suite/build/adapters_scored.py"
OLD_READ = """        with path.open("rb") as f:
            table = TableUnpickler(f).load()
        models = sorted(c for c in table if c + "|total_cost" in table)
"""
NEW_READ = """        rows = offline_router_records(path.with_suffix(".jsonl"))
        columns = rows[0]
        models = sorted(c for c in columns if c + "|total_cost" in columns)
"""
READER = """def offline_router_records(path):
    def unique_pairs(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("duplicate RouterBench JSON key")
            record[key] = value
        return record
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line, object_pairs_hook=unique_pairs) for line in stream]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("RouterBench requires nonempty plain JSON records")
    columns = set(rows[0])
    if any(set(row) != columns for row in rows):
        raise ValueError("RouterBench columns changed between rows")
    return rows


"""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundFile(StrictModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InputFile(BoundFile):
    destination: str
    catalog_ids: list[int] = Field(default_factory=list)


class Admission(StrictModel):
    catalog_id: int
    evaluation_admitted: bool
    license: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class PreparationManifest(StrictModel):
    source_archive: BoundFile
    files: list[InputFile]
    admissions: list[Admission] = Field(default_factory=list)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(item: BoundFile) -> Path:
    if not item.path.is_absolute() or item.path.is_symlink() or not item.path.is_file():
        raise ValueError("input must be an existing absolute regular file, not a symlink")
    if digest(item.path) != item.sha256:
        raise ValueError("input SHA256 mismatch")
    return item.path


def relative_path(value: str) -> Path:
    parts = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or parts.is_absolute()
        or any(part in {"", ".", ".."} for part in value.rstrip("/").split("/"))
        or ":" in parts.parts[0]
    ):
        raise ValueError("unsafe relative path")
    return Path(*parts.parts)


def safe_tar_source(archive: Path, destination: Path) -> None:
    """Pinned source archive only; no links, devices or archive extraction helpers."""
    prefix = f"decision-index-{REVISION}"
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            relative = relative_path(member.name)
            if relative.parts[0] != prefix or member.name.rstrip("/") in seen:
                raise ValueError("unexpected or duplicate source archive member")
            seen.add(member.name.rstrip("/"))
            if not (member.isdir() or member.isfile()):
                raise ValueError("source archive links or special members forbidden")
            target = destination.joinpath(*relative.parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError("missing source archive file")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)


def safe_zip(archive: Path, destination: Path, members: set[str] | None = None) -> list[str]:
    """Validate every member, even those not selected, before writing any file."""
    extracted = []
    with zipfile.ZipFile(archive) as source:
        names: set[str] = set()
        for info in source.infolist():
            relative_path(info.filename)
            name = info.filename.rstrip("/")
            if name in names:
                raise ValueError("duplicate ZIP member")
            names.add(name)
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError("ZIP links or special members forbidden")
            if info.flag_bits & 1:
                raise ValueError("encrypted extraction archive forbidden")
        if members is not None and not members <= names:
            raise ValueError("required ZIP members missing")
        for info in source.infolist():
            if info.is_dir() or (members is not None and info.filename not in members):
                continue
            target = destination / relative_path(info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            extracted.append(info.filename)
    return extracted


def patch_routerbench(source: Path) -> dict[str, str]:
    target = source / SCORED_PATH
    if digest(target) != SCORED_SHA256:
        raise ValueError("RouterBench upstream source hash mismatch")
    original = target.read_text()
    start = original.index("ALLOWED = ")
    end = original.index("def digest(text):", start)
    patched = original[:start] + READER + original[end:]
    if patched.count(OLD_READ) != 1 or patched.count('enumerate(table.to_dict("records"))') != 1:
        raise ValueError("RouterBench source seam mismatch")
    patched = patched.replace("import pickle\n", "", 1).replace(OLD_READ, NEW_READ, 1)
    patched = patched.replace('enumerate(table.to_dict("records"))', "enumerate(rows)", 1)
    if "pickle" in patched or "TableUnpickler" in patched:
        raise ValueError("unexpected remaining pickle reference")
    target.write_text(patched)
    return {"original_sha256": SCORED_SHA256, "derived_sha256": digest(target)}


def recipe_static_scan(source: Path) -> list[dict[str, Any]]:
    """Record textual risk candidates; this is not a proof of transitive safety."""
    pattern = re.compile(
        r"\bpickle\b|joblib|allow_pickle|\beval\(|\bexec\(|subprocess|os\.system|"
        r"import_module|exec_module|np\.load|numpy\.load|torch\.load|urlopen|"
        r"snapshot_download|get_encoding|extractall"
    )
    findings = []
    for path in sorted((source / "decision_index").rglob("*.py")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            matches = pattern.findall(line)
            if matches:
                findings.append(
                    {"path": str(path.relative_to(source)), "line": line_number, "tokens": matches}
                )
    return findings


def prepare(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Produce a distinct work tree and receipt, never an official ready/rebuild result."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists")
    manifest = PreparationManifest.model_validate_json(manifest_path.read_bytes())
    archive = checked_file(manifest.source_archive)
    if manifest.source_archive.sha256 != ARCHIVE_SHA256:
        raise ValueError("source archive must match the fixed upstream revision")
    checked: list[tuple[InputFile, Path, Path]] = []
    destinations: set[str] = set()
    for item in manifest.files:
        relative = relative_path(item.destination)
        if not item.destination.startswith((RAW, "data/sources/", "cache/tiktoken/")):
            raise ValueError("input destination outside the offline Layout")
        if item.destination in destinations or not set(item.catalog_ids) <= CATALOG_IDS:
            raise ValueError("duplicate destination or unknown catalog ID")
        if item.destination in ROUTER_INPUT_HASHES and (
            item.sha256 != ROUTER_INPUT_HASHES[item.destination]
        ):
            raise ValueError("RouterBench input must match the audited conversion/source receipt")
        destinations.add(item.destination)
        checked.append((item, checked_file(item), relative))
    admissions = {item.catalog_id: item for item in manifest.admissions}
    if len(admissions) != len(manifest.admissions) or not admissions.keys() <= CATALOG_IDS:
        raise ValueError("duplicate or unknown admission ID")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="decision-index-prepare-", dir=output.parent) as temp:
        stage = Path(temp) / "prepared"
        source, work = stage / "source", stage / "work"
        source.mkdir(parents=True)
        work.mkdir()
        safe_tar_source(archive, source)
        patch = patch_routerbench(source)
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError("source archive changed during preparation")
        evidence = []
        for item, original, relative in checked:
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as stream, target.open("xb") as sink:
                shutil.copyfileobj(stream, sink)
            if digest(target) != item.sha256:
                raise ValueError("input changed during preparation")
            evidence.append({**item.model_dump(mode="json"), "bytes": target.stat().st_size})
        extractions = []
        cfcolor = work / RAW / "downloads/cfcolor.zip"
        if cfcolor.is_file():
            files = safe_zip(
                cfcolor,
                work / RAW / "repos/cfcolor",
                {
                    "release/allMTurkRatings.mat",
                    "release/themeData.mat",
                },
            )
            extractions.append({"archive": str(cfcolor.relative_to(work)), "members": files})
        for kind in ("model_generations", "evaluation_results"):
            path = work / RAW / f"repos/cruxeval/samples/{kind}.zip"
            if path.is_file():
                files = safe_zip(path, path.parent)
                extractions.append({"archive": str(path.relative_to(work)), "members": files})
        covered = {n for item in manifest.files for n in item.catalog_ids}
        missing_admissions = sorted(
            n for n in CATALOG_IDS if (n not in admissions or not admissions[n].evaluation_admitted)
        )
        missing_router = sorted(
            {
                RAW + f"routerbench/routerbench_{shot}shot.{suffix}"
                for shot in (0, 5)
                for suffix in ("pkl", "jsonl")
            }
            - destinations
        )
        report: dict[str, Any] = {
            "status": "prepared_not_rebuilt",
            "ready": False,
            "exit_code": 0,
            "revision": REVISION,
            "expected_public0_3_counts": {
                "selected_rows": 124971,
                "added_rows": 30419,
                "gsm8k_rows": 2638,
                "scoreable_base_requests_after_exclusions": 109759,
            },
            "source_archive_sha256": ARCHIVE_SHA256,
            "archive_metadata_erratum": (
                "Existing decision-index-assets.json omits the leading 1 of the archive SHA256; "
                "this receipt uses the measured 64-character hash."
            ),
            "input_manifest_sha256": digest(manifest_path),
            "script_sha256": digest(Path(__file__)),
            "output": str(output),
            "inputs": evidence,
            "routerbench_patch": patch,
            "recipe_static_scan": recipe_static_scan(source),
            "residual_execution_risks": [
                "skip_download is a control flag, not enforced network isolation.",
                "tiktoken.get_encoding can fetch cache data if the exact cache is absent.",
                "Recipe git_revision still invokes git; .git provenance closure is required.",
                "Transitive dependencies, package imports and runtime behavior remain unverified.",
            ],
            "extractions": extractions,
            "admissions": [item.model_dump() for item in manifest.admissions],
            "pending": {
                "catalog_materials_not_declared": sorted(CATALOG_IDS - covered),
                "evaluation_admissions": missing_admissions,
                "routerbench_inputs": missing_router,
                "tiktoken_cache": not any(p.startswith("cache/tiktoken/") for p in destinations),
                "recipe_file_closure_verified": False,
                "full_rebuild_and_official_verify": False,
            },
            "boundaries": [
                "No upstream code executed; no pickle loaded; no benchmark rows emitted in report.",
                "Catalog declarations and cache presence are not complete recipe/cache validation.",
                "Only explicit caller evaluation admissions are recorded; no training admission.",
                (
                    "Rebuild exit zero alone is insufficient: "
                    "require official public0.3 counts/hashes and verify."
                ),
                "Next commands pending until recipe closure, admissions and dependencies pass.",
            ],
            "next_call": {
                "cwd": str(output / "source"),
                "environment": {
                    "PYTHONPATH": str(output / "source"),
                    "TIKTOKEN_CACHE_DIR": str(output / "work/cache/tiktoken"),
                    "HF_HUB_OFFLINE": "1",
                },
                "rebuild_argv": [
                    "python",
                    "-m",
                    "decision_index",
                    "suite",
                    "rebuild",
                    "--edition",
                    "0.3",
                    "--work",
                    str(output / "work"),
                    "--skip-download",
                ],
                "import_argv": [
                    "python",
                    "-m",
                    "decision_index",
                    "suite",
                    "import",
                    "--edition",
                    "0.3",
                    "--dir",
                    str(output / "verified-suite"),
                    "--rows",
                    str(
                        output
                        / "work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz"
                    ),
                    "--added-rows",
                    str(
                        output
                        / "work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz"
                    ),
                    "--gsm8k-rows",
                    str(
                        output
                        / "work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz"
                    ),
                ],
                "verify_argv": [
                    "python",
                    "-m",
                    "decision_index",
                    "suite",
                    "verify",
                    "--edition",
                    "0.3",
                    "--dir",
                    str(output / "verified-suite"),
                ],
            },
        }
        (stage / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.manifest, args.output)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "ready": False,
                    "exit_code": 2,
                    "reason": "offline input or preparation validation failed",
                }
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "ready": False,
                "exit_code": 0,
                "report": str(args.output.absolute() / "preparation.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
