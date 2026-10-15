"""Hash existing Decision Index material; never execute recipes or read benchmark rows.

Run on the host holding inputs. Produces a PreparationManifest plus a separate
closure report. License files are evidence candidates, never inferred admissions.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
from pathlib import Path

# Our own reviewed preparation schema only; no upstream imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.decision_index_offline import (  # noqa: E402
    ARCHIVE_SHA256,
    CATALOG_IDS,
    RAW,
    REVISION,
    PreparationManifest,
    checked_file,
    digest,
    relative_path,
)

# File selectors transcribed from the pinned build adapters; no aliases or execution.
RECIPE_FILES = {
    "bfcl": [
        "berkeley-function-call-leaderboard/data/BFCL_v3_*.json",
        "berkeley-function-call-leaderboard/data/possible_answer/BFCL_v3_*.json",
    ],
    "toolret": ["toolret/config.py"],
    "apibank": ["api-bank/apis/*.py", "api-bank/lv1-lv2-samples/*.jsonl"],
    "sgd": ["test/schema.json", "test/dialogues_*.json"],
    "contractnli": ["resources/contract-nli.zip"],
    "pop909cl": ["POP909_processed/*.mid"],
    "searchless_chess": [],
    "esci": [
        "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
        "shopping_queries_dataset/shopping_queries_dataset_products.parquet",
    ],
    "acos": ["data/*/*_quad_*.tsv"],
    "finentity": ["data/FinEntity.json"],
    "isarcasm": ["test/*.csv"],
    "nli4ct": ["test.json", "gold_test.json", "training_data.zip"],
    "cruxeval": [
        "data/cruxeval.jsonl",
        "samples/model_generations.zip",
        "samples/evaluation_results.zip",
    ],
    "forecastbench-datasets": [
        "datasets/question_sets/*.json",
        "datasets/resolution_sets/*_resolution_set.json",
    ],
    "habermas_machine": [],
    "sata": ["src/satabench/methods/data/sata_bench_final_2025.json"],
    "musr": ["datasets/*.json"],
    "simplebench": ["simple_bench_public.json"],
    "cladder": ["data/cladder-v1.zip"],
    "vast": ["data/VAST/vast_test.csv"],
    "cand-BIG-Bench-Hard": ["bbh/*.json"],
    "cand-RAGTruth": ["dataset/response.jsonl", "dataset/source_info.jsonl"],
    "cand-hover": ["data/hover/hover_dev_release_v1.1.json"],
    "jev-phishing-bench": ["run_jev.py"],
}


def constants(path: Path, names: set[str]) -> dict:
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != names:
        raise ValueError(f"missing literal recipe constants: {names - result.keys()}")
    return result


def build(root: Path, router: Path, recipe: Path, archive: Path) -> tuple[dict, dict]:
    for path in (root, router, recipe, archive):
        if not path.is_absolute():
            raise ValueError("all inputs must be absolute")
    if recipe.name != f"decision-index-{REVISION}":
        raise ValueError("unexpected upstream revision directory")
    base = recipe / "decision_index/suite/build"
    pinned = {
        "acquire.py": "94bc9997f9c95ae33ed9d2d00e6c715bd1408b69ecb64de9e31e06317b761c16",
        "adapters_added.py": "66bdf8a82b33d5ffa8d10c25f701904620ec52da41ad040bf752879bad1291b2",
    }
    for name, sha in pinned.items():
        if digest(base / name) != sha:
            raise ValueError("recipe constants differ from audited revision")
    acq = constants(base / "acquire.py", {"GIT", "GIT_FLAT", "HF", "HTTP", "NEEDS"})
    added = constants(base / "adapters_added.py", {"GIT", "HF", "NEEDS", "HOVER_DB", "FILE_SHA"})
    specs = {}
    for table in (acq["NEEDS"], added["NEEDS"]):
        for catalog_id, needs in table.items():
            for spec in needs:
                specs.setdefault(spec, set()).add(catalog_id)
    rules = []
    for spec, ids in sorted(specs.items()):
        kind, key = spec.split(":", 1)
        origin = added if spec in {s for v in added["NEEDS"].values() for s in v} else acq
        expected = {}
        patterns = None
        if kind == "git":
            source = root / "repos" / key
            destination = RAW + "repos/" + key
            sparse = origin["GIT"][key][2]
            patterns = [str(p).rstrip("/") + "/*" for p in sparse] if sparse else None
        elif kind == "gitflat":
            source, destination = root / "raw" / key, RAW + key
            sparse = acq["GIT_FLAT"][key][2]
            patterns = [str(p).rstrip("/") + "/*" for p in sparse] if sparse else None
        elif kind == "http":
            source = root / key
            destination = RAW + key[4:] if key.startswith("raw/") else key
            expected[""] = acq["HTTP"][key][1]
        elif kind == "db":
            rel, _, sha = added["HOVER_DB"]
            source, destination = root / rel, RAW + rel
            expected[""] = sha
        elif kind == "hf":
            source = root / key if origin is acq else root / "raw" / key
            destination = (
                (RAW + key[4:] if key.startswith("raw/") else key) if origin is acq else RAW + key
            )
            if origin is added:
                _, _, rel, sha = added["HF"][key]
                patterns, expected = [rel], {rel: sha}
            else:
                patterns = acq["HF"][key][2]
        else:
            raise ValueError("unknown acquisition kind")
        if kind in {"git", "gitflat"}:
            patterns = RECIPE_FILES[key]
        # Receipt reports/hle-source-download-001.json fixes an authorized alternative
        # source location; destination remains exactly the pinned Layout path.
        if spec == "hf:raw/hle" and not source.exists():
            source = (
                root.parents[3]
                / "artifacts/data/research-eval/hle"
                / "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
                / "test-00000-of-00001.parquet"
            )
            destination = RAW + "hle/data/test-00000-of-00001.parquet"
            patterns = None
            expected = {"": "6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a"}
        rules.append((spec, sorted(ids), source, destination, patterns, expected))
    files, mapped, missing, licenses, rejected, unmapped = [], {}, [], [], [], []
    for spec, ids, source, destination, patterns, expected in rules:
        candidates = (
            [source] if source.is_file() else sorted(source.rglob("*")) if source.is_dir() else []
        )
        found = set()
        for path in candidates:
            if not path.is_file() or path.name.startswith("._"):
                continue
            rel = "" if path == source else path.relative_to(source).as_posix()
            if any(part in {".cache", ".git", "__pycache__"} for part in path.parts):
                continue
            if path.is_symlink():
                rejected.append({"source": str(path), "reason": "symlink"})
                continue
            dest = destination + ("/" + rel if rel else "")
            relative_path(dest)
            license_candidate = path.name.lower().startswith(("license", "copying", "notice"))
            if patterns is not None and not any(
                fnmatch.fnmatch(rel, pattern) for pattern in patterns
            ):
                if license_candidate:
                    licenses.append(
                        {
                            "path": str(path),
                            "sha256": digest(path),
                            "catalog_ids": ids,
                            "decision": "candidate evidence only",
                        }
                    )
                unmapped.append(str(path))
                continue
            actual = digest(path)
            if rel in expected and actual != expected[rel]:
                rejected.append({"source": str(path), "reason": "pinned SHA mismatch"})
                continue
            found.add(rel)
            if dest in mapped:
                item = mapped[dest]
                if item["sha256"] != actual:
                    raise ValueError(f"conflicting destination: {dest}")
                item["catalog_ids"] = sorted(set(item["catalog_ids"]) | set(ids))
            else:
                item = {
                    "path": str(path),
                    "sha256": actual,
                    "destination": dest,
                    "catalog_ids": ids,
                }
                mapped[dest] = item
                files.append(item)
            if path.name.lower().startswith(("license", "copying", "notice")):
                licenses.append(
                    {
                        "path": str(path),
                        "sha256": actual,
                        "catalog_ids": ids,
                        "decision": "candidate evidence only; not dataset admission",
                    }
                )
        if not candidates:
            missing.append({"spec": spec, "source": str(source), "reason": "missing input"})
        for pattern in patterns or []:
            if not any(fnmatch.fnmatch(rel, pattern) for rel in found):
                missing.append({"spec": spec, "source": str(source), "pattern": pattern})
        if spec.startswith(("git:", "gitflat:")) and not (source / ".git").is_dir():
            missing.append({"spec": spec, "reason": "git provenance directory missing"})
    for shot in (0, 5):
        path = router / f"routerbench_{shot}shot.jsonl"
        if path.is_file() and not path.is_symlink():
            item = {
                "path": str(path),
                "sha256": digest(path),
                "destination": RAW + f"routerbench/{path.name}",
                "catalog_ids": [6],
            }
            files.append(item)
        else:
            missing.append({"source": str(path), "reason": "converted RouterBench missing"})
    # Exact adapter-added file pins are separate from acquisition directory presence.
    for rel, expected_sha in added["FILE_SHA"].items():
        item = mapped.get(RAW + "repos/" + rel)
        if item is None or item["sha256"] != expected_sha:
            missing.append(
                {
                    "source": str(root / "repos" / rel),
                    "reason": "required pinned file missing/mismatch",
                }
            )
    manifest = PreparationManifest.model_validate(
        {
            "source_archive": {"path": str(archive), "sha256": ARCHIVE_SHA256},
            "files": files,
            "admissions": [],
        }
    )
    archive_ok = False
    try:
        checked_file(manifest.source_archive)
        archive_ok = True
    except (ValueError, OSError) as exc:
        missing.append({"source": str(archive), "reason": str(exc)})
    destinations = [item.destination for item in manifest.files]
    if len(set(destinations)) != len(destinations):
        raise ValueError("duplicate output destination")
    report = {
        "ready": False,
        "status": "input_mapping_only",
        "revision": REVISION,
        "source_archive_verified": archive_ok,
        "file_count": len(files),
        "input_bytes": sum(Path(item["path"]).stat().st_size for item in files),
        "unmapped_noncache_files": sorted(set(unmapped)),
        "git_provenance": (
            "Excluded .git from material mapping; git_revision needs "
            "separate safe provenance closure"
        ),
        "recipe_basis": {
            str(path): digest(path)
            for path in sorted(base.glob("*.py"))
            if not path.name.startswith("._")
        },
        "missing": missing,
        "rejected": rejected,
        "license_evidence_candidates": licenses,
        "evaluation_admissions_missing": sorted(CATALOG_IDS),
        "pending": [
            "Complete recipe per-file closure (directory presence is insufficient)",
            "Bind explicit dataset evaluation-license decisions",
            "Exact offline tiktoken cache",
            "Official isolated rebuild and verify",
        ],
        "boundaries": [
            "No upstream execution; no pickle; only file names, source text and hashes read",
            "downloadable_verified never implies evaluation_admitted",
        ],
    }
    return manifest.model_dump(mode="json"), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "router", "recipe", "source-archive"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest, report = build(args.root, args.router, args.recipe, args.source_archive)
    # Caller writes the manifest and separate report, avoiding accidental overwrite here.
    print(json.dumps({"manifest": manifest, "report": report}, indent=2))


if __name__ == "__main__":
    main()
