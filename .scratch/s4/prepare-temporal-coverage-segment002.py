"""CPU data-only second coverage segment; parent choice and training are not authorized."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from data.schema import Sample
from train.checkpoint import file_sha256
from train.trainer import data_bytes

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
LINEAGE_SHA = "cb9574e06fc9950198f703ab9107d6d2d42824a5d2399ded2a26a146dd88cbfa"
FIRST_SHA = "1e7f877e04ea311d29ebc1fe1711f31aa2cc91034f8794b666850bcf5f2fff19"
PRIOR_SHA = "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b"
NAMESPACE = "s4-temporal-coverage-segment002:42:"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path):
    return json.loads(path.read_bytes())


def replay(rows):
    return sorted((s.id, data_bytes([s])) for s in rows if s.meta.modality != ["video"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/s4-temporal-coverage-pilot-002"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir.resolve()
    require(not out.exists(), "segment002 output already exists")
    lineage_path = root / "reports/s4-ce0-video-exposure-001.json"
    require(file_sha256(lineage_path) == LINEAGE_SHA, "verified CE0 lineage SHA changed")
    lineage = read(lineage_path)
    require(lineage["status"] == "completed" and lineage["exit_code"] == 0, "lineage incomplete")
    old = frozenset(lineage["lineage_unique_video_families"])
    require(len(old) == 350, "expected verified350CE0families")
    for stage in lineage["stages"]:
        require(
            file_sha256(Path(stage["report"])) == stage["report_sha256"]
            and file_sha256(Path(stage["data"])) == stage["data_sha256"],
            "lineage source binding changed",
        )
    first = root / "artifacts/s4-temporal-coverage-pilot-001"
    require(file_sha256(first / "report.json") == FIRST_SHA, "first coverage preparation changed")
    report = read(first / "report.json")
    require(
        report["status"] == "ready_for_preflight" and report["inventory_isolation"] == "isolated",
        "first source invalid",
    )
    for name, digest in report["output_sha256"].items():
        require(
            Path(name).name == name and file_sha256(first / name) == digest, "first output changed"
        )
    first_rows = [
        Sample.model_validate_json(line)
        for line in (first / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    require(all(s.meta.pool == "train" for s in first_rows), "first rows must be TRAIN")
    first_families = frozenset(s.meta.family_id for s in first_rows if s.meta.modality == ["video"])
    require(len(first_families) == 60, "first coverage must have60video families")
    excluded = old | first_families
    require(len(excluded) == 408, "verified lineage/first union changed")
    prior = root / "artifacts/s4-temporal-curriculum-pilot-001"
    require(
        file_sha256(prior / "report.json") == PRIOR_SHA, "original replay source report changed"
    )
    previous = read(prior / "report.json")
    for name, digest in previous["output_sha256"].items():
        require(
            Path(name).name == name and file_sha256(prior / name) == digest,
            "original replay source changed",
        )
    for evidence in previous["admission_trace"]["files"]:
        require(
            file_sha256(Path(evidence["repository_path"])) == evidence["sha256"],
            "admission evidence changed",
        )
    path = Path(__file__).with_name("prepare-temporal-curriculum-pilot.py")
    spec = importlib.util.spec_from_file_location("segment002_shared_recipe", path)
    assert spec and spec.loader
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.SOURCE_EVIDENCE = previous["admission_trace"]
    pins = previous["input_report_sha256"]
    result = recipe.prepare(
        root,
        out,
        pins["exposure004"],
        pins["claim_preparation"],
        pins["video_source"],
        video_hash_namespace=NAMESPACE,
        excluded_video_families=excluded,
        data_ready_only=True,
        selection_provenance={
            "lineage_report_sha256": LINEAGE_SHA,
            "first_coverage_report_sha256": FIRST_SHA,
            "original_replay_report_sha256": PRIOR_SHA,
            "wrapper_sha256": file_sha256(Path(__file__)),
            "shared_recipe_sha256": file_sha256(path),
            "parent_selection": "pending_root_retention_decision",
        },
    )
    fresh = [
        Sample.model_validate_json(line)
        for line in (out / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    families = {s.meta.family_id for s in fresh if s.meta.modality == ["video"]}
    require(
        len(families) == 60 and not families & excluded, "segment002 violates new-family exclusion"
    )
    require(replay(fresh) == replay(first_rows), "segment002 cross/old replay changed")
    require(
        result["status"] == "data_ready_only"
        and result["pending_parent_selection"]
        and not result["training_authorized"],
        "segment002 parent selection must remain pending",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "exit_code",
                    "inventory_isolation",
                    "pending_parent_selection",
                    "training_authorized",
                    "output_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
