"""One new60-family coverage segment; reuse existing admitted source and replay recipe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from data.schema import Sample
from train.checkpoint import file_sha256
from train.trainer import data_bytes

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
PRIOR_REPORT_SHA = "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b"
NAMESPACE = "s4-temporal-coverage-segment001:42:"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prior_families(samples: list[Sample]) -> frozenset[str]:
    require(
        len(samples) == 120 and all(s.meta.pool == "train" for s in samples),
        "prior must be audited120TRAIN",
    )
    video = [s for s in samples if s.meta.modality == ["video"]]
    families = frozenset(s.meta.family_id for s in video)
    require(len(video) == len(families) == 60, "prior video must have60unique families")
    return families


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/s4-temporal-coverage-pilot-001"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir.resolve()
    prior = root / "artifacts/s4-temporal-curriculum-pilot-001"
    report_path = prior / "report.json"
    require(not out.exists(), "coverage output already exists")
    require(file_sha256(report_path) == PRIOR_REPORT_SHA, "prior temporal report SHA changed")
    previous = json.loads(report_path.read_bytes())
    require(
        previous["status"] == "ready_for_preflight"
        and previous["exit_code"] == 0
        and previous["inventory_isolation"] == "isolated"
        and previous["source_admission"] == "verified_existing_scope",
        "prior temporal source not admitted isolated",
    )
    for name, digest in previous["output_sha256"].items():
        require(
            Path(name).name == name and file_sha256(prior / name) == digest,
            "prior output SHA mismatch:" + name,
        )
    samples = [
        Sample.model_validate_json(line)
        for line in (prior / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    excluded = prior_families(samples)
    for evidence in previous["admission_trace"]["files"]:
        require(
            file_sha256(Path(evidence["repository_path"])) == evidence["sha256"],
            "admission source evidence changed",
        )
    path = Path(__file__).with_name("prepare-temporal-curriculum-pilot.py")
    spec = importlib.util.spec_from_file_location("temporal_coverage_shared_recipe", path)
    assert spec and spec.loader
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.SOURCE_EVIDENCE = previous["admission_trace"]
    receipts = previous["input_report_sha256"]
    result = recipe.prepare(
        root,
        out,
        receipts["exposure004"],
        receipts["claim_preparation"],
        receipts["video_source"],
        video_hash_namespace=NAMESPACE,
        excluded_video_families=excluded,
        selection_provenance={
            "prior_report_sha256": PRIOR_REPORT_SHA,
            "prior_train_sha256": previous["output_sha256"]["train.jsonl"],
            "wrapper_sha256": file_sha256(Path(__file__)),
            "shared_recipe_sha256": file_sha256(path),
        },
    )
    fresh = [
        Sample.model_validate_json(line)
        for line in (out / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = frozenset(s.meta.family_id for s in fresh if s.meta.modality == ["video"])
    require(len(selected) == 60 and not selected & excluded, "new video families overlap prior60")

    # Exact source samples for every cross/old replay; no reweighting or rewriting.
    def replay(rows):
        return sorted((s.id, data_bytes([s])) for s in rows if s.meta.modality != ["video"])

    require(replay(fresh) == replay(samples), "cross/old replay changed")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "exit_code",
                    "inventory_isolation",
                    "video_unique_families",
                    "output_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
