"""Reconstruct actual video exposure on the verified S3→CE0 weight segment; CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import Sample
from train.trainer import LoopConfig, data_bytes, reference_order, smoke_samples

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
STAGES = (
    (
        "s3-stable",
        "s3-stable-probe-train-001",
        "s3-stable-probe-train-001",
        "1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755",
    ),
    (
        "first-joint",
        "s4-joint-pilot-train-001",
        "s4-joint-pilot-train-001",
        "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac",
    ),
    (
        "cross-coverage",
        "s4-crossmodal-retention-training-001",
        "s4-crossmodal-retention-train-001",
        "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818",
    ),
    (
        "ce0",
        "s4-claim-pair-ce0-training-001",
        "s4-claim-pair-ce0-train-001",
        "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed",
    ),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def kind(sample: Sample) -> str:
    words = set(sample.questions[0].instruction.lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else "motion"
        if words & {"moving", "stationary"}
        else "existence"
    )


def actual_video(samples: list[Sample], config: LoopConfig, steps_completed: int) -> list[Sample]:
    require(all(s.meta.pool == "train" for s in samples), "only TRAIN source samples accepted")
    require(0 < steps_completed <= config.steps, "invalid actual completed step count")
    order = reference_order(config.model_copy(update={"steps": steps_completed}), len(samples))
    return [samples[index] for index in order if "video" in samples[index].meta.modality]


def counts(rows: list[Sample]) -> dict[str, Any]:
    return {
        "video_draws": len(rows),
        "unique_video_ids": len({s.id for s in rows}),
        "unique_video_families": len({s.meta.family_id for s in rows}),
        "subgroup_draws": dict(Counter(kind(s) for s in rows)),
        "subgroup_unique_families": {
            task: len({s.meta.family_id for s in rows if kind(s) == task})
            for task in ("temporal", "motion", "existence")
        },
    }


def analyze(root: Path) -> dict[str, Any]:
    stages = []
    stage_families: dict[str, set[str]] = {}
    union = []
    prior_adapter = None
    canonical_by_id = {}
    all_hashes = {}
    for label, report_name, run_name, expected in STAGES:
        report_path = root / "reports" / (report_name + ".json")
        report = json.loads(report_path.read_bytes())
        require(
            report["status"] == "completed"
            and report["exit_code"] == 0
            and report["adapter_sha256"] == expected,
            "frozen lineage report status/adapter mismatch:" + label,
        )
        init = report["init_adapter"]
        require(
            isinstance(init, dict) and len(init["weights_sha256"]) == 64, "missing init receipt"
        )
        if prior_adapter is not None:
            require(
                init["weights_sha256"] == prior_adapter, "weight lineage link mismatch:" + label
            )
        prior_adapter = expected
        raw = dict(report["config"])
        raw.pop("init_adapter", None)
        config = LoopConfig.model_validate(raw)
        data = root / "artifacts" / run_name / "data.jsonl"
        require(
            sha(data) == report["data_sha256"], "actual saved training data SHA mismatch:" + label
        )
        samples = smoke_samples(
            config,
            data,
            expected_data_version=config.expected_data_version,
            expected_data_versions=config.expected_data_versions,
        )
        require(
            hashlib.sha256(data_bytes(samples)).hexdigest() == report["data_sha256"],
            "canonical saved data SHA mismatch",
        )
        require(report["steps_completed"] == config.steps, "lineage run incomplete")
        rows = actual_video(samples, config, report["steps_completed"])
        for sample in rows:
            # Media URI aliases changed in historical runs; bind the source semantic identity
            # and family without pretending paths are new original questions.
            identity = (
                sample.meta.family_id,
                sample.meta.source_sha256,
                sample.questions[0].target.model_dump_json(),
            )
            require(
                canonical_by_id.setdefault(sample.id, identity) == identity,
                "same video ID changed source/family/gold",
            )
        families = {s.meta.family_id for s in rows}
        seen = {s.meta.family_id for s in union}
        stages.append(
            {
                "name": label,
                "adapter_sha256": expected,
                "init_adapter": init,
                "report": str(report_path),
                "report_sha256": sha(report_path),
                "data": str(data),
                "data_sha256": sha(data),
                "seed": config.seed,
                "pool_samples": len(samples),
                "steps_completed": report["steps_completed"],
                "accumulation": config.accumulation,
                "actual_total_draws": report["steps_completed"] * config.accumulation,
                **counts(rows),
                "new_video_families_in_segment": len(families - seen),
                "overlap_previous_segment_families": len(families & seen),
            }
        )
        stage_families[label] = families
        union.extend(rows)
        all_hashes[label] = {"report": sha(report_path), "data": sha(data)}
    fresh_dir = root / "artifacts/s4-temporal-coverage-pilot-001"
    fresh_report = json.loads((fresh_dir / "report.json").read_bytes())
    require(
        fresh_report["status"] == "ready_for_preflight"
        and fresh_report["exit_code"] == 0
        and fresh_report["inventory_isolation"] == "isolated",
        "new coverage preparation failed",
    )
    for name, digest in fresh_report["output_sha256"].items():
        require(
            Path(name).name == name and sha(fresh_dir / name) == digest,
            "new coverage artifact changed",
        )
    fresh_config = LoopConfig.model_validate_json((fresh_dir / "config-ce.json").read_bytes())
    fresh_samples = smoke_samples(
        fresh_config,
        fresh_dir / "train.jsonl",
        expected_data_versions=fresh_config.expected_data_versions,
    )
    fresh = actual_video(fresh_samples, fresh_config, fresh_config.steps)
    fresh_families = {s.meta.family_id for s in fresh}
    lineage_families = {s.meta.family_id for s in union}
    require(
        len(fresh) == len(fresh_families) == 60,
        "new coverage must contain60unique planned video families",
    )
    overlaps = {label: len(fresh_families & families) for label, families in stage_families.items()}

    return {
        "status": "completed",
        "exit_code": 0,
        "gpu_used": False,
        "stages": stages,
        "lineage_union": counts(union),
        "lineage_unique_video_families": sorted(lineage_families),
        "lineage_subgroup_family_sets": {
            task: sorted({s.meta.family_id for s in union if kind(s) == task})
            for task in ("temporal", "motion", "existence")
        },
        "new_coverage_planned": {
            **counts(fresh),
            "overlap_lineage_family_count": len(fresh_families & lineage_families),
            "new_to_verified_lineage_segment": len(fresh_families - lineage_families),
            "overlap_by_stage": overlaps,
            "overlapping_families": sorted(fresh_families & lineage_families),
            "preparation_report_sha256": sha(fresh_dir / "report.json"),
            "not_claimed": (
                "planned source intersection only; newly running coverage not part of CE0 ancestors"
            ),
        },
        "upstream_boundary": {
            "s3_init_adapter": stages[0]["init_adapter"],
            "scope": (
                "verified segment begins at S3 video introduction; earlier adapters are "
                "anchored but not recursively audited"
            ),
        },
        "scope": (
            "actual fixed-shuffle modulo draws on CE0 ancestor segment only; "
            "not full8299pool and not all historical experiments"
        ),
        "runner_sha256": sha(Path(__file__)),
        "bindings": all_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "lineage_union": result["lineage_union"],
                "new_coverage_planned": result["new_coverage_planned"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
