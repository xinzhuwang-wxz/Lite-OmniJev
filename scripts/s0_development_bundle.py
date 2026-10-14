"""Freeze development family/media exclusion across all six cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.s0_development_baselines import bind_build_report, load_samples, validate_samples
from scripts.s0_public_baselines import sha256_file, source_identity

COHORTS = {"text", "image", "screenshot", "audio", "video", "crossmodal"}


def exclusions(cohorts: dict[str, list[Sample]]) -> list[dict[str, Any]]:
    """Join shared media into components; every family in a component stays dev."""
    parents: dict[str, str] = {}
    media_owner: dict[str, str] = {}
    families: dict[str, dict[str, Any]] = {}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(left: str, right: str) -> None:
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    for cohort, samples in sorted(cohorts.items()):
        for sample in samples:
            if sample.meta.pool != "dev":
                raise ValueError("all bundled samples must belong to dev")
            key = json.dumps(
                [sample.meta.source_dataset, sample.meta.family_id], separators=(",", ":")
            )
            parents.setdefault(key, key)
            row = families.setdefault(
                key,
                {
                    "source_dataset": sample.meta.source_dataset,
                    "family_id": sample.meta.family_id,
                    "cohorts": set(),
                    "item_ids": set(),
                    "media_sha256": set(),
                },
            )
            row["cohorts"].add(cohort)
            row["item_ids"].add(sample.id)
            for media in sample.prefix.media:
                if not media.sha256:
                    raise ValueError("media hash required for cross-source exclusion")
                row["media_sha256"].add(media.sha256)
                if media.sha256 in media_owner:
                    join(key, media_owner[media.sha256])
                else:
                    media_owner[media.sha256] = key
    # Every crossmodal component must link to original image and audio cohorts.
    members: dict[str, list[str]] = {}
    for key in families:
        members.setdefault(find(key), []).append(key)
    output = []
    for keys in members.values():
        covered = set().union(*(families[key]["cohorts"] for key in keys))
        if "crossmodal" in covered and not {"image", "audio"} <= covered:
            raise ValueError("crossmodal family must resolve to both original dev components")
        component = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
        for key in keys:
            row = families[key]
            output.append(
                {
                    "source_dataset": row["source_dataset"],
                    "family_id": row["family_id"],
                    "pool": "dev",
                    "component_id": component,
                    "cohorts": sorted(row["cohorts"]),
                    "item_ids": sorted(row["item_ids"]),
                    "media_sha256": sorted(row["media_sha256"]),
                }
            )
    return sorted(output, key=lambda row: (row["source_dataset"], row["family_id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    entries = json.loads(args.cohorts.read_text())["cohorts"]
    if len(entries) != 6 or {entry["name"] for entry in entries} != COHORTS:
        parser.error("all six distinct cohorts required")
    datasets = {}
    sources = []
    for entry in entries:
        data, build_report = Path(entry["data"]), Path(entry["report"])
        bind_build_report(build_report, data)
        samples = load_samples(data)
        validate_samples(samples, cohort=entry["name"])
        datasets[entry["name"]] = samples
        sources.append(
            {**entry, "dev_sha256": sha256_file(data), "report_sha256": sha256_file(build_report)}
        )
    rows = exclusions(datasets)
    args.output.mkdir(parents=True)
    ledger = args.output / "dev-exclusion.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )
    report = {
        "status": "passed",
        "exit_code": 0,
        "source": source_identity(),
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.cohorts),
        "sources": sources,
        "cohort_rows": {name: len(items) for name, items in datasets.items()},
        "family_nodes": len(rows),
        "connected_components": len({row["component_id"] for row in rows}),
        "original_family_nodes": sum("crossmodal" not in row["cohorts"] for row in rows),
        "ledger_sha256": sha256_file(ledger),
        "ledger_path": str(ledger),
        "scope": (
            "dev exclusions only; future training/calibration/holdout must reject "
            "these families and media aliases"
        ),
        "calibration_allocation": "pending",
        "holdout_allocation": "external custodian pending",
        "quality_acceptance": False,
        "gpu_used": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
