"""CPU-only preparation of one 360-draw crossmodal-retention hypothesis.

Preserves every second-pilot draw; appends two unseen TRAIN crossmodal items per
step. No dev-driven selection, gold/media mutation, GPU launch or gate claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample
from scripts.gold_inventory import ComponentRecord, Manifest, audit_manifest, bound_lines
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples, verified_init_adapter

INIT_SHA = "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac"
SECOND_SHA = "14496e283908180303e6d48d0983dbd4c574b4f35f7bb350acf59d9c2a176c9c"
RULES = ("both_true", "image_false_audio_true", "image_true_audio_false")


def require(condition: bool, category: str) -> None:
    if not condition:
        raise ValueError(category)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def train_rows(path: Path) -> list[dict[str, Any]]:
    result = rows(path)
    samples = [Sample.model_validate(row) for row in result]
    require(all(sample.meta.pool == "train" for sample in samples), "non_train_input")
    require(len({sample.id for sample in samples}) == len(samples), "duplicate_train_ids")
    return result


def write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")


def actual_draws(root: Path, name: str, expected_sha: str, count: int) -> tuple[list, dict]:
    report_path = root / f"reports/{name}-train-001.json"
    report = read(report_path)
    require(
        report["status"] == "completed"
        and report["exit_code"] == 0
        and report["steps_completed"] == 30,
        "prior_training_incomplete",
    )
    require(report["adapter_sha256"] == expected_sha, "prior_adapter_lineage")
    config = report["config"]
    require(
        config["seed"] == 42
        and config["steps"] == 30
        and config["accumulation"] == 10
        and config["sample_count"] == count,
        "prior_actual_draw_config",
    )
    prepared = root / f"artifacts/{name}-001"
    receipt = read(prepared / "report.json")
    require(receipt["status"] == "prepared", "prior_preparation_status")
    data = prepared / "train.jsonl"
    key = "train" if name == "s4-joint-pilot" else "train.jsonl"
    require(sha(data) == receipt["output_sha256"][key], "prior_prepared_data_hash")
    source = train_rows(data)
    require(len(source) == count, "prior_data_count")
    runtime = root / f"artifacts/{name}-train-001/data.jsonl"
    canonical = data_bytes([Sample.model_validate(row) for row in source])
    require(
        sha(runtime) == report["data_sha256"]
        and hashlib.sha256(canonical).hexdigest() == report["data_sha256"],
        "prior_actual_runtime_data_binding",
    )
    order = list(range(count))
    random.Random(42).shuffle(order)
    draws = [source[index] for index in order[:300]]
    return draws, {
        "model_revision": report["model_revision"],
        "report": {"path": str(report_path), "sha256": sha(report_path)},
        "prepared_data": {"path": str(data), "sha256": sha(data)},
        "runtime_data": {"path": str(runtime), "sha256": sha(runtime)},
        "draw_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in draws).encode()
        ).hexdigest(),
    }


def inverse_arrange(original_draws: list, added: list) -> list:
    require(len(original_draws) == 300 and len(added) == 60, "draw_counts")
    desired = []
    for step in range(30):
        desired.extend(original_draws[step * 10 : step * 10 + 10] + added[step * 2 : step * 2 + 2])
    order = list(range(360))
    random.Random(42).shuffle(order)
    arranged = list(desired)
    for draw, index in enumerate(order):
        arranged[index] = desired[draw]
    require([arranged[index] for index in order] == desired, "inverse_shuffle_failure")
    for step in range(30):
        require(
            [arranged[index] for index in order[step * 12 : step * 12 + 10]]
            == original_draws[step * 10 : step * 10 + 10],
            "changed_original_step_draw_order",
        )
    return arranged


def prepare(args: argparse.Namespace) -> dict:
    root, out = args.root, args.output_dir
    require(
        all(path.is_absolute() for path in (root, out, args.model_dir, args.asset_manifest)),
        "absolute_paths",
    )
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    start = time.monotonic()
    facts: dict[str, Any] = {}
    try:
        first, first_binding = actual_draws(root, "s4-joint-pilot", INIT_SHA, 1200)
        old, second_binding = actual_draws(root, "s4-video-pilot", SECOND_SHA, 300)
        require(
            args.model_dir.name
            == first_binding["model_revision"]
            == second_binding["model_revision"],
            "frozen_model_revision",
        )
        cross = root / "artifacts/s4-crossmodal-train-001"
        provenance = read(cross / "provenance.json")
        require(provenance["count"] == 200, "crossmodal_admission_count")
        for key, name in [
            ("train", "train.jsonl"),
            ("allocation", "allocation.jsonl"),
            ("components", "components.jsonl"),
            ("inventory", "inventory-manifest.json"),
        ]:
            require(
                sha(cross / name) == provenance["output_sha256"][key],
                "crossmodal_admission_binding",
            )
        candidates = train_rows(cross / "train.jsonl")
        require(len(candidates) == 200, "crossmodal_count")
        components = [
            ComponentRecord.model_validate(row) for row in rows(cross / "components.jsonl")
        ]
        require(all(row.pool == "train" for row in components), "non_train_components")
        component_by_id = {row.item_id: row for row in components}
        require(
            len(component_by_id) == 200
            and set(component_by_id) == {row["id"] for row in candidates},
            "crossmodal_component_identity",
        )
        banned = {row["id"] for row in [*first, *old]}
        component_ids = set(component_by_id)
        facts.update(
            first_seen_cross=len({r["id"] for r in first} & component_ids),
            second_seen_cross=len({r["id"] for r in old} & component_ids),
            excluded_seen_cross=len(banned & component_ids),
        )
        buckets = {rule: [] for rule in RULES}
        for row in candidates:
            sample = Sample.model_validate(row)
            component = component_by_id[sample.id]
            require(
                len(sample.questions) == 1
                and isinstance(sample.questions[0].target, HardTarget)
                and sample.questions[0].target.value == component.program_truth
                and component.proposal_rule in RULES,
                "crossmodal_gold_rule_binding",
            )
            if sample.id not in banned:
                buckets[component.proposal_rule].append(row)
        capacity = {rule: len(values) for rule, values in buckets.items()}
        facts["eligible_rule_capacity"] = capacity
        require(all(count >= 20 for count in capacity.values()), "rule_remaining_capacity_below20")
        selected = {
            rule: sorted(
                values,
                key=lambda row: hashlib.sha256(
                    ("s4-crossmodal-retention:42:" + row["id"]).encode()
                ).hexdigest(),
            )[:20]
            for rule, values in buckets.items()
        }
        # Fixed rule cycle: each three-step block adds two of every rule, 20/rule total.
        added = [selected[rule][index] for index in range(20) for rule in RULES]
        data = inverse_arrange(old, added)
        require(len({row["id"] for row in data}) == 360, "duplicate_candidate_ids")
        require(not ({row["id"] for row in added} & banned), "previously_seen_cross_selected")
        data_path = out / "train.jsonl"
        write(
            data_path,
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in data),
        )
        second = root / "artifacts/s4-video-pilot-001"
        second_receipt = read(second / "report.json")
        config_path = second / "config-30.json"
        require(
            sha(config_path) == second_receipt["output_sha256"]["config-30.json"],
            "second_config_binding",
        )
        old_config = read(config_path)
        require(
            old_config.get("reference_kl_weight", 0) == 0
            and not any(
                key.startswith("reference_") and value for key, value in old_config.items()
            ),
            "reference_must_be_off",
        )
        config = {**old_config, "sample_count": 360, "accumulation": 12}
        require(
            {
                key: value
                for key, value in config.items()
                if key not in ("sample_count", "accumulation")
            }
            == {
                key: value
                for key, value in old_config.items()
                if key not in ("sample_count", "accumulation")
            },
            "unexpected_config_change",
        )
        validated = LoopConfig.model_validate(config)
        smoke_samples(
            validated,
            data_path,
            expected_data_version=config.get("expected_data_version"),
            expected_data_versions=config.get("expected_data_versions"),
        )
        write(out / "config-30.json", config)
        ledger = second / "allocation-ledger.jsonl"
        require(
            sha(ledger) == second_receipt["output_sha256"]["allocation-ledger.jsonl"],
            "second_ledger_binding",
        )
        combined_ledger = out / "allocation-ledger.jsonl"
        # Full existing TRAIN allocation union; no pool reassignment or new ledger invention.
        write(
            combined_ledger,
            ledger.read_text().rstrip("\n")
            + "\n"
            + (cross / "allocation.jsonl").read_text().rstrip("\n")
            + "\n",
        )
        ledger_check = verify_data_is_train_pool(data_path, [combined_ledger])
        exposure = root / "artifacts/s4-pilot-exposure-002"
        exposure_report = read(exposure / "report.json")
        require(
            exposure_report["status"] == "known_pilot_exposures_reconciled"
            and exposure_report["exit_code"] == 0
            and exposure_report["unknown_update_bearing_inputs"] == 0,
            "latest_canonical_exposure_reconciliation",
        )
        inventory_path = exposure / "inventory-manifest.json"
        expected_inventory_sha = exposure_report["output_sha256"]["inventory-manifest.json"]
        require(sha(inventory_path) == expected_inventory_sha, "inventory_binding")
        inventory = Manifest.model_validate_json(inventory_path.read_bytes())
        combined = inventory.model_dump(mode="json", exclude_none=True)
        superseded = []
        selected_ids = {row["id"] for row in data}
        versions = {}
        # Recover exact prior cohort mapping; do not infer screenshot from image modality.
        for artifact in inventory.samples:
            if artifact.expected_pool != "train":
                continue
            for line in bound_lines(artifact):
                row = json.loads(line)
                if row["id"] in selected_ids:
                    cohort = artifact.cohort or (artifact.cohorts_by_data_version or {}).get(
                        row["meta"]["data_version"]
                    )
                    require(cohort is not None, "missing_prior_cohort")
                    version = row["meta"]["data_version"]
                    require(
                        version not in versions or versions[version] == cohort,
                        "cohort_version_conflict",
                    )
                    versions[version] = cohort
        for index, artifact in enumerate(inventory.samples):
            if (
                artifact.expected_pool == "train"
                and artifact.exposure == "current"
                and any(json.loads(line)["id"] in selected_ids for line in bound_lines(artifact))
            ):
                combined["samples"][index]["exposure"] = "historical"
                superseded.append({"path": artifact.path, "sha256": artifact.sha256})
        require(
            set(versions) == {row["meta"]["data_version"] for row in data},
            "candidate_versions_not_covered",
        )
        combined["samples"].append(
            {
                "path": str(data_path),
                "sha256": sha(data_path),
                "exposure": "current",
                "expected_pool": "train",
                "cohorts_by_data_version": versions,
            }
        )
        write(out / "inventory-manifest.json", combined)
        audit = audit_manifest(out / "inventory-manifest.json")
        write(out / "inventory-audit.json", audit.model_dump(mode="json"))
        require(audit.outcome == "isolated", "inventory_isolation_rejected")
        adapter = root / "artifacts/s4-joint-pilot-train-001/adapter"
        identity = verified_init_adapter(adapter, INIT_SHA)
        common = [
            "--config",
            str(out / "config-30.json"),
            "--model-dir",
            str(args.model_dir),
            "--asset-manifest",
            str(args.asset_manifest),
            "--data",
            str(data_path),
            "--allocation-ledger",
            str(combined_ledger),
            "--output-dir",
            str(root / "artifacts/s4-crossmodal-retention-train-001"),
            "--report",
            str(root / "reports/s4-crossmodal-retention-training-001.json"),
            "--init-adapter",
            str(adapter),
            "--init-adapter-sha256",
            INIT_SHA,
        ]
        order = list(range(360))
        random.Random(42).shuffle(order)
        preflight_common = list(common)
        preflight_common[preflight_common.index("--report") + 1] = str(
            root / "reports/s4-crossmodal-retention-preflight-001.json"
        )
        result = {
            "status": "prepared",
            "exit_code": 0,
            "gpu_used": False,
            "training_started": False,
            "stage_gate": False,
            "scope": "one TRAIN coverage candidate; known lineage only; no quality admission",
            "exposure_audit": facts,
            "seed": 42,
            "steps": 30,
            "accumulation": 12,
            "total_unique_draws": 360,
            "preserved_prior_draws": 300,
            "added_draws": 60,
            "added_rule_counts": dict(
                Counter(component_by_id[row["id"]].proposal_rule for row in added)
            ),
            "eligible_rule_capacity": capacity,
            "excluded_seen_ids": len(banned),
            "original10_order_preserved": True,
            "added_ids_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in added).encode()
            ).hexdigest(),
            "actual_draw_ids_sha256": hashlib.sha256(
                "\n".join(data[i]["id"] for i in order).encode()
            ).hexdigest(),
            "prior_draw_bindings": {"first": first_binding, "second": second_binding},
            "init_adapter": identity,
            "source_bindings": {
                "cross_provenance": sha(cross / "provenance.json"),
                "latest_exposure_report": sha(exposure / "report.json"),
                "second_config": sha(config_path),
                "inventory": {"path": str(inventory_path), "sha256": sha(inventory_path)},
            },
            "superseded_current_train_artifacts": superseded,
            "inventory_audit": "isolated",
            "ledger_check": ledger_check,
            "output_sha256": {
                name: sha(out / name)
                for name in (
                    "train.jsonl",
                    "config-30.json",
                    "allocation-ledger.jsonl",
                    "inventory-manifest.json",
                    "inventory-audit.json",
                )
            },
            "preflight_argv": [
                args.python,
                "-m",
                "scripts.train_s1",
                "preflight",
                *preflight_common,
            ],
            "foreground_run_argv": [
                "timeout",
                "--signal=INT",
                "--kill-after=30s",
                "600s",
                args.python,
                "-m",
                "scripts.train_s1",
                "run",
                *common,
            ],
            "wallclock_limit_seconds": 600,
            "model_asset_preflight": "not_run",
            "cost": None,
            "script_sha256": sha(Path(__file__)),
            "python": platform.python_version(),
            "duration_seconds": time.monotonic() - start,
        }
        write(out / "report.json", result)
        return result
    except Exception as exc:
        if not (out / "report.json").exists():
            write(
                out / "report.json",
                {
                    "status": "blocked",
                    "exit_code": 1,
                    "gpu_used": False,
                    "failure_category": str(exc),
                    "aggregate_audit": facts,
                    "duration_seconds": time.monotonic() - start,
                    "cost": None,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument(
        "--python", default="/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.root / "artifacts/s4-crossmodal-retention-pilot-001"
    os.umask(0o077)
    try:
        report = prepare(args)
        print(
            json.dumps({"status": report["status"], "total_unique_draws": 360, "gpu_used": False})
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "exit_code": 1,
                    "failure_category": str(exc),
                    "gpu_used": False,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
