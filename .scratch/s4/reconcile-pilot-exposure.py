"""Reconcile terminal pilot canonical TRAIN inputs with known saved exposures.

CPU-only. The full inventory is passed only to the existing aggregate audit.
Original inventories/indexes/reports remain unchanged. Unknown saved training
SHA values reject reconciliation; this is not proof of unknown external history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.gold_inventory import Manifest, audit_manifest

TERMINAL = {"completed", "failed", "blocked"}
PILOTS = ("s4-joint-pilot", "s4-video-pilot")


def require(value: bool, category: str) -> None:
    if not value:
        raise ValueError(category)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path) -> dict[str, str]:
    require(path.is_absolute(), "absolute_binding")
    return {"path": str(path), "sha256": sha(path)}


def read_json(path: Path) -> tuple[Any, dict[str, str]]:
    binding = bind(path)
    payload = json.loads(path.read_bytes())
    require(sha(path) == binding["sha256"], "mutable_report")
    return payload, binding


def write(path: Path, payload: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def canonical_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        sample = Sample.model_validate_json(line)
        require(sample.meta.pool == "train", "non_train_canonical_input")
        require(sample.id not in result, "duplicate_sample_id")
        result[sample.id] = sample.model_dump(mode="json")
    return result


def equivalent(runtime: Path, preparation: Path) -> dict[str, Any]:
    # Full normalized Sample equality includes supervision, metadata and every media URI.
    first, second = canonical_rows(runtime), canonical_rows(preparation)
    require(first == second and bool(first), "canonical_sample_mismatch")
    return {
        "rows": len(first),
        "data_versions": sorted({r["meta"]["data_version"] for r in first.values()}),
    }


def terminal_report(report: dict[str, Any]) -> None:
    require(report.get("status") in TERMINAL, "nonterminal_report")
    require(
        type(report.get("steps_completed")) is int and report["steps_completed"] >= 0,
        "invalid_steps",
    )
    require(type(report.get("exit_code")) is int, "missing_final_exit_code")
    require(
        report["status"] != "completed" or report["exit_code"] == 0, "inconsistent_final_status"
    )


def version_mapping(inventory: Manifest) -> dict[str, str]:
    # Only explicit inventory declarations; do not infer cohort from sample modalities.
    result: dict[str, str] = {}
    for artifact in inventory.samples:
        if artifact.expected_pool != "train" or not artifact.cohorts_by_data_version:
            continue
        for version, cohort in artifact.cohorts_by_data_version.items():
            require(
                version not in result or result[version] == cohort, "inventory_version_conflict"
            )
            result[version] = cohort
    require(bool(result), "missing_inventory_mixed_mapping")
    return result


def pilot_paths(
    root: Path,
    *,
    crossmodal_retention: bool = False,
    claim_pair: bool = False,
    temporal_curriculum: bool = False,
) -> list[tuple[str, Path, Path, Path]]:
    paths = [
        (
            name,
            root / "reports" / (name + "-train-001.json"),
            root / "artifacts" / (name + "-train-001"),
            root / "artifacts" / (name + "-001"),
        )
        for name in PILOTS
    ]
    if crossmodal_retention or claim_pair or temporal_curriculum:
        paths.append(
            (
                "s4-crossmodal-retention",
                root / "reports/s4-crossmodal-retention-training-001.json",
                root / "artifacts/s4-crossmodal-retention-train-001",
                root / "artifacts/s4-crossmodal-retention-pilot-002",
            )
        )
    if claim_pair or temporal_curriculum:
        paths.extend(
            (
                "s4-claim-pair-" + arm,
                root / "reports" / ("s4-claim-pair-" + arm + "-training-001.json"),
                root / "artifacts" / ("s4-claim-pair-" + arm + "-train-001"),
                root / "artifacts/s4-claim-pair-pilot-001",
            )
            for arm in ("ce0", "pair01")
        )
    if temporal_curriculum:
        paths.append(
            (
                "s4-temporal-curriculum",
                root / "reports/s4-temporal-curriculum-training-001.json",
                root / "artifacts/s4-temporal-curriculum-train-001",
                root / "artifacts/s4-temporal-curriculum-pilot-001",
            )
        )
    return paths


def temporal_preparation_passed(report: dict[str, Any]) -> None:
    require(
        report.get("status") == "ready_for_preflight"
        and report.get("exit_code") == 0
        and report.get("data_ready_for_preflight") is True
        and report.get("inventory_isolation") == "isolated"
        and report.get("source_admission") == "verified_existing_scope"
        and report.get("total_draws") == 120,
        "temporal_preparation_not_passed",
    )


def preparation_passed(report: dict[str, Any], *, claim_pair: bool = False) -> None:
    if claim_pair:
        require(
            report.get("status") == "prepared_pending_pair_hashes"
            and report.get("exit_code") == 0
            and report.get("expanded_isolation") == "isolated"
            and report.get("pilot_isolation") == "isolated"
            and report.get("draws") == 120
            and report.get("pair_count") == 20,
            "claim_preparation_not_passed",
        )
    else:
        require(
            report.get("status") == "prepared" and report.get("exit_code") == 0,
            "preparation_not_passed",
        )


def claim_terminal(report: dict[str, Any], name: str) -> None:
    terminal_report(report)
    require(report["steps_completed"] <= 10, "claim_steps_out_of_budget")
    require(
        report["status"] != "completed" or report["steps_completed"] == 10, "claim_completed_short"
    )
    require(
        report["status"] != "blocked" or report["steps_completed"] == 0, "claim_blocked_updates"
    )
    require(report["status"] != "failed" or report["exit_code"] != 0, "claim_failed_exit_code")
    config = report.get("config", {})
    require(
        all(
            type(config.get(k)) is int and config[k] == v
            for k, v in {"seed": 42, "steps": 10, "accumulation": 12, "sample_count": 120}.items()
        ),
        "claim_draw_config",
    )
    expected_weight = 0.1 if name.endswith("pair01") else 0.0
    require(config.get("pair_loss_weight", 0.0) == expected_weight, "claim_arm_weight")


def base_inventory(
    root: Path,
    *,
    crossmodal_retention: bool,
    claim_pair: bool = False,
    temporal_curriculum: bool = False,
) -> tuple[Path, dict[str, str] | None]:
    if temporal_curriculum:
        exposure = root / "artifacts/s4-pilot-exposure-004"
        report, binding = read_json(exposure / "report.json")
        require(
            report.get("status") == "known_pilot_exposures_reconciled"
            and report.get("exit_code") == 0
            and report.get("unknown_update_bearing_inputs") == 0,
            "exposure004_not_passed",
        )
        path = exposure / "inventory-manifest.json"
        require(
            sha(path) == report["output_sha256"]["inventory-manifest.json"],
            "base_inventory_preparation_sha",
        )
        return path, binding
    if claim_pair:
        preparation = root / "artifacts/s4-claim-pair-pilot-001"
        report, binding = read_json(preparation / "report.json")
        preparation_passed(report, claim_pair=True)
        path = preparation / "inventory-manifest.json"
        require(
            sha(path) == report["output_sha256"]["inventory-manifest.json"],
            "base_inventory_preparation_sha",
        )
        return path, binding
    if not crossmodal_retention:
        return root / "artifacts/s4-video-pilot-001/inventory-manifest.json", None
    preparation = root / "artifacts/s4-crossmodal-retention-pilot-002"
    report, binding = read_json(preparation / "report.json")
    require(
        report.get("status") == "prepared" and report.get("exit_code") == 0,
        "preparation_not_passed",
    )
    path = preparation / "inventory-manifest.json"
    require(
        sha(path) == report["output_sha256"]["inventory-manifest.json"],
        "base_inventory_preparation_sha",
    )
    return path, binding


def prepare(
    root: Path,
    legacy_root: Path,
    output: Path,
    historical_index: Path,
    *,
    crossmodal_retention: bool = False,
    claim_pair: bool = False,
    temporal_curriculum: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    require(
        all(p.is_absolute() for p in (root, legacy_root, output, historical_index)),
        "absolute_paths",
    )
    require(not output.exists(), "output_exists")
    output.mkdir(parents=True, mode=0o700)
    try:
        inventory_path, base_preparation_binding = base_inventory(
            root,
            crossmodal_retention=crossmodal_retention,
            claim_pair=claim_pair,
            temporal_curriculum=temporal_curriculum,
        )
        raw_inventory, inventory_binding = read_json(inventory_path)
        inventory = Manifest.model_validate(raw_inventory)
        mappings = version_mapping(inventory)
        combined = inventory.model_dump(mode="json", exclude_none=True)
        hashes = {a.sha256 for a in inventory.samples if a.expected_pool == "train"}
        all_hashes = {a.sha256 for a in inventory.samples}
        bindings: dict[str, str] = {inventory_binding["path"]: inventory_binding["sha256"]}
        if base_preparation_binding is not None:
            bindings[base_preparation_binding["path"]] = base_preparation_binding["sha256"]
        pilots, new_artifacts = [], []
        for name, final_path, run_dir, preparation_dir in pilot_paths(
            root,
            crossmodal_retention=crossmodal_retention,
            claim_pair=claim_pair,
            temporal_curriculum=temporal_curriculum,
        ):
            final, final_binding = read_json(final_path)
            runtime_report, run_binding = read_json(run_dir / "run.json")
            terminal_report(final)
            terminal_report(runtime_report)
            is_claim = (claim_pair or temporal_curriculum) and name.startswith("s4-claim-pair-")
            is_temporal = temporal_curriculum and name == "s4-temporal-curriculum"
            if is_claim or is_temporal:
                claim_terminal(final, name)
                claim_terminal(runtime_report, name)
                require(
                    final.get("config") == runtime_report.get("config"),
                    "claim_final_config_mismatch",
                )
            for key in ("status", "steps_completed", "exit_code", "data_sha256"):
                require(final.get(key) == runtime_report.get(key), "final_run_report_mismatch")
            runtime_data = run_dir / "data.jsonl"
            runtime_binding = bind(runtime_data)
            require(runtime_binding["sha256"] == final.get("data_sha256"), "runtime_data_sha")
            prep_report, prep_report_binding = read_json(preparation_dir / "report.json")
            if is_temporal:
                temporal_preparation_passed(prep_report)
                for filename in (
                    "allocation-ledger.jsonl",
                    "config-ce.json",
                    "draw-recipe.json",
                    "inventory-manifest.json",
                    "inventory-audit.json",
                ):
                    b = bind(preparation_dir / filename)
                    require(
                        b["sha256"] == prep_report["output_sha256"][filename],
                        "temporal_preparation_output_sha",
                    )
                    require(
                        b["path"] not in bindings or bindings[b["path"]] == b["sha256"],
                        "mutable_report",
                    )
                    bindings[b["path"]] = b["sha256"]
                require(
                    final.get("init_adapter")
                    == runtime_report.get("init_adapter")
                    == prep_report["init_adapter"],
                    "temporal_init_binding",
                )
            else:
                preparation_passed(prep_report, claim_pair=is_claim)
            preparation_data = preparation_dir / "train.jsonl"
            preparation_binding = bind(preparation_data)
            expected = prep_report["output_sha256"].get(
                "train", prep_report["output_sha256"].get("train.jsonl")
            )
            require(preparation_binding["sha256"] == expected, "preparation_data_sha")
            comparison = equivalent(runtime_data, preparation_data)
            if is_claim or is_temporal:
                require(comparison["rows"] == 120, "claim_runtime_count")
            require(
                all(version in mappings for version in comparison["data_versions"]),
                "unmapped_data_version",
            )
            artifact = {
                **runtime_binding,
                "cohorts_by_data_version": {v: mappings[v] for v in comparison["data_versions"]},
                "exposure": "historical",
                "expected_pool": "train",
            }
            appended = False
            if final["steps_completed"] > 0 and runtime_binding["sha256"] not in hashes:
                require(
                    runtime_binding["sha256"] not in all_hashes, "canonical_sha_nontrain_collision"
                )
                combined["samples"].append(artifact)
                new_artifacts.append(artifact)
                hashes.add(runtime_binding["sha256"])
                all_hashes.add(runtime_binding["sha256"])
                appended = True
            for binding in (
                final_binding,
                run_binding,
                runtime_binding,
                prep_report_binding,
                preparation_binding,
            ):
                require(
                    binding["path"] not in bindings
                    or bindings[binding["path"]] == binding["sha256"],
                    "mutable_report",
                )
                bindings[binding["path"]] = binding["sha256"]
            pilots.append(
                {
                    "name": name,
                    "status": final["status"],
                    "steps_completed": final["steps_completed"],
                    "update_bearing": final["steps_completed"] > 0,
                    "appended_historical_train": appended,
                    "final_report": final_binding,
                    "run_report": run_binding,
                    "runtime_data": runtime_binding,
                    "preparation_data": preparation_binding,
                    "preparation_report": prep_report_binding,
                    "canonical_equivalence": comparison,
                }
            )
        index, index_binding = read_json(historical_index)
        require(
            index.get("status") == "indexed" and not index.get("unresolved"), "old_index_incomplete"
        )
        bindings[index_binding["path"]] = index_binding["sha256"]
        known_paths: set[Path] = set()
        indexed_count = 0
        for item in index["files"]:
            require(item["sha256"] in hashes, "indexed_input_missing_inventory")
            data_path = Path(item["path"])
            require(sha(data_path) == item["sha256"], "indexed_input_sha")
            bindings[str(data_path)] = item["sha256"]
            indexed_count += 1
            for report_binding in item["report_bindings"]:
                path = Path(report_binding["path"])
                report, current = read_json(path)
                require(current["sha256"] == report_binding["sha256"], "old_report_mutated")
                require(report.get("data_sha256") == item["sha256"], "old_report_input_binding")
                require(
                    str(path) not in bindings or bindings[str(path)] == current["sha256"],
                    "mutable_report",
                )
                bindings[str(path)] = current["sha256"]
                known_paths.add(path)
        discovered = (
            known_paths
            | set((root / "reports").glob("*.json"))
            | set((root / "artifacts").rglob("run.json"))
            | set((legacy_root / "artifacts/training").rglob("run.json"))
        )
        update_reports = []
        unknown = []
        for path in sorted(discovered):
            try:
                report, report_binding = read_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                require(
                    path.name != "run.json" and path not in known_paths,
                    "unreadable_training_report",
                )
                continue
            if (
                not isinstance(report, dict)
                or type(report.get("steps_completed")) is not int
                or report["steps_completed"] <= 0
            ):
                continue
            terminal_report(report)
            data_sha = report.get("data_sha256")
            require(
                isinstance(data_sha, str) and len(data_sha) == 64, "update_report_without_data_sha"
            )
            require(
                str(path) not in bindings or bindings[str(path)] == report_binding["sha256"],
                "mutable_report",
            )
            bindings[str(path)] = report_binding["sha256"]
            entry = {
                "report": report_binding,
                "data_sha256": data_sha,
                "steps_completed": report["steps_completed"],
                "status": report["status"],
            }
            update_reports.append(entry)
            if data_sha not in hashes:
                unknown.append(entry)
        write(
            output / "discovered-update-reports.json",
            {"update_reports": update_reports, "unknown": unknown},
        )
        require(not unknown, "unknown_update_bearing_input")
        write(output / "inventory-manifest.json", combined)
        audit = audit_manifest(output / "inventory-manifest.json")
        write(output / "inventory-audit.json", audit.model_dump(mode="json"))
        require(audit.outcome == "isolated", "full_inventory_rejected")
        # Final reports and data must still match the initial snapshots after discovery/audit.
        require(
            all(sha(Path(path)) == expected for path, expected in bindings.items()),
            "mutable_report_or_input",
        )
        report = {
            "status": "known_pilot_exposures_reconciled",
            "exit_code": 0,
            "scope": (
                "Known saved update-bearing report discovery; not all external jobs, "
                "gold readmission or S4 acceptance"
            ),
            "gpu_used": False,
            "base_inventory": inventory_binding,
            "historical_index": index_binding,
            "pilots": pilots,
            "appended_artifacts": new_artifacts,
            "indexed_input_files": indexed_count,
            "discovered_update_bearing_reports": len(update_reports),
            "unknown_update_bearing_inputs": 0,
            "all_bound_input_sha256": bindings,
            "output_sha256": {
                name: sha(output / name)
                for name in (
                    "inventory-manifest.json",
                    "inventory-audit.json",
                    "discovered-update-reports.json",
                )
            },
            "duration_seconds": time.monotonic() - started,
            "runner_sha256": sha(Path(__file__)),
            "cost": None,
        }
        if crossmodal_retention or claim_pair or temporal_curriculum:
            report.update(
                base_preparation_report=base_preparation_binding, crossmodal_retention_included=True
            )
        if claim_pair or temporal_curriculum:
            report["claim_pair_included"] = True
        if temporal_curriculum:
            report["temporal_curriculum_included"] = True
            report["base_exposure_report"] = base_preparation_binding
        write(output / "report.json", report)
        return report
    except Exception as exc:
        if not (output / "report.json").exists():
            allowed = {
                "nonterminal_report",
                "old_report_mutated",
                "mutable_report",
                "mutable_report_or_input",
                "canonical_sample_mismatch",
                "unknown_update_bearing_input",
                "full_inventory_rejected",
            }
            category = (
                str(exc)
                if isinstance(exc, ValueError) and str(exc) in allowed
                else "exposure_reconciliation_rejected"
            )
            write(
                output / "report.json",
                {
                    "status": "rejected",
                    "exit_code": 1,
                    "categories": [category],
                    "gpu_used": False,
                    "duration_seconds": time.monotonic() - started,
                    "cost": None,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-index", type=Path)
    parser.add_argument(
        "--crossmodal-retention",
        action="store_true",
        help="Bind prepare002 inventory and include the terminal 360-draw candidate",
    )
    parser.add_argument(
        "--claim-pair",
        action="store_true",
        help="Bind claim pilot inventory and both terminal 10-step arms; includes older pilots",
    )
    parser.add_argument(
        "--temporal-curriculum",
        action="store_true",
        help="Use terminal exposure004 base and include temporal training plus all older pilots",
    )
    args = parser.parse_args()
    try:
        report = prepare(
            args.root,
            args.legacy_root,
            args.output_dir,
            args.historical_index
            or args.root / "artifacts/s4-gold-inventory-001/historical-training-index.json",
            crossmodal_retention=args.crossmodal_retention,
            claim_pair=args.claim_pair,
            temporal_curriculum=args.temporal_curriculum,
        )
    except Exception:
        print(json.dumps({"status": "rejected", "exit_code": 1}))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "appended_artifacts": len(report["appended_artifacts"]),
                "update_bearing_reports": report["discovered_update_bearing_reports"],
                "exit_code": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
