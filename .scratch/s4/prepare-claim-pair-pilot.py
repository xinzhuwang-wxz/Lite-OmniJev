"""CPU inventory isolation and fixed 120-draw claim-pair pilot preparation."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines, digest
from scripts.gold_media_bundle import private_write, real_path
from scripts.s1_text_pools import canonical_sample_sha
from scripts.train_s1 import verify_data_is_train_pool
from train.trainer import LoopConfig, data_bytes, smoke_samples, verified_init_adapter

INIT_SHA = "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818"
PAIR_VERSION = "crossmodal-controlled-train-pairs-v1"


class PreparationRejected(ValueError):
    pass


def require(value: bool, why: str) -> None:
    if not value:
        raise PreparationRejected(why)


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    require(not path.exists(), "output_exists")
    private_write(
        path, value if isinstance(value, str) else json.dumps(value, sort_keys=True, indent=2)
    )


def bound_outputs(root: Path, receipt: dict[str, Any], names: list[str]) -> None:
    for name in names:
        require(sha(root / name) == receipt["output_sha256"][name], "source_output_sha")


def inverse_order(desired: list[Any]) -> list[Any]:
    require(len(desired) == 120, "draw_count")
    order = list(range(120))
    random.Random(42).shuffle(order)
    arranged = list(desired)
    for draw, index in enumerate(order):
        arranged[index] = desired[draw]
    require([arranged[i] for i in order] == desired, "inverse_order")
    return arranged


def exact_ledger(samples: list[Sample], ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for s in samples:
        matches = [
            a
            for a in ledger
            if a["item_id"] == s.id and a["derived_sample_sha256"] == canonical_sample_sha(s)
        ]
        require(bool(matches) and all(a["pool"] == "train" for a in matches), "exact_source_ledger")
        require(all(a == matches[0] for a in matches), "conflicting_source_ledger")
        selected.append(matches[0])
    return selected


def cohorts(samples: list[Sample], inventory: Manifest) -> dict[str, str]:
    wanted = {(s.id, s.meta.data_version): canonical_sample_sha(s) for s in samples}
    require(len(wanted) == len(samples), "duplicate_id_version")
    matched = set()
    result: dict[str, str] = {}
    for artifact in inventory.samples:
        if artifact.expected_pool != "train":
            continue
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            key = (sample.id, sample.meta.data_version)
            if key not in wanted:
                continue
            require(canonical_sample_sha(sample) == wanted[key], "source_sample_mismatch")
            cohort = artifact.cohort or (artifact.cohorts_by_data_version or {}).get(key[1])
            require(cohort is not None and result.get(key[1], cohort) == cohort, "cohort_conflict")
            assert cohort is not None
            result[key[1]] = cohort
            matched.add(key)
    require(matched == wanted.keys(), "unmapped_exact_source")
    return result


def prepare(
    root: Path, out: Path, pair_report_sha: str, exposure_report_sha: str, coverage_report_sha: str
) -> dict[str, Any]:
    started = time.monotonic()
    for path in (root, out):
        real_path(path)
    require(not out.exists(), "output_exists")
    pairroot = root / "artifacts/s4-crossmodal-pairs-001"
    pairreport = root / "reports/s4-crossmodal-pairs-build-001.json"
    exposure = root / "artifacts/s4-pilot-exposure-003"
    coverage = root / "artifacts/s4-crossmodal-retention-pilot-002"
    for path, expected in [
        (pairreport, pair_report_sha),
        (exposure / "report.json", exposure_report_sha),
        (coverage / "report.json", coverage_report_sha),
    ]:
        require(sha(path) == expected, "receipt_sha")
    pr, er, cr = read(pairreport), read(exposure / "report.json"), read(coverage / "report.json")
    require(
        pr["status"] == "new_prepared_not_admitted"
        and pr["count"] == 600
        and pr["group_count"] == 200,
        "pair_source_status",
    )
    require(
        er["status"] == "known_pilot_exposures_reconciled"
        and er["exit_code"] == 0
        and er["unknown_update_bearing_inputs"] == 0,
        "exposure_status",
    )
    bound_outputs(
        pairroot,
        {"output_sha256": {k + ".jsonl": v for k, v in pr["output_sha256"].items()}},
        ["train.jsonl", "components.jsonl", "allocation.jsonl", "groups.jsonl"],
    )
    bound_outputs(exposure, er, ["inventory-manifest.json"])
    require(cr["status"] == "prepared" and cr["exit_code"] == 0, "coverage_status")
    bound_outputs(coverage, cr, ["train.jsonl", "allocation-ledger.jsonl", "config-30.json"])
    inventory = Manifest.model_validate_json((exposure / "inventory-manifest.json").read_bytes())
    paired = [Sample.model_validate(r) for r in rows(pairroot / "train.jsonl")]
    require(
        len({s.id for s in paired}) == 600
        and all(s.meta.pool == "train" and s.meta.data_version == PAIR_VERSION for s in paired),
        "paired_identity",
    )
    pair_by_id = {s.id: s for s in paired}
    exact_ledger(paired, rows(pairroot / "allocation.jsonl"))
    # New IDs must not shadow any existing TRAIN source, even historical versions.
    for artifact in inventory.samples:
        if artifact.expected_pool == "train":
            require(
                not any(
                    Sample.model_validate_json(line).id in pair_by_id
                    for line in bound_lines(artifact)
                ),
                "new_id_overwrites_history",
            )
    expanded = inventory.model_dump(mode="json", exclude_none=True)
    expanded["samples"].append(
        {
            "path": str(pairroot / "train.jsonl"),
            "sha256": sha(pairroot / "train.jsonl"),
            "cohorts_by_data_version": {PAIR_VERSION: "crossmodal"},
            "exposure": "current",
            "expected_pool": "train",
        }
    )
    expanded["component_ledgers"].append(
        {"path": str(pairroot / "components.jsonl"), "sha256": sha(pairroot / "components.jsonl")}
    )
    out.mkdir(mode=0o700)
    write(out / "inventory-expanded.json", expanded)
    audit = audit_manifest(out / "inventory-expanded.json")
    write(out / "inventory-expanded-audit.json", audit.model_dump(mode="json"))
    require(audit.outcome == "isolated", "expanded_isolation")
    replay = [Sample.model_validate(r) for r in rows(coverage / "train.jsonl")]
    require(
        all(s.meta.pool == "train" for s in replay) and len({s.id for s in replay}) == 360,
        "coverage_train",
    )
    mapping = cohorts(replay, inventory)
    buckets = {
        c: sorted(
            [s for s in replay if mapping[s.meta.data_version] == c],
            key=lambda s: digest(("s4-claim-replay:42:" + s.id).encode()),
        )
        for c in ["video", "audio", "image", "screenshot", "text"]
    }
    require(
        all(len(v) >= (40 if k == "video" else 10) for k, v in buckets.items()), "replay_capacity"
    )
    groups = sorted(
        rows(pairroot / "groups.jsonl"),
        key=lambda g: digest(("s4-claim-pair:42:" + g["group_id"]).encode()),
    )
    require(len(groups) == 200 and len({g["group_id"] for g in groups}) == 200, "group_coverage")
    components = {c["item_id"]: c for c in rows(pairroot / "components.jsonl")}
    selected, pair_recipe = [], []
    for n, g in enumerate(groups[:20]):
        rule = "image_false_audio_true" if n % 2 == 0 else "image_true_audio_false"
        positive, negative = (pair_by_id[g["variants"][r]] for r in ["both_true", rule])
        pc, nc = components[positive.id], components[negative.id]
        shared = "audio" if rule == "image_false_audio_true" else "image"
        require(
            pc[shared] == nc[shared]
            and pc["program_truth"] == "yes"
            and nc["program_truth"] == "no",
            "controlled_pair",
        )
        require(
            positive.meta.family_id == negative.meta.family_id
            and positive.questions[0].instruction != negative.questions[0].instruction,
            "claim_semantics",
        )
        selected.extend([positive, negative])
        pair_recipe.append(
            {
                "group_id": g["group_id"],
                "negative_rule": rule,
                "positive_id": positive.id,
                "negative_id": negative.id,
                "positive_draw": (n // 2) * 12 + (n % 2) * 2,
                "negative_draw": (n // 2) * 12 + (n % 2) * 2 + 1,
                "shared_modality": shared,
                "expected_components": {"positive": pc, "negative": nc},
            }
        )
    desired = []
    for step in range(10):
        desired.extend(selected[step * 4 : step * 4 + 4])
        desired.extend(buckets["video"][step * 4 : step * 4 + 4])
        desired.extend(buckets[c][step] for c in ["audio", "image", "screenshot", "text"])
    require(len({s.id for s in desired}) == 120, "duplicate_draw")
    arranged = inverse_order(desired)
    write(out / "train.jsonl", data_bytes(arranged).decode())
    ledger = exact_ledger(
        arranged, rows(coverage / "allocation-ledger.jsonl") + rows(pairroot / "allocation.jsonl")
    )
    write(
        out / "allocation-ledger.jsonl",
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in ledger),
    )
    verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    final = json.loads(json.dumps(expanded))
    selected_ids = {s.id for s in arranged}
    for a in final["samples"]:
        if a["expected_pool"] == "train" and a["exposure"] == "current":
            if any(
                json.loads(line)["id"] in selected_ids
                for line in bound_lines(
                    next(
                        x
                        for x in Manifest.model_validate(final).samples
                        if x.path == a["path"] and x.sha256 == a["sha256"]
                    )
                )
            ):
                a["exposure"] = "historical"
    versions = {
        s.meta.data_version: (
            "crossmodal" if s.meta.data_version == PAIR_VERSION else mapping[s.meta.data_version]
        )
        for s in arranged
    }
    final["samples"].append(
        {
            "path": str(out / "train.jsonl"),
            "sha256": sha(out / "train.jsonl"),
            "cohorts_by_data_version": versions,
            "exposure": "current",
            "expected_pool": "train",
        }
    )
    write(out / "inventory-manifest.json", final)
    final_audit = audit_manifest(out / "inventory-manifest.json")
    write(out / "inventory-audit.json", final_audit.model_dump(mode="json"))
    require(final_audit.outcome == "isolated", "pilot_isolation")
    oldconfig = read(coverage / "config-30.json")
    config = {
        **oldconfig,
        "seed": 42,
        "steps": 10,
        "sample_count": 120,
        "accumulation": 12,
        "midway_step": 5,
        "checkpoint_every": 5,
        "expected_data_versions": sorted(versions),
        "media_roots": [*oldconfig["media_roots"], str(pairroot)],
    }
    require(not oldconfig.get("reference_kl_weight", 0), "reference_off")
    validated = LoopConfig.model_validate(config)
    smoke_samples(
        validated, out / "train.jsonl", expected_data_versions=config["expected_data_versions"]
    )
    write(out / "config-ce0.json", config)
    # This is a recipe, deliberately NOT a LoopConfig with invented sidecar hashes.
    write(
        out / "pair-override-recipe.json",
        {
            "status": "prepared_pending_pair_hashes",
            "pair_loss_weight": 0.1,
            "schema": "train-claim-pairs-v1",
            "required_pending": [
                "pair_sidecar",
                "pair_sidecar_sha256",
                "pair_ledger",
                "pair_ledger_sha256",
            ],
            "data_sha256": sha(out / "train.jsonl"),
            "ledger_sha256": sha(out / "allocation-ledger.jsonl"),
            "seed": 42,
            "steps": 10,
            "accumulation": 12,
        },
    )
    order = list(range(120))
    random.Random(42).shuffle(order)
    recipe = {
        "schema_version": "train-claim-pairs-v1",
        "data_sha256": sha(out / "train.jsonl"),
        "ledger_sha256": sha(out / "allocation-ledger.jsonl"),
        "seed": 42,
        "steps": 10,
        "accumulation": 12,
        "pairs": pair_recipe,
        "draws": [
            {
                "draw": i,
                "step": i // 12,
                "offset": i % 12,
                "data_index": order[i],
                "item_id": s.id,
                "data_version": s.meta.data_version,
                "sample_sha256": canonical_sample_sha(s),
                "cohort": versions[s.meta.data_version],
            }
            for i, s in enumerate(desired)
        ],
    }
    write(out / "draw-recipe.json", recipe)
    adapter = verified_init_adapter(
        root / "artifacts/s4-crossmodal-retention-train-001/adapter", INIT_SHA
    )
    return {
        "status": "prepared_pending_pair_hashes",
        "exit_code": 0,
        "gpu_used": False,
        "training_started": False,
        "expanded_isolation": audit.outcome,
        "pilot_isolation": final_audit.outcome,
        "draws": 120,
        "pair_count": 20,
        "cohorts": dict(Counter(versions[s.meta.data_version] for s in desired)),
        "negative_rules": dict(Counter(p["negative_rule"] for p in pair_recipe)),
        "init_adapter": adapter,
        "input_report_sha256": {
            "pairs": pair_report_sha,
            "exposure": exposure_report_sha,
            "coverage": coverage_report_sha,
        },
        "output_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        "config_changes": {
            k: {"old": oldconfig.get(k), "new": v}
            for k, v in config.items()
            if oldconfig.get(k) != v
        },
        "duration_seconds": time.monotonic() - started,
        "cost": None,
        "python": platform.python_version(),
        "script_sha256": sha(Path(__file__)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    for name in ["pair", "exposure", "coverage"]:
        p.add_argument("--" + name + "-report-sha256", required=True)
    a = p.parse_args()
    try:
        r = prepare(
            a.root,
            a.output_dir,
            a.pair_report_sha256,
            a.exposure_report_sha256,
            a.coverage_report_sha256,
        )
        write(a.output_dir / "report.json", r)
        return 0
    except Exception as exc:
        # No exception text or sample content escapes. Preserve any partial outputs.
        failure = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, PreparationRejected) else "private_exception",
            "exit_code": 1,
        }
        if a.output_dir.is_dir() and not (a.output_dir / "report.json").exists():
            write(a.output_dir / "report.json", failure)
        print(json.dumps(failure))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
