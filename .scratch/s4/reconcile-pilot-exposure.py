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
from train.trainer import LoopConfig

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


def terminal_report(report: dict[str, Any], *, allow_paused: bool = False) -> None:
    require(
        report.get("status") in TERMINAL | ({"paused"} if allow_paused else set()),
        "nonterminal_report",
    )
    require(
        type(report.get("steps_completed")) is int and report["steps_completed"] >= 0,
        "invalid_steps",
    )
    require(type(report.get("exit_code")) is int, "missing_final_exit_code")
    require(
        report["status"] not in {"completed", "paused"} or report["exit_code"] == 0,
        "inconsistent_final_status",
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
    temporal_brier: bool = False,
    learnability: bool = False,
    temporal_coverage: bool = False,
    coverage_segment002: bool = False,
    polarity_arm: str | None = None,
    coverage_segment003: bool = False,
    temporal_order_contrast: bool = False,
) -> list[tuple[str, Path, Path, Path]]:
    if polarity_arm is not None:
        require(polarity_arm in {"control", "polarity"}, "polarity_single_arm")
        coverage_segment002 = True
    if temporal_order_contrast:
        coverage_segment003 = True
    if coverage_segment003:
        coverage_segment002 = True
    paths = [
        (
            name,
            root / "reports" / (name + "-train-001.json"),
            root / "artifacts" / (name + "-train-001"),
            root / "artifacts" / (name + "-001"),
        )
        for name in PILOTS
    ]
    if (
        crossmodal_retention
        or claim_pair
        or temporal_curriculum
        or temporal_brier
        or learnability
        or temporal_coverage
        or coverage_segment002
    ):
        paths.append(
            (
                "s4-crossmodal-retention",
                root / "reports/s4-crossmodal-retention-training-001.json",
                root / "artifacts/s4-crossmodal-retention-train-001",
                root / "artifacts/s4-crossmodal-retention-pilot-002",
            )
        )
    if (
        claim_pair
        or temporal_curriculum
        or temporal_brier
        or learnability
        or temporal_coverage
        or coverage_segment002
    ):
        paths.extend(
            (
                "s4-claim-pair-" + arm,
                root / "reports" / ("s4-claim-pair-" + arm + "-training-001.json"),
                root / "artifacts" / ("s4-claim-pair-" + arm + "-train-001"),
                root / "artifacts/s4-claim-pair-pilot-001",
            )
            for arm in ("ce0", "pair01")
        )
    if (
        temporal_curriculum
        or temporal_brier
        or learnability
        or temporal_coverage
        or coverage_segment002
    ):
        paths.append(
            (
                "s4-temporal-curriculum",
                root / "reports/s4-temporal-curriculum-training-001.json",
                root / "artifacts/s4-temporal-curriculum-train-001",
                root / "artifacts/s4-temporal-curriculum-pilot-001",
            )
        )
    if temporal_brier or learnability or temporal_coverage or coverage_segment002:
        paths.append(
            (
                "s4-temporal-brier",
                root / "reports/s4-temporal-brier-training-001.json",
                root / "artifacts/s4-temporal-brier-train-001",
                root / "artifacts/s4-temporal-curriculum-pilot-001",
            )
        )
    if learnability or temporal_coverage or coverage_segment002:
        paths.append(
            (
                "s4-train-learnability",
                root / "reports/s4-train-learnability-training-001.json",
                root / "artifacts/s4-train-learnability-train-001",
                root / "artifacts/s4-train-learnability-001",
            )
        )
    if temporal_coverage or coverage_segment002:
        paths.append(
            (
                "s4-temporal-coverage",
                root / "reports/s4-temporal-coverage-training-001.json",
                root / "artifacts/s4-temporal-coverage-train-001",
                root / "artifacts/s4-temporal-coverage-pilot-001",
            )
        )
    if coverage_segment002:
        paths.append(
            (
                "s4-temporal-coverage-segment002",
                root / "reports/s4-temporal-coverage-training-002.json",
                root / "artifacts/s4-temporal-coverage-train-002",
                root / "artifacts/s4-temporal-coverage-pilot-002",
            )
        )
    if polarity_arm is not None:
        name = "s4-video-polarity-" + polarity_arm
        paths.append(
            (
                name,
                root / "reports" / (name + "-training-001.json"),
                root / "artifacts" / (name + "-train-001"),
                root / "artifacts/s4-video-polarity-pilot-001" / polarity_arm,
            )
        )
    if coverage_segment003:
        paths.append(
            (
                "s4-temporal-coverage-segment003",
                root / "reports/s4-temporal-coverage-training-003.json",
                root / "artifacts/s4-temporal-coverage-train-003",
                root / "artifacts/s4-temporal-coverage-pilot-003",
            )
        )
    if temporal_order_contrast:
        paths.append(
            (
                "s4-temporal-order-contrast",
                root / "reports/s4-temporal-order-contrast-training-001.json",
                root / "artifacts/s4-temporal-order-contrast-train-001",
                root / "artifacts/s4-temporal-order-contrast-pilot-001",
            )
        )
    return paths


def learnability_terminal(report: dict[str, Any]) -> None:
    terminal_report(report)
    require(
        report["steps_completed"] <= 10
        and (report["status"] != "completed" or report["steps_completed"] == 10),
        "learnability_step_budget",
    )
    config = report.get("config", {})
    require(
        all(
            type(config.get(k)) is int and config[k] == v
            for k, v in {"seed": 42, "steps": 10, "accumulation": 12, "sample_count": 12}.items()
        ),
        "learnability_repeated_draw_config",
    )
    require(
        config.get("brier_weight", 0) == 0
        and config.get("pair_loss_weight", 0) == 0
        and config.get("reference_kl_weight", 0) == 0,
        "learnability_ce_only",
    )
    require(
        report["status"] != "blocked" or report["steps_completed"] == 0,
        "learnability_blocked_updates",
    )


def learnability_subset(prepared: Path, source: Path) -> dict[str, int]:
    subset, all_rows = canonical_rows(prepared), canonical_rows(source)
    require(len(subset) == 12 and all_rows.keys() >= subset.keys(), "learnability_subset_count")
    require(
        all(row == all_rows[ident] for ident, row in subset.items()),
        "learnability_full_sample_subset",
    )
    require(
        len({r["meta"]["family_id"] for r in subset.values()}) == 12, "learnability_unique_families"
    )
    require(
        all(
            r["meta"]["modality"] == ["video"]
            and len(r["questions"]) == 1
            and r["questions"][0]["target"]["kind"] == "hard"
            for r in subset.values()
        ),
        "learnability_video_gold",
    )
    targets = [r["questions"][0]["target"]["value"] for r in subset.values()]
    require(targets.count("yes") == targets.count("no") == 6, "learnability_gold_balance")
    return {"unique_samples": 12, "unique_families": 12, "yes": 6, "no": 6, "scheduled_draws": 120}


POLARITY_DATA = {
    "control": "f7cbd1a5ee86c92cb13fc9377d9942b4e6907e0182d685431103b4d20ddc85bc",
    "polarity": "0235ff3f039c28229c897bf79a5f353958d34d4bda486035cc72a21df691cb84",
}


def arm_current_inventory(combined: dict[str, Any], artifact: dict[str, Any]) -> None:
    require(
        artifact["expected_pool"] == "train" and artifact["exposure"] == "current",
        "polarity_current_train",
    )
    # Historical inventory is not a claim that every row was actually drawn.
    for prior in combined["samples"]:
        if prior["expected_pool"] == "train":
            prior["exposure"] = "historical"
    combined["samples"].append(artifact)


def polarity_terminal_binding(arm: str, launch: dict[str, Any], final: dict[str, Any]) -> None:
    require(arm in POLARITY_DATA and launch.get("arm") == arm, "polarity_single_arm")
    claim_terminal(final, "s4-video-polarity-" + arm)
    require(
        launch.get("status") in {"completed", "failed"} and type(launch.get("exit_code")) is int,
        "polarity_launcher_nonterminal",
    )
    require(
        launch["selected_init_adapter"] == final.get("init_adapter")
        and final["init_adapter"]["weights_sha256"]
        == "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
        "polarity_actual_parent",
    )
    require(final.get("data_sha256") == POLARITY_DATA[arm], "polarity_actual_data")
    c = dict(final["config"])
    if "init_adapter" in c:
        require(c.pop("init_adapter") == launch["selected_init_adapter"], "polarity_config_init")
    parsed = LoopConfig.model_validate(c)
    require(
        parsed.reference_kl_weight == parsed.pair_loss_weight == parsed.brier_weight == 0
        and parsed.midway_step == parsed.checkpoint_every == 5,
        "polarity_ce_contract",
    )
    require(
        launch.get("training_limit_seconds") == 240 and launch.get("dev50_limit_seconds") == 180,
        "polarity_launcher_budget",
    )


def polarity_bindings(
    root: Path,
    prepared: Path,
    prep: dict[str, Any],
    final: dict[str, Any],
    runtime: dict[str, Any],
    arm: str | None,
) -> tuple[dict[str, str], dict[str, str]]:
    require(arm in POLARITY_DATA, "polarity_single_arm")
    assert arm is not None
    launch_path = root / "reports" / ("s4-video-polarity-" + arm + "-launch-001.json")
    launch, lb = read_json(launch_path)
    polarity_terminal_binding(arm, launch, final)
    polarity_terminal_binding(arm, launch, runtime)
    require(
        prep.get("status") == "cpu_prepared_pending_root_training_decision"
        and prep.get("exit_code") == 0
        and prep.get("training_authorized") is False
        and prep.get("families") == 30
        and prep["arms"][arm]["inventory_isolation"] == "isolated",
        "polarity_prepared_isolation",
    )
    require(
        sha(prepared.parent / "report.json") == launch["prepared_report_sha256"],
        "polarity_preparation_receipt",
    )
    bindings = {lb["path"]: lb["sha256"]}
    for name, expected in prep["arms"][arm]["output_sha256"].items():
        require(Path(name).name == name, "polarity_safe_filename")
        b = bind(prepared / name)
        require(b["sha256"] == expected, "polarity_preparation_output")
        bindings[b["path"]] = b["sha256"]
    audit = json.loads((prepared / "inventory-audit.json").read_bytes())
    require(
        audit["outcome"] == "isolated"
        and audit["manifest_sha256"] == sha(prepared / "inventory-manifest.json"),
        "polarity_bound_isolation",
    )
    for path, expected in launch["bound_inputs_sha256"].items():
        require(sha(Path(path)) == expected, "polarity_launcher_input")
        bindings[path] = expected
    runner = root.parent / "s4-video-polarity-launch-code-002/launch-video-polarity-pilot.py"
    require(sha(runner) == launch["launcher_sha256"], "polarity_launcher_code")
    bindings[str(runner)] = launch["launcher_sha256"]
    final_path = root / "reports" / ("s4-video-polarity-" + arm + "-training-001.json")
    actual = launch["actual_training_receipt"]
    require(
        actual["path"] == str(final_path)
        and actual["sha256"] == sha(final_path)
        and actual["status"] == final["status"]
        and actual["steps_completed"] == final["steps_completed"],
        "polarity_launcher_training_receipt",
    )
    argv = launch["training_argv"]
    require(
        argv[:4] == ["timeout", "--signal=INT", "--kill-after=20s", "240"],
        "polarity_training_timeout",
    )
    for flag, expected in {
        "--data": str(prepared / "train.jsonl"),
        "--allocation-ledger": str(prepared / "allocation-ledger.jsonl"),
        "--config": str(prepared / "config-ce.json"),
        "--output-dir": str(root / "artifacts" / ("s4-video-polarity-" + arm + "-train-001")),
        "--init-adapter": launch["selected_init_adapter"]["path"],
        "--init-adapter-sha256": launch["selected_init_adapter"]["weights_sha256"],
    }.items():
        require(
            argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
            "polarity_launcher_argv",
        )
    c = {k: v for k, v in final["config"].items() if k != "init_adapter"}
    require(
        LoopConfig.model_validate(c)
        == LoopConfig.model_validate(json.loads((prepared / "config-ce.json").read_bytes())),
        "polarity_config_exact",
    )
    samples = canonical_rows(prepared / "train.jsonl")
    from scripts.s1_text_pools import canonical_sample_sha

    rows = [
        json.loads(line)
        for line in (prepared / "allocation-ledger.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    require(
        len(rows) == len(samples) == 120 and len({r["item_id"] for r in rows}) == 120,
        "polarity_ledger_count",
    )
    for r in rows:
        sample = Sample.model_validate(samples[r["item_id"]])
        require(
            r["pool"] == "train"
            and r["family_id"] == sample.meta.family_id
            and r["derived_sample_sha256"] == canonical_sample_sha(sample),
            "polarity_exact_ledger",
        )
    arm_manifest = Manifest.model_validate_json((prepared / "inventory-manifest.json").read_bytes())
    mapping = version_mapping(arm_manifest)
    version = "clevrer-train-video-noul-polarity-" + arm + "-v1"
    require(
        mapping.get(version) == "video"
        and len([s for s in samples.values() if s["meta"]["data_version"] == version]) == 60,
        "polarity_video_mapping",
    )
    if final["status"] == "completed":
        weights = (
            root
            / "artifacts"
            / ("s4-video-polarity-" + arm + "-train-001/adapter/adapter_model.safetensors")
        )
        require(
            sha(weights) == final.get("adapter_sha256") == runtime.get("adapter_sha256"),
            "polarity_output_weights",
        )
        bindings[str(weights)] = sha(weights)
    return bindings, mapping


def order_parent_binding(
    prep: dict[str, Any], launch: dict[str, Any], final: dict[str, Any]
) -> None:
    require(
        prep.get("status") == "data_ready_only"
        and prep.get("exit_code") == 0
        and prep.get("parent_model") is None
        and prep.get("training_authorized") is False
        and prep.get("pending_parent_selection") is True
        and prep.get("inventory_isolation") == "isolated"
        and prep.get("safe_family_count") == prep.get("video_families") == 30
        and prep.get("video_rows") == prep.get("independently_evaluated_programs") == 60,
        "order_preparation_contract",
    )
    require(
        launch.get("status") in {"completed", "failed"}
        and type(launch.get("exit_code")) is int
        and launch.get("parent_selection")
        == "root_selected_f2db_balanced_retention_and_full_video_evidence",
        "order_launcher_terminal_parent",
    )
    require(
        final.get("init_adapter") == launch["selected_init_adapter"]
        and final["init_adapter"]["weights_sha256"]
        == "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
        "order_actual_parent",
    )
    config = dict(final["config"])
    if "init_adapter" in config:
        require(config.pop("init_adapter") == final["init_adapter"], "order_config_parent")
    c = LoopConfig.model_validate(config)
    require(
        c.seed == 42
        and c.steps == 10
        and c.sample_count == 120
        and c.accumulation == 12
        and c.checkpoint_every == c.midway_step == 5
        and c.reference_kl_weight == c.pair_loss_weight == c.brier_weight == 0,
        "order_ce_budget",
    )


def order_program_binding(sample: Sample, row: dict[str, Any]) -> None:
    from data.schema import HardTarget

    p = row["program_provenance"]
    program = p["program"]
    require(
        p["original_postfix_execution"] == "not_executed_source_binding_only",
        "order_postfix_boundary",
    )
    require(
        program["schema_version"] == "clevrer-collision-order-program-v1"
        and program["operation"] == "less_frame"
        and len(program["events"]) == 2,
        "order_program_schema",
    )
    frames = [e["frame"] for e in program["events"]]
    require(
        all(type(f) is int and f >= 0 for f in frames) and abs(frames[0] - frames[1]) >= 16,
        "order_frame_gap",
    )
    truth = "yes" if frames[0] < frames[1] else "no"
    require(
        sample.meta.pool == "train"
        and sample.meta.modality == ["video"]
        and len(sample.questions) == 1
        and sample.questions[0].type == "noul"
        and isinstance(sample.questions[0].target, HardTarget)
        and sample.questions[0].target.value == program["result"] == truth
        and sample.questions[0].target.source == "clevrer_collision_annotation_less_frame_v1",
        "order_program_gold",
    )
    require(
        hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
        == p["program_sha256"]
        and hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        == sample.meta.source_sha256,
        "order_program_provenance_hash",
    )


def order_contrast_bindings(
    root: Path, prepared: Path, prep: dict[str, Any], final: dict[str, Any], runtime: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    import zipfile
    from collections import Counter

    from scripts.s1_text_pools import canonical_sample_sha

    launch_path = root / "reports/s4-temporal-order-contrast-launch-001.json"
    launch, lb = read_json(launch_path)
    order_parent_binding(prep, launch, final)
    order_parent_binding(prep, launch, runtime)
    require(
        sha(prepared / "report.json")
        == launch["prepared_report_sha256"]
        == "7870a39aec8f64939dabd4035f2416b4a1c454f8dd9ec7e88f9891ffbffc28ab",
        "order_frozen_preparation",
    )
    require(
        final["data_sha256"]
        == runtime["data_sha256"]
        == "f16fbf93029461cf520522f4037ff829284a7a4a7c531d1ecd4b9e0b7bbe8ef1",
        "order_fixed_data",
    )
    bindings = {lb["path"]: lb["sha256"]}

    def bound(path: Path, expected: str) -> None:
        require(sha(path) == expected, "order_bound_input")
        bindings[str(path)] = expected

    for name, expected in prep["output_sha256"].items():
        require(Path(name).name == name, "order_safe_output_name")
        bound(prepared / name, expected)
    sources = root.parent / "s4-temporal-order-prepare-owned-001"
    bound(sources / "prepare-temporal-order-contrast.py", prep["helper_sha256"])
    for name, expected in prep["reused_helper_sha256"].items():
        require(
            name in {"inspect-temporal-order-contrast", "prepare-claim-pair-pilot"},
            "order_reused_helper_scope",
        )
        bound(sources / (name + ".py"), expected)
    for path, expected in [
        (
            root / "artifacts/s4-pilot-exposure-012/report.json",
            launch["latest_exposure_report_sha256"],
        ),
        (
            root / "artifacts/s4-temporal-coverage-video500-001/receipt.json",
            launch["parent_full_video_receipt_sha256"],
        ),
        (
            root / "artifacts/s4-temporal-coverage-old-retention-001/receipt.json",
            launch["retention_report_sha256"],
        ),
        (
            root / "artifacts/s4-temporal-coverage-video150-001/receipt.json",
            launch["video_screen_sha256"],
        ),
        (
            root / "artifacts/s4-temporal-coverage-original100-001/receipt-recovered-002.json",
            launch["cross_screen_sha256"],
        ),
    ]:
        bound(path, expected)
    require(
        launch["latest_exposure_report_sha256"]
        == prep["exposure_base"]["report_sha256"]
        == "db21553a169282aa4eb440e8fe79b5021e9ac6d37357e90bb27dc8be53f85fdf",
        "order_base012_binding",
    )
    official = root.parent / "lite-omnijev/source-audit/clevrer-official/files"
    annotations = official / "annotations/train/annotation_train.zip"
    for path, expected in [
        (annotations, prep["source_sha256"]["annotations"]),
        (official / "questions/train.json", prep["source_sha256"]["questions"]),
        (official / "README.txt", prep["source_sha256"]["readme"]),
        (root / "artifacts/s3-pools-video-001/run-a/train.jsonl", prep["source_sha256"]["train"]),
    ]:
        bound(path, expected)
    samples = canonical_rows(prepared / "train.jsonl")
    ledger = [
        json.loads(line)
        for line in (prepared / "allocation-ledger.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    require(
        len(ledger) == len(samples) == 120 and len({r["item_id"] for r in ledger}) == 120,
        "order_ledger120",
    )
    originals = {
        canonical_sample_sha(Sample.model_validate(r)): r
        for r in canonical_rows(root / "artifacts/s3-pools-video-001/run-a/train.jsonl").values()
    }
    video = []
    family_programs: dict[str, list[dict[str, Any]]] = {}
    checked_members = {}
    with zipfile.ZipFile(annotations) as archive:
        for row in ledger:
            sample = Sample.model_validate(samples[row["item_id"]])
            require(
                row["pool"] == "train"
                and row["family_id"] == sample.meta.family_id
                and row["derived_sample_sha256"] == canonical_sample_sha(sample),
                "order_exact_allocation",
            )
            if sample.meta.modality != ["video"]:
                continue
            order_program_binding(sample, row)
            family_programs.setdefault(sample.meta.family_id, []).append(
                row["program_provenance"]["program"]
            )
            video.append(sample)
            p = row["program_provenance"]
            require(
                p["annotations_zip_sha256"] == prep["source_sha256"]["annotations"]
                and p["official_questions_sha256"] == prep["source_sha256"]["questions"]
                and p["source_sample_sha256"] == row["original_sample_sha256"],
                "order_source_provenance",
            )
            original = Sample.model_validate(originals[p["source_sample_sha256"]])
            require(
                original.prefix == sample.prefix
                and original.meta.family_id == sample.meta.family_id,
                "order_original_train_component",
            )
            name = p["annotation_member"]
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and name.endswith(
                    f"/annotation_{int(sample.meta.family_id.split(':')[1]):05d}.json"
                )
                and archive.namelist().count(name) == 1,
                "order_annotation_member_identity",
            )
            if name not in checked_members:
                info = archive.getinfo(name)
                require(
                    0 < info.file_size < 5_000_000 and not info.flag_bits & 1,
                    "order_bounded_member",
                )
                checked_members[name] = hashlib.sha256(archive.read(info)).hexdigest()
            require(checked_members[name] == p["annotation_sha256"], "order_annotation_member_sha")
    counts = Counter(s.meta.family_id for s in video)
    require(
        len(video) == 60 and len(counts) == 30 and set(counts.values()) == {2},
        "order30_families60_programs",
    )
    for programs in family_programs.values():
        require(
            len(programs) == 2
            and programs[0]["events"] == list(reversed(programs[1]["events"]))
            and {p["result"] for p in programs} == {"yes", "no"},
            "order_independent_pair_programs",
        )
    replay = canonical_rows(root / "artifacts/s4-temporal-curriculum-pilot-001/train.jsonl")
    require(
        {k: v for k, v in samples.items() if v["meta"]["modality"] != ["video"]}
        == {k: v for k, v in replay.items() if v["meta"]["modality"] != ["video"]},
        "order_same60_replay",
    )
    bound(
        root / "artifacts/s4-temporal-curriculum-pilot-001/report.json",
        prep["replay_report_sha256"],
    )
    config = {k: v for k, v in final["config"].items() if k != "init_adapter"}
    require(
        LoopConfig.model_validate(config)
        == LoopConfig.model_validate(json.loads((prepared / "config-ce.json").read_bytes())),
        "order_exact_config",
    )
    audit = json.loads((prepared / "inventory-audit.json").read_bytes())
    require(
        audit["outcome"] == "isolated"
        and audit["manifest_sha256"] == sha(prepared / "inventory-manifest.json"),
        "order_prepared_isolation",
    )
    mapping = version_mapping(
        Manifest.model_validate_json((prepared / "inventory-manifest.json").read_bytes())
    )
    require(
        mapping.get("clevrer-train-collision-order-v1") == "video"
        and all(s.meta.data_version == "clevrer-train-collision-order-v1" for s in video),
        "order_program_version_mapping",
    )
    parent = launch["selected_init_adapter"]
    for name, expected in [
        ("adapter_model.safetensors", parent["weights_sha256"]),
        ("adapter_config.json", parent["config_sha256"]),
    ]:
        bound(Path(parent["path"]) / name, expected)
    if final["status"] == "completed":
        bound(
            root
            / "artifacts/s4-temporal-order-contrast-train-001/adapter/adapter_model.safetensors",
            final["adapter_sha256"],
        )
        require(final["adapter_sha256"] == runtime["adapter_sha256"], "order_final_adapter")
    return bindings, mapping


def segment002_parent(
    prep: dict[str, Any], launch: dict[str, Any], final: dict[str, Any], *, segment003: bool = False
) -> None:
    require(
        prep.get("status") == "data_ready_only"
        and prep.get("exit_code") == 0
        and prep.get("pending_parent_selection") is True
        and prep.get("training_authorized") is False
        and prep.get("inventory_isolation") == "isolated"
        and prep.get("source_admission") == "verified_existing_scope",
        "segment002_data_only_preparation",
    )
    require(
        launch.get("status") in {"completed", "failed"}
        and type(launch.get("exit_code")) is int
        and launch.get("parent_selection")
        == (
            "root_selected_second_segment_after_retention_and_full_video"
            if segment003
            else "root_selected_after_first_segment_screening"
        ),
        "segment002_parent_selection",
    )
    chosen = launch["selected_init_adapter"]
    require(
        chosen.get("weights_sha256")
        == (
            "250ef3e3c8a5359e6d7b6aaaeafa6d12b253603669a4b9f8a7a70c52c4b72707"
            if segment003
            else "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
        )
        and final.get("init_adapter") == chosen
        and chosen != prep["init_adapter"],
        "segment002_selected_parent",
    )
    config = dict(final["config"])
    if "init_adapter" in config:
        require(config.pop("init_adapter") == chosen, "segment002_config_parent")
    c = LoopConfig.model_validate(config)
    require(
        c.sample_count == 120
        and c.steps == 10
        and c.accumulation == 12
        and c.seed == 42
        and c.brier_weight == c.reference_kl_weight == c.pair_loss_weight == 0,
        "segment002_ce_contract",
    )


def segment002_provenance(
    root: Path,
    prepared: Path,
    prep: dict[str, Any],
    final: dict[str, Any],
    *,
    segment003: bool = False,
) -> dict[str, str]:
    number = "003" if segment003 else "002"
    prior_number = "002" if segment003 else "001"
    launch_path = root / "reports" / f"s4-temporal-coverage-launch-{number}.json"
    launch, lb = read_json(launch_path)
    segment002_parent(prep, launch, final, segment003=segment003)
    require(
        sha(prepared / "report.json") == launch["prepared_report_sha256"],
        "segment002_prepared_receipt",
    )
    bindings = {lb["path"]: lb["sha256"]}
    selection = prep["video_selection"]
    provenance = selection["provenance"]
    lineage_path = root / "reports/s4-ce0-video-exposure-001.json"
    first = root / "artifacts/s4-temporal-coverage-pilot-001"
    prior = root / "artifacts/s4-temporal-curriculum-pilot-001"
    for path, expected in (
        (lineage_path, provenance["lineage_report_sha256"]),
        (first / "report.json", provenance["first_coverage_report_sha256"]),
        (prior / "report.json", provenance["original_replay_report_sha256"]),
        (
            root.parent
            / (
                "s4-temporal-coverage-prepare-segment003/prepare-temporal-coverage-segment003.py"
                if segment003
                else "s4-temporal-coverage-prepare-19a778f/prepare-temporal-coverage-segment002.py"
            ),
            provenance["wrapper_sha256"],
        ),
        (
            root.parent
            / (
                "s4-temporal-coverage-prepare-segment003/prepare-temporal-curriculum-pilot.py"
                if segment003
                else "s4-temporal-coverage-prepare-19a778f/prepare-temporal-curriculum-pilot.py"
            ),
            provenance["shared_recipe_sha256"],
        ),
        (
            prepared / "train.jsonl",
            "ec15914f19ffa437c92cb5d98cdc503e5af1c99304d512146bcc797c7c8815f4"
            if segment003
            else "4fe42a806265af995f8f3d6f61ecf0aadfa82c63f86f64db2240ed138ce9aa5b",
        ),
        (
            prepared / "allocation-ledger.jsonl",
            "1bbaf15699cf82c36be94c64c5360491b9e828a58b6df885dff7787a371d8389"
            if segment003
            else "7eb9524cb5f77503f13febe4aed3d2b1c2c6bc4436be967a4ddf5c55a8503255",
        ),
        (
            root / "artifacts" / f"s4-temporal-coverage-old-retention-{prior_number}/receipt.json",
            launch["retention_report_sha256"],
        ),
        (
            root / "artifacts" / f"s4-temporal-coverage-video150-{prior_number}/receipt.json",
            launch["video_screen_sha256"],
        ),
        (
            root
            / "artifacts"
            / (
                "s4-temporal-coverage-original100-002/receipt.json"
                if segment003
                else "s4-temporal-coverage-original100-001/receipt-recovered-002.json"
            ),
            launch["cross_screen_sha256"],
        ),
    ):
        require(sha(path) == expected, "segment002_provenance_sha")
        bindings[str(path)] = expected
    lineage = json.loads(lineage_path.read_bytes())
    require(
        lineage.get("status") == "completed"
        and lineage.get("exit_code") == 0
        and len(set(lineage["lineage_unique_video_families"])) == 350,
        "segment002_lineage",
    )
    for stage in lineage["stages"]:
        for field in ("report", "data"):
            path = Path(stage[field])
            expected = stage[field + "_sha256"]
            require(sha(path) == expected, "segment002_lineage_sha")
            bindings[str(path)] = expected
    fresh = canonical_rows(prepared / "train.jsonl")
    replay = canonical_rows(prior / "train.jsonl")
    old = set(lineage["lineage_unique_video_families"]) | {
        r["meta"]["family_id"]
        for r in canonical_rows(first / "train.jsonl").values()
        if r["meta"]["modality"] == ["video"]
    }
    if segment003:
        second = root / "artifacts/s4-temporal-coverage-pilot-002"
        for path, expected in [
            (second / "report.json", provenance["second_coverage_report_sha256"]),
            (
                root / "artifacts/s4-pilot-exposure-011/report.json",
                launch["latest_exposure_report_sha256"],
            ),
            (
                root / "artifacts/s4-temporal-coverage-video500-002/receipt.json",
                launch["parent_full_video_receipt_sha256"],
            ),
        ]:
            require(sha(path) == expected, "segment003_parent_selection_sha")
            bindings[str(path)] = expected
        second_report = json.loads((second / "report.json").read_bytes())
        require(
            sha(second / "train.jsonl") == second_report["output_sha256"]["train.jsonl"],
            "segment003_second_data",
        )
        bindings[str(second / "train.jsonl")] = sha(second / "train.jsonl")
        second_families = {
            r["meta"]["family_id"]
            for r in canonical_rows(second / "train.jsonl").values()
            if r["meta"]["modality"] == ["video"]
        }
        require(
            len(second_families) == 60 and not old & second_families, "segment003_second_families"
        )
        old |= second_families
    selected = [
        r["meta"]["family_id"] for r in fresh.values() if r["meta"]["modality"] == ["video"]
    ]
    require(
        len(old) == selection["excluded_prior_family_count"] == (468 if segment003 else 408)
        and sorted(old) == selection["excluded_prior_families"]
        and len(selected) == len(set(selected)) == 60
        and sorted(selected) == selection["selected_families"]
        and not old & set(selected)
        and selection["hash_namespace"] == f"s4-temporal-coverage-segment{number}:42:",
        "segment002_family_selection",
    )
    require(
        len(fresh) == 120
        and {k: v for k, v in fresh.items() if v["meta"]["modality"] != ["video"]}
        == {k: v for k, v in replay.items() if v["meta"]["modality"] != ["video"]},
        "segment002_replay_canonical",
    )
    config = {k: v for k, v in final["config"].items() if k != "init_adapter"}
    require(
        LoopConfig.model_validate(config)
        == LoopConfig.model_validate(json.loads((prepared / "config-ce.json").read_bytes())),
        "segment002_config_exact",
    )
    chosen = launch["selected_init_adapter"]
    for name, expected in [
        ("adapter_model.safetensors", chosen["weights_sha256"]),
        ("adapter_config.json", chosen["config_sha256"]),
    ]:
        path = Path(chosen["path"]) / name
        require(sha(path) == expected, "segment002_parent_files")
        bindings[str(path)] = expected
    return bindings


def coverage_provenance(
    root: Path, prepared: Path, report: dict[str, Any], final: dict[str, Any]
) -> dict[str, str]:
    prior = root / "artifacts/s4-temporal-curriculum-pilot-001"
    selection = report["video_selection"]
    provenance = selection["provenance"]
    bindings = {}
    for path, expected in (
        (prior / "report.json", provenance["prior_report_sha256"]),
        (prior / "train.jsonl", provenance["prior_train_sha256"]),
        (
            root.parent / "s4-temporal-coverage-prepare-d42ec27/prepare-temporal-coverage-pilot.py",
            provenance["wrapper_sha256"],
        ),
        (
            root.parent
            / "s4-temporal-coverage-prepare-d42ec27/prepare-temporal-curriculum-pilot.py",
            provenance["shared_recipe_sha256"],
        ),
        (
            prepared / "train.jsonl",
            "276afa595a42379dd01c50353dc8e0ad253d43c70367682b59775d8f9ced730f",
        ),
        (
            prepared / "allocation-ledger.jsonl",
            "ef34a421d311a62b42cf96868a759cd658483bda88309200e2a20842f7867a72",
        ),
        (
            root / "artifacts/s4-temporal-coverage-train-001/adapter/adapter_model.safetensors",
            "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
        ),
    ):
        require(sha(path) == expected, "coverage_provenance_sha")
        bindings[str(path)] = expected
    require(
        provenance["prior_report_sha256"]
        == "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b",
        "coverage_prior_receipt",
    )
    require(
        final.get("adapter_sha256")
        == bindings[
            str(root / "artifacts/s4-temporal-coverage-train-001/adapter/adapter_model.safetensors")
        ],
        "coverage_adapter_receipt",
    )
    require(
        selection["hash_namespace"] == "s4-temporal-coverage-segment001:42:",
        "coverage_hash_namespace",
    )
    current, previous = (
        canonical_rows(prepared / "train.jsonl"),
        canonical_rows(prior / "train.jsonl"),
    )
    coverage_families(current, previous, selection)
    config = dict(final["config"])
    if "init_adapter" in config:
        require(config.pop("init_adapter") == report["init_adapter"], "coverage_config_init")
    expected_config = LoopConfig.model_validate(
        json.loads((prepared / "config-ce.json").read_bytes())
    )
    actual_config = LoopConfig.model_validate(config)
    require(
        actual_config == expected_config
        and actual_config.brier_weight
        == actual_config.reference_kl_weight
        == actual_config.pair_loss_weight
        == 0,
        "coverage_ce_config",
    )
    return bindings


def coverage_families(
    current: dict[str, dict[str, Any]],
    previous: dict[str, dict[str, Any]],
    selection: dict[str, Any],
) -> None:
    def videos(rows: dict[str, dict[str, Any]]) -> set[str]:
        values = [
            r["meta"]["family_id"] for r in rows.values() if r["meta"]["modality"] == ["video"]
        ]
        require(len(values) == len(set(values)) == 60, "coverage_video_families")
        return set(values)

    fresh, prior = videos(current), videos(previous)
    require(not fresh & prior, "coverage_prior_family_overlap")
    require(len(current) == len(previous) == 120, "coverage_sample_count")
    require(
        sorted(fresh) == selection["selected_families"]
        and sorted(prior) == selection["excluded_prior_families"]
        and selection["excluded_prior_family_count"] == 60,
        "coverage_selection_binding",
    )
    require(
        {k: v for k, v in current.items() if v["meta"]["modality"] != ["video"]}
        == {k: v for k, v in previous.items() if v["meta"]["modality"] != ["video"]},
        "coverage_replay_canonical",
    )


def brier_config_matches(
    base: dict[str, Any],
    actual: dict[str, Any],
    expected_init: dict[str, str] | None = None,
) -> None:
    # Trainer report config also carries bound initialization metadata. It is not
    # a LoopConfig field; remove only this exact, separately verified identity.
    if "init_adapter" in actual:
        require(
            expected_init is not None and actual["init_adapter"] == expected_init,
            "brier_report_init_metadata",
        )
        actual = {k: v for k, v in actual.items() if k != "init_adapter"}
    baseline = LoopConfig.model_validate(base)
    candidate = LoopConfig.model_validate(actual)
    require(baseline.brier_weight == 0 and candidate.brier_weight == 0.5, "brier_objective_weight")
    wanted = baseline.model_copy(update={"brier_weight": 0.5})
    require(
        candidate.model_dump(mode="json") == wanted.model_dump(mode="json"),
        "brier_only_config_difference",
    )


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
    temporal_brier: bool = False,
    learnability: bool = False,
    temporal_coverage: bool = False,
    coverage_segment002: bool = False,
    polarity_arm: str | None = None,
    coverage_segment003: bool = False,
    temporal_order_contrast: bool = False,
) -> tuple[Path, dict[str, str] | None]:
    if temporal_order_contrast:
        coverage_segment003 = True
    if coverage_segment003:
        coverage_segment002 = True
    if polarity_arm is not None:
        require(polarity_arm in {"control", "polarity"}, "polarity_single_arm")
        coverage_segment002 = True
    if (
        temporal_curriculum
        or temporal_brier
        or learnability
        or temporal_coverage
        or coverage_segment002
    ):
        exposure = root / (
            "artifacts/s4-pilot-exposure-012"
            if temporal_order_contrast
            else "artifacts/s4-pilot-exposure-011"
            if coverage_segment003
            else "artifacts/s4-pilot-exposure-010"
            if polarity_arm == "polarity"
            else "artifacts/s4-pilot-exposure-009"
            if polarity_arm == "control"
            else "artifacts/s4-pilot-exposure-008"
            if coverage_segment002
            else "artifacts/s4-pilot-exposure-007"
            if temporal_coverage
            else "artifacts/s4-pilot-exposure-006-retry-001"
            if learnability
            else "artifacts/s4-pilot-exposure-005"
            if temporal_brier
            else "artifacts/s4-pilot-exposure-004"
        )
        report, binding = read_json(exposure / "report.json")
        require(
            report.get("status") == "known_pilot_exposures_reconciled"
            and report.get("exit_code") == 0
            and report.get("unknown_update_bearing_inputs") == 0,
            "exposure004_not_passed",
        )
        if temporal_brier or learnability or temporal_coverage or coverage_segment002:
            require(bool(report.get("all_bound_input_sha256")), "missing005_input_bindings")
            require(
                all(
                    sha(Path(p)) == expected
                    for p, expected in report["all_bound_input_sha256"].items()
                ),
                "mutated005_bound_input",
            )
        if temporal_order_contrast:
            require(
                report.get("coverage_segment003_included") is True,
                "prior_segment003_exposure_required",
            )
        elif coverage_segment003:
            require(report.get("polarity_arm") == "polarity", "prior_polarity_exposure_required")
        if polarity_arm == "polarity":
            require(report.get("polarity_arm") == "control", "prior_control_exposure_required")
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
    temporal_brier: bool = False,
    learnability: bool = False,
    temporal_coverage: bool = False,
    coverage_segment002: bool = False,
    polarity_arm: str | None = None,
    coverage_segment003: bool = False,
    temporal_order_contrast: bool = False,
) -> dict[str, Any]:
    if polarity_arm is not None:
        require(polarity_arm in {"control", "polarity"}, "polarity_single_arm")
        coverage_segment002 = True
    if temporal_order_contrast:
        coverage_segment003 = True
    if coverage_segment003:
        require(polarity_arm is None, "single_experiment_branch")
        coverage_segment002 = True
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
            temporal_brier=temporal_brier,
            learnability=learnability,
            temporal_coverage=temporal_coverage,
            coverage_segment002=coverage_segment002,
            polarity_arm=polarity_arm,
            coverage_segment003=coverage_segment003,
            temporal_order_contrast=temporal_order_contrast,
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
        if (
            polarity_arm is not None or coverage_segment003
        ) and base_preparation_binding is not None:
            prior_base = json.loads(Path(base_preparation_binding["path"]).read_bytes())
            bindings.update(prior_base["all_bound_input_sha256"])
        pilots, new_artifacts = [], []
        for name, final_path, run_dir, preparation_dir in pilot_paths(
            root,
            crossmodal_retention=crossmodal_retention,
            claim_pair=claim_pair,
            temporal_curriculum=temporal_curriculum,
            temporal_brier=temporal_brier,
            learnability=learnability,
            temporal_coverage=temporal_coverage,
            coverage_segment002=coverage_segment002,
            polarity_arm=polarity_arm,
            coverage_segment003=coverage_segment003,
            temporal_order_contrast=temporal_order_contrast,
        ):
            final, final_binding = read_json(final_path)
            runtime_report, run_binding = read_json(run_dir / "run.json")
            terminal_report(final)
            terminal_report(runtime_report)
            is_claim = (
                claim_pair
                or temporal_curriculum
                or temporal_brier
                or learnability
                or temporal_coverage
                or coverage_segment002
            ) and name.startswith("s4-claim-pair-")
            is_temporal = (
                temporal_curriculum
                or temporal_brier
                or learnability
                or temporal_coverage
                or coverage_segment002
            ) and name == "s4-temporal-curriculum"
            is_brier = (
                temporal_brier or learnability or temporal_coverage or coverage_segment002
            ) and name == "s4-temporal-brier"
            is_learnability = (
                learnability or temporal_coverage or coverage_segment002
            ) and name == "s4-train-learnability"
            is_coverage = (
                temporal_coverage or coverage_segment002
            ) and name == "s4-temporal-coverage"
            is_segment002 = coverage_segment002 and name == "s4-temporal-coverage-segment002"
            is_segment003 = coverage_segment003 and name == "s4-temporal-coverage-segment003"
            is_order = temporal_order_contrast and name == "s4-temporal-order-contrast"
            is_polarity = polarity_arm is not None and name == "s4-video-polarity-" + polarity_arm
            if is_polarity:
                claim_terminal(final, name)
                claim_terminal(runtime_report, name)
                require(
                    final.get("config") == runtime_report.get("config")
                    and final.get("init_adapter") == runtime_report.get("init_adapter"),
                    "polarity_final_run_identity",
                )
            if is_learnability:
                learnability_terminal(final)
                learnability_terminal(runtime_report)
                require(
                    final.get("config") == runtime_report.get("config"),
                    "learnability_runtime_config_mismatch",
                )
            if (
                is_claim
                or is_temporal
                or is_brier
                or is_coverage
                or is_segment002
                or is_segment003
                or is_polarity
                or is_order
            ):
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
            prep_receipt = (
                preparation_dir.parent / "report.json"
                if is_polarity
                else preparation_dir / "report.json"
            )
            prep_report, prep_report_binding = read_json(prep_receipt)
            if is_order:
                extra_bindings, extra_mapping = order_contrast_bindings(
                    root, preparation_dir, prep_report, final, runtime_report
                )
                for bound_path, expected in extra_bindings.items():
                    require(
                        bound_path not in bindings or bindings[bound_path] == expected,
                        "mutable_report",
                    )
                    bindings[bound_path] = expected
                for version, cohort in extra_mapping.items():
                    require(
                        version not in mappings or mappings[version] == cohort,
                        "inventory_version_conflict",
                    )
                    mappings[version] = cohort
            elif is_polarity:
                extra_bindings, extra_mapping = polarity_bindings(
                    root, preparation_dir, prep_report, final, runtime_report, polarity_arm
                )
                for bound_path, expected in extra_bindings.items():
                    require(
                        bound_path not in bindings or bindings[bound_path] == expected,
                        "mutable_report",
                    )
                    bindings[bound_path] = expected
                for version, cohort in extra_mapping.items():
                    require(
                        version not in mappings or mappings[version] == cohort,
                        "inventory_version_conflict",
                    )
                    mappings[version] = cohort
            elif is_learnability:
                require(
                    prep_report.get("status") == "ready_for_preflight"
                    and prep_report.get("exit_code") == 0
                    and prep_report.get("samples") == 12
                    and prep_report.get("draws") == 120
                    and prep_report.get("source_isolation") == "isolated",
                    "learnability_preparation",
                )
                weights = bind(run_dir / "adapter/adapter_model.safetensors")
                require(
                    weights["sha256"]
                    == final.get("adapter_sha256")
                    == runtime_report.get("adapter_sha256")
                    == "02a579d5094d61d53018046a0c542edcef19b88b929fb3bdd8714448366d54c7",
                    "learnability_trained_adapter",
                )
                bindings[weights["path"]] = weights["sha256"]
                source_dir = root / "artifacts/s4-temporal-curriculum-pilot-001"
                require(
                    sha(source_dir / "report.json") == prep_report["source_report_sha256"],
                    "learnability_source_receipt",
                )
                require(
                    sha(source_dir / "train.jsonl") == prep_report["source_data_sha256"],
                    "learnability_source_data",
                )
                for filename in ("allocation-ledger.jsonl", "config-ce.json"):
                    b = bind(preparation_dir / filename)
                    require(
                        b["sha256"] == prep_report["output_sha256"][filename],
                        "learnability_preparation_sha",
                    )
                    bindings[b["path"]] = b["sha256"]
                require(
                    sha(preparation_dir / "allocation-ledger.jsonl")
                    == sha(source_dir / "allocation-ledger.jsonl"),
                    "learnability_source_ledger",
                )
                require(
                    final.get("init_adapter")
                    == runtime_report.get("init_adapter")
                    == prep_report["init_adapter"],
                    "learnability_init",
                )
                actual_config = dict(final["config"])
                if "init_adapter" in actual_config:
                    require(
                        actual_config.pop("init_adapter") == prep_report["init_adapter"],
                        "learnability_config_init",
                    )
                require(
                    LoopConfig.model_validate(actual_config)
                    == LoopConfig.model_validate(
                        json.loads((preparation_dir / "config-ce.json").read_bytes())
                    ),
                    "learnability_config_identity",
                )
            elif is_temporal or is_brier or is_coverage or is_segment002 or is_segment003:
                if not (is_segment002 or is_segment003):
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
                    == (
                        read_json(
                            root
                            / "reports"
                            / (
                                "s4-temporal-coverage-launch-003.json"
                                if is_segment003
                                else "s4-temporal-coverage-launch-002.json"
                            )
                        )[0]["selected_init_adapter"]
                        if is_segment002 or is_segment003
                        else prep_report["init_adapter"]
                    ),
                    "temporal_init_binding",
                )
            else:
                preparation_passed(prep_report, claim_pair=is_claim)
            if is_coverage:
                require(
                    final.get("adapter_sha256") == runtime_report.get("adapter_sha256"),
                    "coverage_runtime_adapter",
                )
                for bound_path, expected in coverage_provenance(
                    root, preparation_dir, prep_report, final
                ).items():
                    require(
                        bound_path not in bindings or bindings[bound_path] == expected,
                        "mutable_report",
                    )
                    bindings[bound_path] = expected
            if is_segment002 or is_segment003:
                require(
                    final.get("adapter_sha256") == runtime_report.get("adapter_sha256"),
                    "segment002_runtime_adapter",
                )
                if (
                    not is_segment003
                    or final["status"] == "completed"
                    or final.get("adapter_sha256")
                ):
                    weights = bind(run_dir / "adapter/adapter_model.safetensors")
                    require(
                        weights["sha256"] == final.get("adapter_sha256"),
                        "segment002_output_weights",
                    )
                    bindings[weights["path"]] = weights["sha256"]
                for bound_path, expected in segment002_provenance(
                    root, preparation_dir, prep_report, final, segment003=is_segment003
                ).items():
                    require(
                        bound_path not in bindings or bindings[bound_path] == expected,
                        "mutable_report",
                    )
                    bindings[bound_path] = expected
            if is_brier:
                config_path = root / "artifacts/s4-temporal-brier-pilot-001/config-brier.json"
                config_payload, config_binding = read_json(config_path)
                launch_path = root / "reports/s4-temporal-brier-launch-001.json"
                launch, launch_binding = read_json(launch_path)
                require(
                    launch.get("status") in {"completed", "failed"}
                    and type(launch.get("exit_code")) is int,
                    "brier_launcher_nonterminal",
                )
                require(
                    launch["prepared_report_sha256"] == prep_report_binding["sha256"]
                    and launch["config_sha256"] == config_binding["sha256"],
                    "brier_launcher_binding",
                )
                base_config, _ = read_json(preparation_dir / "config-ce.json")
                brier_config_matches(base_config, config_payload, prep_report["init_adapter"])
                brier_config_matches(base_config, final["config"], prep_report["init_adapter"])
                for b in (config_binding, launch_binding):
                    require(
                        b["path"] not in bindings or bindings[b["path"]] == b["sha256"],
                        "mutable_report",
                    )
                    bindings[b["path"]] = b["sha256"]
            preparation_data = preparation_dir / "train.jsonl"
            preparation_binding = bind(preparation_data)
            expected_outputs = (
                prep_report["arms"][polarity_arm]["output_sha256"]
                if is_polarity
                else prep_report["output_sha256"]
            )
            expected = expected_outputs.get("train", expected_outputs.get("train.jsonl"))
            require(preparation_binding["sha256"] == expected, "preparation_data_sha")
            comparison = equivalent(runtime_data, preparation_data)
            if is_learnability:
                comparison.update(
                    learnability_subset(
                        preparation_data,
                        root / "artifacts/s4-temporal-curriculum-pilot-001/train.jsonl",
                    )
                )
            if (
                is_claim
                or is_temporal
                or is_brier
                or is_coverage
                or is_segment002
                or is_segment003
                or is_polarity
                or is_order
            ):
                require(comparison["rows"] == 120, "claim_runtime_count")
            require(
                all(version in mappings for version in comparison["data_versions"]),
                "unmapped_data_version",
            )
            artifact = {
                **runtime_binding,
                "cohorts_by_data_version": {v: mappings[v] for v in comparison["data_versions"]},
                "exposure": "current" if is_polarity or is_segment003 or is_order else "historical",
                "expected_pool": "train",
            }
            appended = False
            if final["steps_completed"] > 0 and runtime_binding["sha256"] not in hashes:
                require(
                    runtime_binding["sha256"] not in all_hashes, "canonical_sha_nontrain_collision"
                )
                if is_polarity or is_segment003 or is_order:
                    arm_current_inventory(combined, artifact)
                else:
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
                    "appended_historical_train": appended
                    and not (is_polarity or is_segment003 or is_order),
                    **(
                        {"appended_current_train": appended}
                        if is_polarity or is_segment003 or is_order
                        else {}
                    ),
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
            terminal_report(report, allow_paused=True)
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
        if (
            crossmodal_retention
            or claim_pair
            or temporal_curriculum
            or temporal_brier
            or learnability
            or temporal_coverage
            or coverage_segment002
        ):
            report.update(
                base_preparation_report=base_preparation_binding, crossmodal_retention_included=True
            )
        if (
            claim_pair
            or temporal_curriculum
            or temporal_brier
            or learnability
            or temporal_coverage
            or coverage_segment002
        ):
            report["claim_pair_included"] = True
        if (
            temporal_curriculum
            or temporal_brier
            or learnability
            or temporal_coverage
            or coverage_segment002
        ):
            report["temporal_curriculum_included"] = True
            report["base_exposure_report"] = base_preparation_binding
        if temporal_brier or learnability or temporal_coverage or coverage_segment002:
            report["temporal_brier_included"] = True
        if learnability or temporal_coverage or coverage_segment002:
            report["learnability_included"] = True
        if temporal_coverage or coverage_segment002:
            report["temporal_coverage_included"] = True
        if coverage_segment002:
            report["coverage_segment002_included"] = True
        if polarity_arm is not None:
            report["polarity_arm"] = polarity_arm
        if coverage_segment003:
            report["coverage_segment003_included"] = True
        if temporal_order_contrast:
            report["temporal_order_contrast_included"] = True
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


CONTINUOUS_DATA_SHA = "aeee70458ca202e68f80f9d9b7dc617929a4bc1d4c2eb1e81113a25a207441a0"
CONTINUOUS_CONFIG_SHA = "d8e9e5b0fc8427b32f39165c58486bfd93123fea1a584b5c0f8508efcb389414"
CONTINUOUS_PREP_SHA = "bdf25dbe6955b983d263d13d3436f930481dc908241ef12b56526545156b1cc9"
CONTINUOUS_BASE_SHA = "5872a667ef0935436430e22f8c041c85f4e4f2c0c4ffc1e0076e4a9e6525b15e"
CONTINUOUS_PARENT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"


def continuous_interval(final: dict, start: int, end: int) -> dict:
    """Cumulative optimizer steps define one non-overlapping exposure interval."""
    terminal_report(final, allow_paused=True)
    require(
        type(start) is int and type(end) is int and 0 <= start <= end <= 100,
        "continuous_step_interval",
    )
    require(final["steps_completed"] == end and end - start <= 10, "continuous_cumulative_steps")
    require(final["status"] != "blocked" or end == start, "continuous_blocked_updates")
    if final["status"] in {"paused", "completed"}:
        require(end > start and final["exit_code"] == 0, "continuous_success_interval")
        require(
            (final["status"] == "completed") == (end == 100), "continuous_global_terminal_status"
        )
        invocation = final.get("invocation", {})
        require(
            invocation.get("start_step") == start
            and invocation.get("steps_completed") == end - start
            and invocation.get("steps_requested") == 10
            and invocation.get("global_steps_target") == 100
            and invocation.get("next_draw") == end * 12,
            "continuous_invocation_receipt",
        )
    return {
        "start_step": start,
        "end_step": end,
        "steps_this_invocation": end - start,
        "actual_draw_interval": [12 * start, 12 * end],
        "actual_draws_this_invocation": 12 * (end - start),
        "actual_global_draws_seen": 12 * end,
        "global_pool_draw_capacity": 1200,
        "cumulative_steps_are_not_summed": True,
        "actual_seen_boundary": "completed optimizer blocks",
        "partial_step_draws_unknown": final["status"] == "failed",
        "possible_global_draws_seen_upper_bound": (
            min(100, start + 10, end + 1) * 12 if final["status"] == "failed" else end * 12
        ),
    }


def continuous_checkpoint(
    path: Path, final: dict, expected_step: int, bindings: dict[str, str]
) -> dict:
    import random

    from train.checkpoint import load_checkpoint_metadata, verify_checkpoint

    require(path.is_absolute(), "continuous_checkpoint_absolute")
    metadata = load_checkpoint_metadata(path)
    order = list(range(1200))
    random.Random(42).shuffle(order)
    step = verify_checkpoint(
        metadata,
        config=final["config"],
        model_revision=final["model_revision"],
        data_sha256=final["data_sha256"],
        asset_hashes=final["asset_hashes"],
        order=order,
    )
    require(step == expected_step, "continuous_checkpoint_step")
    files = [path / "checkpoint.json", *(path / n for n in metadata["file_sha256"])]
    for file in files:
        bindings[str(file)] = sha(file)
    return {
        "path": str(path),
        "step": step,
        "metadata": bind(path / "checkpoint.json"),
        "tensor_sha256": metadata["file_sha256"],
        "config_sha256": metadata["config_sha256"],
        "order_sha256": metadata["order_sha256"],
    }


def continuous_fixed_identity(prep: dict, config: dict, final: dict) -> None:
    require(
        prep.get("status") == "data_ready_only"
        and prep.get("parent") is None
        and prep.get("training_authorized") is False
        and prep.get("inventory_isolation") == "isolated",
        "continuous_data_only_preparation",
    )
    require(
        (
            config.get("steps"),
            config.get("sample_count"),
            config.get("accumulation"),
            config.get("seed"),
            config.get("rank"),
            config.get("alpha"),
            config.get("video_num_frames"),
            config.get("image_max_soft_tokens"),
        )
        == (100, 1200, 12, 42, 16, 32, 16, 1120),
        "continuous_frozen_config",
    )
    require(
        not any(
            config.get(k, 0) for k in ("pair_loss_weight", "reference_kl_weight", "brier_weight")
        ),
        "continuous_ce_only",
    )
    parent = final.get("init_adapter", {})
    require(
        parent.get("weights_sha256") == CONTINUOUS_PARENT_SHA, "continuous_original_f2db_parent"
    )
    actual = dict(final["config"])
    require(actual.pop("init_adapter", None) == parent, "continuous_config_parent")
    expected = LoopConfig.model_validate(config).model_dump()
    # Mirror the trainer's historical zero-loss compatibility normalization.
    from train.trainer import REFERENCE_FIELDS

    if not expected.get("reference_kl_weight", 0):
        for field in REFERENCE_FIELDS:
            expected.pop(field, None)
    if expected.get("brier_weight") == 0:
        expected.pop("brier_weight")
    if expected.get("preparation_mode") == "eager":
        expected.pop("preparation_mode")
        expected.pop("preparation_cache_bytes")
    require(actual == expected, "continuous_runtime_config_identity")


def prepare_continuous(
    root: Path, legacy_root: Path, output: Path, historical_index: Path, launch_path: Path
) -> dict:
    """One settled invocation over one immutable pool, based on the preceding inventory."""
    started = time.monotonic()
    require(
        all(p.is_absolute() for p in (root, legacy_root, output, historical_index, launch_path)),
        "absolute_paths",
    )
    require(not output.exists(), "output_exists")
    output.mkdir(parents=True, mode=0o700)
    try:
        launch, launch_binding = read_json(launch_path)
        require(
            launch.get("status") in TERMINAL | {"paused"} and type(launch.get("exit_code")) is int,
            "continuous_launch_terminal",
        )
        eb = launch["exposure_binding"]
        bindings = {launch_binding["path"]: launch_binding["sha256"]}

        def bound(key):
            b = eb[key]
            path = Path(b["path"])
            require(
                path.is_absolute() and sha(path) == b["sha256"], "continuous_launcher_bound_input"
            )
            bindings[str(path)] = b["sha256"]
            return path

        base_path = bound("base_exposure_report")
        base, base_binding = read_json(base_path)
        require(
            base.get("status") == "known_pilot_exposures_reconciled"
            and base.get("exit_code") == 0
            and base.get("unknown_update_bearing_inputs") == 0,
            "continuous_base_passed",
        )
        start, end = eb["start_step"], eb["end_step"]
        history = base.get("continuous_invocations", [])
        if history:
            previous = history[-1]
            require(
                previous["end_step"] == start and previous["status"] == "paused",
                "continuous_prior_interval",
            )
        else:
            require(
                start == 0 and base_binding["sha256"] == CONTINUOUS_BASE_SHA,
                "continuous_initial013",
            )
        prior_bindings = base["all_bound_input_sha256"]
        require(
            all(sha(Path(p)) == h for p, h in prior_bindings.items()), "continuous_prior_mutation"
        )
        bindings.update(prior_bindings)
        base_inventory = base_path.with_name("inventory-manifest.json")
        require(
            sha(base_inventory) == base["output_sha256"]["inventory-manifest.json"],
            "continuous_base_inventory_sha",
        )
        bindings[str(base_inventory)] = sha(base_inventory)
        inventory = Manifest.model_validate_json(base_inventory.read_bytes())
        mappings = version_mapping(inventory)
        combined = inventory.model_dump(mode="json", exclude_none=True)
        prep_path = bound("prepared_report")
        require(sha(prep_path) == CONTINUOUS_PREP_SHA, "continuous_preparation_sha")
        prep = json.loads(prep_path.read_bytes())
        for name, expected in prep["output_sha256"].items():
            require(Path(name).name == name, "continuous_safe_preparation_name")
            path = prep_path.with_name(name)
            require(sha(path) == expected, "continuous_preparation_output_sha")
            bindings[str(path)] = expected
        config_path = bound("config_file")
        require(sha(config_path) == CONTINUOUS_CONFIG_SHA, "continuous_global_config_sha")
        config = json.loads(config_path.read_bytes())
        final_path, snapshot_path = bound("final_report"), bound("runtime_report_snapshot")
        require(
            snapshot_path.name != "run.json" and final_path != snapshot_path,
            "continuous_immutable_report_names",
        )
        final, runtime = json.loads(final_path.read_bytes()), json.loads(snapshot_path.read_bytes())
        final_core = dict(final)
        ledger_check = final_core.pop("allocation_ledger_check", None)
        if ledger_check is not None:
            require(
                ledger_check.get("verified_train") == 1200
                and ledger_check.get("ledger_rows", 0) >= 1200,
                "continuous_final_train_ledger_check",
            )
        require(final_core == runtime, "continuous_final_snapshot_identity")
        require(
            final["status"] == launch["status"] and final["exit_code"] == launch["exit_code"],
            "continuous_launcher_terminal_identity",
        )
        raw_path = Path(eb["runtime_report_path"])
        require(
            raw_path.is_absolute()
            and raw_path.name == "run.json"
            and sha(raw_path) == sha(snapshot_path),
            "continuous_current_runtime_snapshot",
        )
        # run.json is deliberately observed, never persisted as an immutable history binding.
        mutable_observation = bind(raw_path)
        interval = continuous_interval(final, start, end)
        continuous_fixed_identity(prep, config, final)
        runtime_data = bound("runtime_data")
        require(
            sha(runtime_data) == final["data_sha256"] == CONTINUOUS_DATA_SHA,
            "continuous_runtime_data_sha",
        )
        comparison = equivalent(runtime_data, prep_path.with_name("train.jsonl"))
        require(
            comparison["rows"] == 1200 and all(v in mappings for v in comparison["data_versions"]),
            "continuous_pool_mapping",
        )
        sources = eb["source_files"]
        if history:
            require(sources == history[-1]["source_files"], "continuous_source_identity_changed")
        require(
            {"trainer.py", "checkpoint.py", "train_s1.py"}
            <= {Path(b["path"]).name for b in sources},
            "continuous_source_bindings_required",
        )
        for b in sources:
            path = Path(b["path"])
            require(path.is_absolute() and sha(path) == b["sha256"], "continuous_source_sha")
            bindings[str(path)] = b["sha256"]
        parent = final["init_adapter"]
        require(
            launch.get("selected_init_adapter") == parent
            and launch.get("parent_selection")
            == "root_selected_f2db_balanced_retention_and_full_video_evidence",
            "continuous_root_selected_parent",
        )
        require(
            runtime_data == raw_path.parent / "data.jsonl", "continuous_actual_runtime_data_path"
        )
        parent_dir = Path(parent["path"])
        require(
            parent_dir.is_absolute()
            and sha(parent_dir / "adapter_model.safetensors") == CONTINUOUS_PARENT_SHA,
            "continuous_parent_weights_sha",
        )
        bindings[str(parent_dir / "adapter_model.safetensors")] = CONTINUOUS_PARENT_SHA
        require(
            sha(parent_dir / "adapter_config.json") == parent["config_sha256"],
            "continuous_parent_config_sha",
        )
        bindings[str(parent_dir / "adapter_config.json")] = parent["config_sha256"]
        run_dir = raw_path.parent
        resume = eb["resume_from"]
        resume_identity = None
        if start:
            require(
                resume == final.get("resumed_from") == history[-1]["checkpoint"]["path"],
                "continuous_exact_resume_path",
            )
            require(Path(resume).parent == run_dir / "checkpoints", "continuous_same_output_resume")
            resume_identity = continuous_checkpoint(Path(resume), final, start, bindings)
            require(
                resume_identity == history[-1]["checkpoint"],
                "continuous_resume_checkpoint_identity",
            )
        else:
            require(resume is None and not final.get("resumed_from"), "continuous_first_no_resume")
        checkpoint = None
        adapter = None
        if final["status"] in {"paused", "completed"}:
            checkpoint_path = run_dir / "checkpoints" / f"step-{end:04d}"
            require(final.get("checkpoint") == str(checkpoint_path), "continuous_output_checkpoint")
            checkpoint = continuous_checkpoint(checkpoint_path, final, end, bindings)
            adapter_dir = run_dir / f"adapter-invocation-step-{end:04d}"
            require(final.get("adapter_dir") == str(adapter_dir), "continuous_output_adapter")
            adapter = bind(adapter_dir / "adapter_model.safetensors")
            require(adapter["sha256"] == final["adapter_sha256"], "continuous_adapter_sha")
            bindings[adapter["path"]] = adapter["sha256"]
            bindings[str(adapter_dir / "adapter_config.json")] = sha(
                adapter_dir / "adapter_config.json"
            )
        else:
            # Failed partial updates remain exposure even without a final adapter/checkpoint.
            if final.get("checkpoint"):
                path = Path(final["checkpoint"])
                require(
                    path.parent == run_dir / "checkpoints", "continuous_failed_checkpoint_scope"
                )
                checkpoint = continuous_checkpoint(
                    path, final, int(path.name.removeprefix("step-")), bindings
                )
                require(start <= checkpoint["step"] <= end, "continuous_failed_checkpoint_range")
        appended = []
        if end > start:
            existing = [a for a in combined["samples"] if a["sha256"] == CONTINUOUS_DATA_SHA]
            if existing:
                require(
                    len(existing) == 1
                    and existing[0]["expected_pool"] == "train"
                    and existing[0]["exposure"] == "current",
                    "continuous_single_current_pool",
                )
            else:
                artifact = {
                    **bind(runtime_data),
                    "expected_pool": "train",
                    "exposure": "current",
                    "cohorts_by_data_version": {
                        v: mappings[v] for v in comparison["data_versions"]
                    },
                }
                arm_current_inventory(combined, artifact)
                appended.append(artifact)
                component_path = prep_path.with_name("components.jsonl")
                combined["component_ledgers"].append(bind(component_path))
        hashes = {a["sha256"] for a in combined["samples"] if a["expected_pool"] == "train"}
        index, index_binding = read_json(historical_index)
        require(
            index.get("status") == "indexed" and not index.get("unresolved"), "old_index_incomplete"
        )
        bindings[index_binding["path"]] = index_binding["sha256"]
        known_paths = set()
        for item in index["files"]:
            require(
                item["sha256"] in hashes and sha(Path(item["path"])) == item["sha256"],
                "continuous_indexed_input",
            )
            bindings[item["path"]] = item["sha256"]
            for b in item["report_bindings"]:
                path = Path(b["path"])
                require(sha(path) == b["sha256"], "old_report_mutated")
                require(
                    json.loads(path.read_bytes()).get("data_sha256") == item["sha256"],
                    "old_report_input_binding",
                )
                bindings[str(path)] = b["sha256"]
                known_paths.add(path)
        discovered = (
            known_paths
            | set((root / "reports").glob("*.json"))
            | set((root / "artifacts").rglob("run.json"))
            | set((legacy_root / "artifacts/training").rglob("run.json"))
            | {Path(p) for p in prior_bindings if Path(p).suffix == ".json"}
            | {final_path, snapshot_path}
        )
        updates, unknown = [], []
        for path in sorted(discovered):
            try:
                r, b = read_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                require(
                    path.name != "run.json" and path not in known_paths,
                    "unreadable_training_report",
                )
                continue
            if (
                not isinstance(r, dict)
                or type(r.get("steps_completed")) is not int
                or r["steps_completed"] <= 0
            ):
                continue
            terminal_report(r, allow_paused=True)
            data_sha = r.get("data_sha256")
            require(
                isinstance(data_sha, str) and len(data_sha) == 64, "update_report_without_data_sha"
            )
            if path != raw_path:
                require(
                    str(path) not in bindings or bindings[str(path)] == b["sha256"],
                    "mutable_report",
                )
                bindings[str(path)] = b["sha256"]
            entry = {
                "report": b,
                "data_sha256": data_sha,
                "steps_completed": r["steps_completed"],
                "status": r["status"],
            }
            updates.append(entry)
            if data_sha not in hashes:
                unknown.append(entry)
        write(
            output / "discovered-update-reports.json",
            {"update_reports": updates, "unknown": unknown},
        )
        require(not unknown, "unknown_update_bearing_input")
        write(output / "inventory-manifest.json", combined)
        audit = audit_manifest(output / "inventory-manifest.json")
        write(output / "inventory-audit.json", audit.model_dump(mode="json"))
        require(audit.outcome == "isolated", "full_inventory_rejected")
        require(
            all(sha(Path(p)) == h for p, h in bindings.items())
            and sha(raw_path) == mutable_observation["sha256"],
            "mutable_report_or_input",
        )
        invocation = {
            **interval,
            "status": final["status"],
            "update_bearing": end > start,
            "launch_report": launch_binding,
            "final_report": bind(final_path),
            "runtime_report_snapshot": bind(snapshot_path),
            "runtime_report_observation": mutable_observation,
            "checkpoint": checkpoint,
            "resume_checkpoint": resume_identity,
            "adapter": adapter,
            "original_init_adapter": parent,
            "source_files": sources,
        }
        history = [*history, invocation]
        report = {
            "status": "known_pilot_exposures_reconciled",
            "exit_code": 0,
            "gpu_used": False,
            "scope": (
                "Known saved reports; pool capacity is not actual exposure or quality acceptance"
            ),
            "continuous_main_pool_included": True,
            "base_exposure_report": base_binding,
            "base_inventory": bind(base_inventory),
            "historical_index": index_binding,
            "continuous_invocations": history,
            "actual_global_draws_seen": end * 12,
            "global_pool_draw_capacity": 1200,
            "pilots": [invocation],
            "appended_artifacts": appended,
            "indexed_input_files": len(index["files"]),
            "discovered_update_bearing_reports": len(updates),
            "unknown_update_bearing_inputs": 0,
            "all_bound_input_sha256": bindings,
            "output_sha256": {
                n: sha(output / n)
                for n in (
                    "inventory-manifest.json",
                    "inventory-audit.json",
                    "discovered-update-reports.json",
                )
            },
            "runner_sha256": sha(Path(__file__)),
            "duration_seconds": time.monotonic() - started,
            "cost": None,
        }
        write(output / "report.json", report)
        return report
    except Exception as exc:
        if not (output / "report.json").exists():
            write(
                output / "report.json",
                {
                    "status": "rejected",
                    "exit_code": 1,
                    "gpu_used": False,
                    "category": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
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
    parser.add_argument(
        "--temporal-brier",
        action="store_true",
        help="Use settled exposure005 base and include matched CE+Brier terminal run",
    )
    parser.add_argument(
        "--learnability",
        action="store_true",
        help="Use006retry base and bind terminal12-sample/120-draw learnability updates",
    )
    parser.add_argument(
        "--temporal-coverage",
        action="store_true",
        help="Use007 base and bind new terminal60-family coverage segment",
    )
    parser.add_argument(
        "--coverage-segment002",
        action="store_true",
        help="Use008 base; bind terminal segment002 to launcher-selected parent",
    )
    parser.add_argument(
        "--polarity-arm",
        choices=["control", "polarity"],
        help="Bind terminal single arm; control uses009, polarity requires010 settled control",
    )
    parser.add_argument(
        "--coverage-segment003",
        action="store_true",
        help="Use011 base; bind third segment to selected250ef parent and468-family exclusion",
    )
    parser.add_argument(
        "--temporal-order-contrast",
        action="store_true",
        help="Use012 base; bind30-family independently evaluated order-program gold",
    )
    parser.add_argument("--continuous-main-pool", action="store_true")
    parser.add_argument("--continuous-launch-report", type=Path)
    parser.add_argument("--continuous-base-exposure", type=Path)
    args = parser.parse_args()
    try:
        if args.continuous_base_exposure is not None:
            require(args.continuous_main_pool, "continuous_base_requires_branch")
            launch_path = (
                args.continuous_launch_report
                or args.root / "reports/s4-continuous-main-pool-launch-001.json"
            )
            launch = json.loads(launch_path.read_bytes())
            require(
                launch["exposure_binding"]["base_exposure_report"]["path"]
                == str(args.continuous_base_exposure),
                "continuous_cli_base_mismatch",
            )
        report = (
            prepare_continuous(
                args.root,
                args.legacy_root,
                args.output_dir,
                args.historical_index
                or args.root / "artifacts/s4-gold-inventory-001/historical-training-index.json",
                args.continuous_launch_report
                or args.root / "reports/s4-continuous-main-pool-launch-001.json",
            )
            if args.continuous_main_pool
            else prepare(
                args.root,
                args.legacy_root,
                args.output_dir,
                args.historical_index
                or args.root / "artifacts/s4-gold-inventory-001/historical-training-index.json",
                crossmodal_retention=args.crossmodal_retention,
                claim_pair=args.claim_pair,
                temporal_curriculum=args.temporal_curriculum,
                temporal_brier=args.temporal_brier,
                learnability=args.learnability,
                temporal_coverage=args.temporal_coverage,
                coverage_segment002=args.coverage_segment002,
                polarity_arm=args.polarity_arm,
                coverage_segment003=args.coverage_segment003,
                temporal_order_contrast=args.temporal_order_contrast,
            )
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
