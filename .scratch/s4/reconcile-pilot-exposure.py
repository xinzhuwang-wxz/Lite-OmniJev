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
    native_event_condition: bool = False,
    video_specialist: bool = False,
    video_text_matched: bool = False,
) -> list[tuple[str, Path, Path, Path]]:
    if video_text_matched:
        return [
            (
                "s4-video-text-matched",
                root / "reports/s4-video-text-matched-training-001.json",
                root / "artifacts/s4-video-text-matched-train-001",
                root / "artifacts/s4-video-text-matched-pilot-001",
            )
        ]
    if video_specialist:
        return [
            (
                "s4-video-specialist",
                root / "reports/s4-video-specialist-training-001.json",
                root / "artifacts/s4-video-specialist-train-001",
                root / "artifacts/s4-video-specialist-pilot-001",
            )
        ]
    if native_event_condition:
        return [
            (
                "s4-native-event-condition",
                root / "reports/s4-native-event-condition-training-001.json",
                root / "artifacts/s4-native-event-condition-train-001",
                root / "artifacts/s4-native-event-condition-pilot-001",
            )
        ]
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
    "control": ("f7cbd1a5ee86c92cb13fc9377d9942b4e6907e0182d685431103b4d20ddc85bc"),
    "polarity": ("0235ff3f039c28229c897bf79a5f353958d34d4bda486035cc72a21df691cb84"),
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


NATIVE_PREP_SHA = "401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696"
NATIVE_DATA_SHA = "fa3a16c9b0b2343fcc69fae21e7505e421c610aa5dbc6993d096498ba6789a72"
NATIVE_BASE_SHA = "debb448fab4cfe4affe3e578c53f6bc98f79ac2f89f32a44bdfe4c8ed50b9e5e"
SPECIALIST_BASE_SHA = "0164a2cdf0db9381cc76ae7411bd99f459789f4ae057c1cdfbf89d9276870aae"
SPECIALIST_PREP_SHA = "af13cfecc2ca1a27985c5fe25528b0b12578e2c5ddc66d517a86e14f77fc1942"
SPECIALIST_DATA_SHA = "11d453d3521646df888c7899b6a0a67ad4a5606d956251c96d69b41e73b0cdbe"


MATCHED_BASE_SHA = "5b3d3a26bfad2f35dc85e3f166a5d77f3c0996469fa6cbd0cbe962aa98b7e8b6"
MATCHED_PREP_SHA = "3e764378147c06816a3f14fe71967f53d83fb50871237e094ceb19587419db93"


def native_parent_binding(prep, launch, final, *, specialist=False, text_matched=False):
    claim_terminal(final, "s4-video-specialist" if specialist else "s4-native-event-condition")
    require(
        prep.get("status") == "data_ready_only"
        and prep.get("exit_code") == 0
        and prep.get("parent_model") is None
        and prep.get("training_authorized") is False
        and prep.get("pending_parent_selection") is True
        and prep.get("inventory_isolation") == "isolated",
        "native_preparation_contract",
    )
    if specialist:
        require(
            prep.get("unique_video_families") == 60
            and prep.get("actual_video_draws") == 120
            and prep.get("draws_per_sample") == 2
            and prep.get("new_source_families") == 0,
            "specialist_storage_draw_contract",
        )
    elif text_matched:
        require(
            prep.get("video_rows") == prep.get("video_families") == 60
            and prep.get("pair_count") == 30
            and prep.get("pair_metadata_only") is True
            and prep.get("nonmedia_prefix_question_equal") is True
            and prep.get("actual_draw_pair_offsets") == [0, 2, 4],
            "matched_preparation_contract",
        )
    else:
        require(
            prep.get("video_rows") == prep.get("video_families") == 60
            and prep.get("original_official_gold")
            == "unchanged; proxy is never a target or admission/quality gate",
            "native_preparation_contract",
        )
    parent = launch.get("selected_init_adapter")
    require(
        isinstance(parent, dict)
        and parent.get("weights_sha256") == CONTINUOUS_PARENT_SHA
        and launch.get("parent_selection")
        == "root_selected_f2db_balanced_retention_and_full_video_evidence"
        and final.get("init_adapter") == parent,
        "native_original_f2db_parent",
    )
    config = final["config"]
    require(
        config.get("init_adapter") == parent
        and (
            config.get("seed"),
            config.get("steps"),
            config.get("sample_count"),
            config.get("accumulation"),
            config.get("video_num_frames"),
            config.get("image_max_soft_tokens"),
            config.get("rank"),
            config.get("alpha"),
        )
        == (42, 10, 60 if specialist else 120, 12, 16, 1120, 16, 32)
        and not any(
            config.get(k, 0) for k in ("pair_loss_weight", "reference_kl_weight", "brier_weight")
        )
        and not final.get("resumed_from")
        and not final.get("invocation"),
        "native_independent_ce_budget",
    )
    if specialist:
        require(config.get("modality") == "video", "specialist_video_only")


def matched_nonmedia_sha(sample):
    prefix = sample.prefix.model_dump(mode="json")
    prefix.pop("media")
    question = sample.questions[0].model_dump(mode="json")
    question.pop("target")
    return hashlib.sha256(
        json.dumps({"prefix": prefix, "question": question}, sort_keys=True).encode()
    ).hexdigest()


def matched_pair_binding(left, right, left_program, right_program, metadata):
    from scripts.s1_text_pools import canonical_sample_sha

    require(
        left.meta.pool == right.meta.pool == "train"
        and left.meta.family_id != right.meta.family_id
        and left.prefix.media[0].sha256 != right.prefix.media[0].sha256,
        "matched_train_different_media_family",
    )
    require(
        {left.questions[0].target.value, right.questions[0].target.value} == {"yes", "no"}
        and left_program == right_program
        and matched_nonmedia_sha(left) == matched_nonmedia_sha(right),
        "matched_exact_nonmedia_program_opposite_gold",
    )
    require(
        metadata["model_nonmedia_question_sha256"] == matched_nonmedia_sha(left)
        and metadata["official_program_sha256"]
        == hashlib.sha256(json.dumps(left_program, sort_keys=True).encode()).hexdigest()
        and metadata["text_sha256"]
        == hashlib.sha256(left.questions[0].instruction.encode()).hexdigest(),
        "matched_metadata_identity",
    )
    members = {m["item_id"]: m for m in metadata["members"]}
    require(set(members) == {left.id, right.id}, "matched_pair_member_ids")
    for sample in (left, right):
        m = members[sample.id]
        require(
            m["family_id"] == sample.meta.family_id
            and m["sample_sha256"] == canonical_sample_sha(sample)
            and m["official_source_sha256"] == sample.meta.source_sha256
            and m["media_sha256"] == sample.prefix.media[0].sha256
            and m["original_gold"] == sample.questions[0].target.value,
            "matched_source_member",
        )


def native_event_bindings(root, prepared, prep, final, runtime, *, text_matched=False):
    import random
    import zipfile
    from collections import Counter

    from scripts.s1_text_pools import canonical_sample_sha

    prefix = "s4-video-text-matched" if text_matched else "s4-native-event-condition"
    launch, lb = read_json(root / "reports" / (prefix + "-launch-001.json"))
    native_parent_binding(prep, launch, final, text_matched=text_matched)
    native_parent_binding(prep, launch, runtime, text_matched=text_matched)
    require(
        launch["status"] == final["status"] and launch["exit_code"] == final["exit_code"],
        "native_launch_terminal",
    )
    require(
        sha(prepared / "report.json")
        == launch["prepared_report_sha256"]
        == (MATCHED_PREP_SHA if text_matched else NATIVE_PREP_SHA)
        and final["data_sha256"] == runtime["data_sha256"] == prep["output_sha256"]["train.jsonl"]
        and (text_matched or final["data_sha256"] == NATIVE_DATA_SHA),
        "native_fixed_preparation_data",
    )
    bindings = {lb["path"]: lb["sha256"]}

    def bound(path, expected):
        require(sha(path) == expected, "native_bound_source")
        bindings[str(path)] = expected

    for name, expected in prep["output_sha256"].items():
        require(Path(name).name == name, "native_safe_output_name")
        bound(prepared / name, expected)
    base = root / (
        "artifacts/s4-pilot-exposure-017-retry-001"
        if text_matched
        else "artifacts/s4-pilot-exposure-015"
    )
    base_sha = MATCHED_BASE_SHA if text_matched else NATIVE_BASE_SHA
    require(
        prep["exposure_base"]["report_sha256"]
        == launch["latest_exposure_report_sha256"]
        == base_sha,
        "native_base015",
    )
    bound(base / "report.json", base_sha)
    bound(base / "inventory-manifest.json", prep["exposure_base"]["manifest_sha256"])
    recovered_path = root / "reports/s4-continuous-main-pool-launch-recovered-002.json"
    recovered = json.loads(recovered_path.read_bytes())
    original = recovered["original_failed_launch"]
    bound(Path(original["path"]), original["sha256"])

    sources = root.parent / "s4-native-event-cpu-001"
    bound(
        sources
        / (
            "prepare-video-text-matched-pilot.py"
            if text_matched
            else "prepare-native-event-condition-pilot.py"
        ),
        prep["helper_sha256"],
    )
    for name, expected in prep["reused_helper_sha256"].items():
        require(
            name in {"inspect-temporal-order-contrast", "prepare-claim-pair-pilot"},
            "native_reused_helper_scope",
        )
        bound(sources / (name + ".py"), expected)
    for relative, field in [
        (
            "artifacts/s4-temporal-coverage-video500-001/receipt.json",
            "parent_full_video_receipt_sha256",
        ),
        (
            "artifacts/s4-temporal-coverage-old-retention-001/receipt.json",
            "retention_report_sha256",
        ),
        ("artifacts/s4-temporal-coverage-video150-001/receipt.json", "video_screen_sha256"),
        (
            "artifacts/s4-temporal-coverage-original100-001/receipt-recovered-002.json",
            "cross_screen_sha256",
        ),
    ]:
        bound(root / relative, launch[field])
    require(launch["source_commit"] == "3fe0ab2", "native_training_core")
    bound(root.parent / "omnijev-s4-claim-pair-001.tar", launch["source_archive_sha256"])
    official = root.parent / "lite-omnijev/source-audit/clevrer-official/files"
    for relative, key in [
        ("questions/train.json", "questions"),
        ("annotations/train/annotation_train.zip", "annotations"),
        ("README.txt", "readme"),
    ]:
        bound(official / relative, prep["source_sha256"][key])
    pool = root / "artifacts/s3-pools-video-001/run-a/train.jsonl"
    bound(pool, prep["source_sha256"]["train"])
    rows = canonical_rows(prepared / "train.jsonl")
    originals = canonical_rows(pool)
    ledger = [
        json.loads(line)
        for line in (prepared / "allocation-ledger.jsonl").read_bytes().splitlines()
    ]
    require(
        len(rows) == len(ledger) == len({row["item_id"] for row in ledger}) == 120,
        "native_ledger120",
    )
    for entry in ledger:
        s = Sample.model_validate(rows[entry["item_id"]])
        require(
            entry["pool"] == "train"
            and entry["family_id"] == s.meta.family_id
            and entry["derived_sample_sha256"] == canonical_sample_sha(s),
            "native_exact_allocation",
        )
    video = {k: v for k, v in rows.items() if v["meta"]["modality"] == ["video"]}
    require(
        len(video) == len({v["meta"]["family_id"] for v in video.values()}) == 60,
        "native60_original_families",
    )
    require(all(originals.get(k) == v for k, v in video.items()), "native_official_gold_unchanged")
    require(
        Counter(v["questions"][0]["target"]["value"] for v in video.values())
        == {"yes": 30, "no": 30},
        "native_gold_balance",
    )
    replay_dir = root / "artifacts/s4-temporal-curriculum-pilot-001"
    bound(replay_dir / "report.json", prep["replay_report_sha256"])
    replay = canonical_rows(replay_dir / "train.jsonl")
    require(
        {k: v for k, v in rows.items() if k not in video}
        == {k: v for k, v in replay.items() if v["meta"]["modality"] != ["video"]},
        "native_same60_replay",
    )
    if text_matched:
        metadata = json.loads((prepared / "matched-pair-metadata.json").read_bytes())
        require(
            metadata["metadata_only"] is True
            and metadata["target_source"]
            == "original official gold only; pairs are metadata, no pairloss",
            "matched_metadata_only",
        )
        byid = {}
        for pair in metadata["rows"]:
            require(
                pair.get("metadata_only") is True and len(pair["members"]) == 2, "matched_pair_size"
            )
            for member in pair["members"]:
                require(member["item_id"] not in byid, "matched_duplicate_member")
                byid[member["item_id"]] = {
                    **member,
                    "program_sha256": pair["official_program_sha256"],
                    "pair_metadata": pair,
                }
        require(len(metadata["rows"]) == 30, "matched_pair_count")
    else:
        metadata = json.loads((prepared / "native-program-metadata.json").read_bytes())
        require(
            metadata["metadata_only"] is True
            and metadata["target_source"] == "original official gold only",
            "native_proxy_metadata_only",
        )
        byid = {m["item_id"]: m for m in metadata["rows"]}
    require(set(byid) == set(video), "native_metadata_members")
    programs = {}
    scenes = {
        s["scene_index"]: s for s in json.loads((official / "questions/train.json").read_bytes())
    }
    with zipfile.ZipFile(official / "annotations/train/annotation_train.zip") as archive:
        for ident, row in video.items():
            s = Sample.model_validate(row)
            m = byid[ident]
            q = official_video_question(s, scenes)
            require(
                m["sample_sha256"] == canonical_sample_sha(s)
                and m["program_sha256"]
                == hashlib.sha256(json.dumps(q["program"], sort_keys=True).encode()).hexdigest(),
                "native_program_metadata_sha",
            )
            programs[ident] = q["program"]
            if not text_matched:
                info = archive.getinfo(m["annotation_member"])
                require(
                    0 < info.file_size < 5_000_000
                    and not info.flag_bits & 1
                    and hashlib.sha256(archive.read(info)).hexdigest() == m["annotation_sha256"],
                    "native_annotation_binding",
                )
    mapping = version_mapping(
        Manifest.model_validate_json((prepared / "inventory-manifest.json").read_bytes())
    )
    config = {k: v for k, v in final["config"].items() if k != "init_adapter"}
    require(
        LoopConfig.model_validate(config)
        == LoopConfig.model_validate_json((prepared / "config-ce.json").read_bytes()),
        "native_config_exact",
    )
    arranged = [
        Sample.model_validate_json(line)
        for line in (prepared / "train.jsonl").read_bytes().splitlines()
    ]
    order = list(range(120))
    random.Random(42).shuffle(order)
    for step in range(10):
        block = [arranged[i] for i in order[step * 12 : step * 12 + 12]]
        require(
            Counter(mapping[s.meta.data_version] for s in block)
            == {"video": 6, "crossmodal": 2, "audio": 1, "image": 1, "screenshot": 1, "text": 1},
            "native_actual_block",
        )
        require(
            Counter(s.questions[0].target.value for s in block if s.meta.modality == ["video"])
            == {"yes": 3, "no": 3},
            "native_actual_block_gold",
        )
        if text_matched:
            for offset in (0, 2, 4):
                left, right = block[offset : offset + 2]
                matched_pair_binding(
                    left,
                    right,
                    programs[left.id],
                    programs[right.id],
                    byid[left.id]["pair_metadata"],
                )
                require(
                    byid[right.id]["pair_metadata"] == byid[left.id]["pair_metadata"],
                    "matched_actual_pair_adjacency",
                )
    for name, expected in [
        ("adapter_model.safetensors", final["init_adapter"]["weights_sha256"]),
        ("adapter_config.json", final["init_adapter"]["config_sha256"]),
    ]:
        bound(Path(final["init_adapter"]["path"]) / name, expected)
    if final["status"] == "completed":
        require(final["adapter_sha256"] == runtime["adapter_sha256"], "native_final_adapter")
        bound(
            root / "artifacts" / (prefix + "-train-001") / "adapter/adapter_model.safetensors",
            final["adapter_sha256"],
        )
    return bindings, mapping


def specialist_bindings(root, prepared, prep, final, runtime):
    """Reuse the complete native official-source audit; specialist only changes draws/config."""
    from collections import Counter

    from scripts.s1_text_pools import canonical_sample_sha
    from train.trainer import data_bytes

    native_dir = root / "artifacts/s4-native-event-condition-pilot-001"
    native_prep = json.loads((native_dir / "report.json").read_bytes())
    native_final = json.loads(
        (root / "reports/s4-native-event-condition-training-001.json").read_bytes()
    )
    native_runtime = json.loads(
        (root / "artifacts/s4-native-event-condition-train-001/run.json").read_bytes()
    )
    bindings, mapping = native_event_bindings(
        root, native_dir, native_prep, native_final, native_runtime
    )
    launch, lb = read_json(root / "reports/s4-video-specialist-launch-001.json")
    bindings[lb["path"]] = lb["sha256"]
    native_parent_binding(prep, launch, final, specialist=True)
    native_parent_binding(prep, launch, runtime, specialist=True)
    require(
        launch["status"] == final["status"] and launch["exit_code"] == final["exit_code"],
        "specialist_launch_terminal",
    )
    require(
        sha(prepared / "report.json") == launch["prepared_report_sha256"] == SPECIALIST_PREP_SHA
        and final["data_sha256"] == runtime["data_sha256"] == SPECIALIST_DATA_SHA,
        "specialist_frozen_data",
    )

    def bound(path, expected):
        require(sha(path) == expected, "specialist_bound_input")
        bindings[str(path)] = expected

    for name, expected in prep["output_sha256"].items():
        require(Path(name).name == name, "specialist_safe_output_name")
        bound(prepared / name, expected)
    require(
        prep["source_report_sha256"] == NATIVE_PREP_SHA
        and prep["source_data_sha256"] == NATIVE_DATA_SHA
        and prep["exposure_base"]["report_sha256"]
        == launch["latest_exposure_report_sha256"]
        == SPECIALIST_BASE_SHA,
        "specialist_native_source_and_base016",
    )
    base = root / "artifacts/s4-pilot-exposure-016"
    bound(base / "report.json", SPECIALIST_BASE_SHA)
    bound(base / "inventory-manifest.json", prep["exposure_base"]["manifest_sha256"])
    sources = root.parent / "s4-native-event-cpu-001"
    bound(sources / "prepare-video-specialist-pilot.py", prep["helper_sha256"])
    bound(sources / "prepare-native-event-condition-pilot.py", prep["native_helper_sha256"])
    require(launch["source_commit"] == "3fe0ab2", "specialist_old_core")
    bound(root.parent / "omnijev-s4-claim-pair-001.tar", launch["source_archive_sha256"])
    rows = canonical_rows(prepared / "train.jsonl")
    native_rows = canonical_rows(native_dir / "train.jsonl")
    require(
        rows == {k: v for k, v in native_rows.items() if v["meta"]["modality"] == ["video"]}
        and len(rows) == 60,
        "specialist_same_native60",
    )
    ledger = [
        json.loads(line)
        for line in (prepared / "allocation-ledger.jsonl").read_bytes().splitlines()
    ]
    require(len(ledger) == len({x["item_id"] for x in ledger}) == 60, "specialist_ledger60")
    for entry in ledger:
        sample = Sample.model_validate(rows[entry["item_id"]])
        require(
            entry["pool"] == "train"
            and entry["family_id"] == sample.meta.family_id
            and entry["derived_sample_sha256"] == canonical_sample_sha(sample),
            "specialist_exact_ledger",
        )
    actual = specialist_draws(
        [
            Sample.model_validate_json(line)
            for line in (prepared / "train.jsonl").read_bytes().splitlines()
        ]
    )
    require(
        hashlib.sha256(data_bytes(actual)).hexdigest() == prep["draw_order_canonical_sha256"],
        "specialist_draw_order_sha",
    )
    recipe = json.loads((prepared / "draw-recipe.json").read_bytes())
    require(
        recipe["actual_draws"] == 120
        and recipe["unique_samples"] == 60
        and len(recipe["draws"]) == 120,
        "specialist_recipe_size",
    )
    for i, (sample, draw) in enumerate(zip(actual, recipe["draws"], strict=True)):
        require(
            draw["draw"] == i
            and draw["step"] == i // 12
            and draw["offset"] == i % 12
            and draw["item_id"] == sample.id
            and draw["sample_sha256"] == canonical_sample_sha(sample),
            "specialist_actual_draw_binding",
        )
    require(Counter(s.id for s in actual) == {k: 2 for k in rows}, "specialist_exact_two_passes")
    config = {k: v for k, v in final["config"].items() if k != "init_adapter"}
    require(
        LoopConfig.model_validate(config)
        == LoopConfig.model_validate_json((prepared / "config-ce.json").read_bytes()),
        "specialist_config_exact",
    )
    if final["status"] == "completed":
        require(final["adapter_sha256"] == runtime["adapter_sha256"], "specialist_final_adapter")
        bound(
            root / "artifacts/s4-video-specialist-train-001/adapter/adapter_model.safetensors",
            final["adapter_sha256"],
        )
    return bindings, mapping


def specialist_draws(samples):
    """Sixty unchanged TRAIN rows, seed42 shuffle cycled twice, no alias inflation."""
    import random

    require(
        len(samples)
        == len({s.id for s in samples})
        == len({s.meta.family_id for s in samples})
        == 60
        and all(s.meta.pool == "train" and s.meta.modality == ["video"] for s in samples),
        "specialist60_video_train",
    )
    order = list(range(60))
    random.Random(42).shuffle(order)
    return [samples[order[i % 60]] for i in range(120)]


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
            for k, v in {
                "seed": 42,
                "steps": 10,
                "accumulation": 12,
                "sample_count": 60 if name == "s4-video-specialist" else 120,
            }.items()
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
    native_event_condition: bool = False,
    video_specialist: bool = False,
    video_text_matched: bool = False,
) -> tuple[Path, dict[str, str] | None]:
    if native_event_condition or video_specialist or video_text_matched:
        exposure = root / (
            "artifacts/s4-pilot-exposure-017-retry-001"
            if video_text_matched
            else "artifacts/s4-pilot-exposure-016"
            if video_specialist
            else "artifacts/s4-pilot-exposure-015"
        )
        report, binding = read_json(exposure / "report.json")
        require(
            binding["sha256"]
            == (
                MATCHED_BASE_SHA
                if video_text_matched
                else SPECIALIST_BASE_SHA
                if video_specialist
                else NATIVE_BASE_SHA
            )
            and report.get("exit_code") == 0
            and report.get("unknown_update_bearing_inputs") == 0,
            "native_base015_passed",
        )
        require(
            all(sha(Path(p)) == h for p, h in report["all_bound_input_sha256"].items()),
            "native_prior_bindings_mutated",
        )
        path = exposure / "inventory-manifest.json"
        require(
            sha(path) == report["output_sha256"]["inventory-manifest.json"], "native_base_inventory"
        )
        return path, binding
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
    native_event_condition: bool = False,
    video_specialist: bool = False,
    video_text_matched: bool = False,
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
            native_event_condition=native_event_condition,
            video_specialist=video_specialist,
            video_text_matched=video_text_matched,
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
            polarity_arm is not None
            or coverage_segment003
            or native_event_condition
            or video_specialist
            or video_text_matched
        ) and base_preparation_binding is not None:
            prior_base = json.loads(Path(base_preparation_binding["path"]).read_bytes())
            bindings.update(prior_base["all_bound_input_sha256"])
        prior_base = (
            json.loads(Path(base_preparation_binding["path"]).read_bytes())
            if native_event_condition or video_specialist or video_text_matched
            else {}
        )
        mutable_continuous = {}
        for invocation in prior_base.get("continuous_invocations", []):
            observed = invocation["runtime_report_observation"]
            mutable_continuous[observed["path"]] = observed["sha256"]
        require(
            all(sha(Path(p)) == h for p, h in mutable_continuous.items()),
            "native_continuous_runtime_changed",
        )
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
            native_event_condition=native_event_condition,
            video_specialist=video_specialist,
            video_text_matched=video_text_matched,
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
            is_specialist = video_specialist and name == "s4-video-specialist"
            is_matched = video_text_matched and name == "s4-video-text-matched"
            is_native = (
                (native_event_condition and name == "s4-native-event-condition")
                or is_specialist
                or is_matched
            )
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
                or is_native
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
            if is_native:
                binder = specialist_bindings if is_specialist else native_event_bindings
                extra_bindings, extra_mapping = binder(
                    root,
                    preparation_dir,
                    prep_report,
                    final,
                    runtime_report,
                    **({"text_matched": True} if is_matched else {}),
                )
                bindings.update(extra_bindings)
                mappings.update(extra_mapping)
            elif is_order:
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
                or is_native
            ):
                require(comparison["rows"] == (60 if is_specialist else 120), "claim_runtime_count")
            require(
                all(version in mappings for version in comparison["data_versions"]),
                "unmapped_data_version",
            )
            artifact = {
                **runtime_binding,
                "cohorts_by_data_version": {v: mappings[v] for v in comparison["data_versions"]},
                "exposure": "current"
                if is_polarity or is_segment003 or is_order or is_native
                else "historical",
                "expected_pool": "train",
            }
            appended = False
            if final["steps_completed"] > 0 and runtime_binding["sha256"] not in hashes:
                require(
                    runtime_binding["sha256"] not in all_hashes, "canonical_sha_nontrain_collision"
                )
                if is_polarity or is_segment003 or is_order or is_native:
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
                    and not (is_polarity or is_segment003 or is_order or is_native),
                    **(
                        {"appended_current_train": appended}
                        if is_polarity or is_segment003 or is_order or is_native
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
            if str(path) not in mutable_continuous:
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
        require(
            all(sha(Path(p)) == h for p, h in mutable_continuous.items()),
            "native_continuous_runtime_changed",
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
        if native_event_condition or video_specialist or video_text_matched:
            report.update(
                native_event_condition_included=True,
                video_specialist_included=video_specialist
                or bool(prior_base.get("video_specialist_included")),
                base_exposure_report=base_preparation_binding,
                continuous_invocations=prior_base["continuous_invocations"],
                continuous_actual_global_draws_seen=prior_base["actual_global_draws_seen"],
                actual_global_draws_seen=prior_base["actual_global_draws_seen"],
                global_pool_draw_capacity=prior_base["global_pool_draw_capacity"],
                native_actual_draws_seen=(
                    prior_base["native_actual_draws_seen"]
                    if video_specialist or video_text_matched
                    else pilots[0]["steps_completed"] * 12
                ),
                continuous_pool_capacity_not_exposure=True,
            )
        if video_text_matched:
            report.update(
                video_text_matched_included=True,
                text_matched_actual_draws_seen=pilots[0]["steps_completed"] * 12,
                text_matched_pairs=30,
                specialist_actual_draws_seen=prior_base["specialist_actual_draws_seen"],
                specialist_storage_samples=prior_base["specialist_storage_samples"],
                specialist_unique_families=prior_base["specialist_unique_families"],
                specialist_new_source_families=0,
            )
        if video_specialist:
            report.update(
                specialist_actual_draws_seen=pilots[0]["steps_completed"] * 12,
                specialist_storage_samples=60,
                specialist_unique_families=60,
                specialist_new_source_families=0,
            )
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


MATCHED_CONTINUOUS_PREP_SHA = "2c51e16cb88ab91dfb45dca84a2325b371ad259dce522a01fa50e042090cfa87"
MATCHED_CONTINUOUS_DATA_SHA = "17cb9a0066639dafd1feb21e3819cc38a367c86e78ce2a2907a5047d6cb437aa"
MATCHED_CONTINUOUS_CONFIG_SHA = "d8e9e5b0fc8427b32f39165c58486bfd93123fea1a584b5c0f8508efcb389414"
MATCHED_CONTINUOUS_BASE_SHA = "b70aa762a2d66492b53d5a44a03c1877d9d3fcd2861e7c5ad43c77b84715e659"


def official_video_question(sample, scenes):
    """Existing official question/program/source identity, shared by both paired recipes."""
    from scripts.s0_media_dev import source_hash_for_video

    index = int(sample.meta.family_id.split(":")[1])
    question = next(
        q
        for q in scenes[index]["questions"]
        if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
    )
    require(
        question["question"] == sample.questions[0].instruction
        and question["answer"].casefold() == sample.questions[0].target.value
        and source_hash_for_video({"scene_index": index}, question, sample.prefix.media[0].sha256)
        == sample.meta.source_sha256,
        "native_official_question_source",
    )
    return question


def continuous_matched_bindings(root, prepared, prep, *, program_stratified=False):
    """The second frozen profile reuses original gold and matched-pair identity checks."""
    import random
    from collections import Counter

    from scripts.s1_text_pools import canonical_sample_sha
    from train.trainer import data_bytes

    if program_stratified:
        require(
            prep["video_rows"] == prep["video_families"] == prep["replay_draws"] == 600
            and prep["global_steps"] == 100
            and prep["invocation_steps_initial"] == 10
            and prep["automatic_full_run_authorized"] is False,
            "program_stratified_recipe",
        )
    else:
        require(
            prep["pair_count"] == 300
            and prep["video_rows"] == prep["video_families"] == 600
            and prep["global_steps"] == 100
            and prep["invocation_steps_initial"] == 10
            and prep["actual_draw_pair_offsets"] == [0, 2, 4]
            and prep["pair_metadata_only"] is True
            and prep["automatic_full_run_authorized"] is False,
            "continuous_matched_recipe",
        )
    bindings = {}

    def bound(path, expected):
        require(sha(path) == expected, "continuous_matched_source_sha")
        bindings[str(path)] = expected

    sources = root.parent / "s4-native-event-cpu-001"
    bound(
        sources
        / (
            "prepare-program-stratified-pool.py"
            if program_stratified
            else "prepare-continuous-matched-pool.py"
        ),
        prep["helper_sha256"],
    )
    for name, expected in prep["reused_helper_sha256"].items():
        require(
            name in {"inspect-temporal-order-contrast", "prepare-claim-pair-pilot"},
            "matched_reused_helper",
        )
        bound(sources / (name + ".py"), expected)
    official = root.parent / "lite-omnijev/source-audit/clevrer-official/files"
    for relative, key in [
        ("questions/train.json", "questions"),
        ("README.txt", "readme"),
        ("annotations/train/annotation_train.zip", "annotations"),
    ]:
        bound(official / relative, prep["source_sha256"][key])
    pool = root / "artifacts/s3-pools-video-001/run-a/train.jsonl"
    bound(pool, prep["source_sha256"]["train"])
    originals = canonical_rows(pool)
    rows = canonical_rows(prepared / "train.jsonl")
    video = {k: v for k, v in rows.items() if v["meta"]["modality"] == ["video"]}
    require(
        len(rows) == 1200
        and len(video) == len({v["meta"]["family_id"] for v in video.values()}) == 600
        and all(originals.get(k) == v for k, v in video.items()),
        "continuous_matched_original600",
    )
    replay = (
        root
        / "artifacts"
        / (
            "s4-continuous-matched-pool-001"
            if program_stratified
            else "s4-continuous-main-pool-001"
        )
    )
    bound(replay / "report.json", prep["replay_report_sha256"])
    require(
        prep["replay_report_sha256"]
        == (MATCHED_CONTINUOUS_PREP_SHA if program_stratified else CONTINUOUS_PREP_SHA),
        "continuous_matched_fixed_replay",
    )
    rr = json.loads((replay / "report.json").read_bytes())
    for name in ("train.jsonl", "components.jsonl"):
        bound(replay / name, rr["output_sha256"][name])
    replay_rows = canonical_rows(replay / "train.jsonl")
    require(
        {k: v for k, v in rows.items() if k not in video}
        == {k: v for k, v in replay_rows.items() if v["meta"]["modality"] != ["video"]}
        and sha(prepared / "components.jsonl") == sha(replay / "components.jsonl"),
        "continuous_matched_replay_canonical",
    )
    ledger = [
        json.loads(line)
        for line in (prepared / "allocation-ledger.jsonl").read_bytes().splitlines()
    ]
    require(
        len(ledger) == len({entry["item_id"] for entry in ledger}) == 1200, "matched_ledger1200"
    )
    for entry in ledger:
        sample = Sample.model_validate(rows[entry["item_id"]])
        require(
            entry["pool"] == "train"
            and entry["family_id"] == sample.meta.family_id
            and entry["derived_sample_sha256"] == canonical_sample_sha(sample),
            "matched_exact_ledger",
        )
    scenes = {
        scene["scene_index"]: scene
        for scene in json.loads((official / "questions/train.json").read_bytes())
    }
    programs = {
        k: official_video_question(Sample.model_validate(v), scenes)["program"]
        for k, v in video.items()
    }
    if program_stratified:
        strata = program_strata_binding(prepared, prep, video, programs)
        bindings.update(program_exclusion_bindings(root, prep, video))
    else:
        metadata = json.loads((prepared / "matched-pair-metadata.json").read_bytes())
        require(
            metadata["metadata_only"] is True
            and len(metadata["rows"]) == 300
            and metadata["target_source"]
            == "original official gold only; pairs are metadata, no pairloss",
            "continuous_matched_metadata300",
        )
        pairs = {}
        for pair in metadata["rows"]:
            require(
                pair["metadata_only"] is True and len(pair["members"]) == 2, "matched_pair_size"
            )
            ids = [m["item_id"] for m in pair["members"]]
            require(all(k in video and k not in pairs for k in ids), "matched_duplicate_member")
            matched_pair_binding(
                *(Sample.model_validate(video[k]) for k in ids), *(programs[k] for k in ids), pair
            )
            pairs.update({k: pair for k in ids})
        require(set(pairs) == set(video), "matched_all600_members")
    arranged = [
        Sample.model_validate_json(line)
        for line in (prepared / "train.jsonl").read_bytes().splitlines()
    ]
    order = list(range(1200))
    random.Random(42).shuffle(order)
    actual = [arranged[i] for i in order]
    require(
        hashlib.sha256(data_bytes(actual)).hexdigest() == prep["draw_order_canonical_sha256"],
        "matched_order_sha",
    )
    mapping = version_mapping(
        Manifest.model_validate_json((prepared / "inventory-manifest.json").read_bytes())
    )
    recipe = json.loads((prepared / "draw-recipe.json").read_bytes())
    require(recipe["global_steps"] == 100 and len(recipe["draws"]) == 1200, "matched_global_recipe")
    for index, (sample, draw) in enumerate(zip(actual, recipe["draws"], strict=True)):
        require(
            draw["draw"] == index
            and draw["step"] == index // 12
            and draw["offset"] == index % 12
            and draw["item_id"] == sample.id
            and draw["sample_sha256"] == canonical_sample_sha(sample)
            and draw["cohort"] == mapping[sample.meta.data_version],
            "matched_actual_draw_binding",
        )
    for step in range(100):
        block = actual[12 * step : 12 * step + 12]
        require(
            Counter(d["cohort"] for d in recipe["draws"][12 * step : 12 * step + 12])
            == {"video": 6, "crossmodal": 2, "audio": 1, "image": 1, "screenshot": 1, "text": 1},
            "matched_block_quota",
        )
        if program_stratified:
            require(
                Counter(strata[s.id] for s in block[:6])
                == dict(
                    zip(
                        PROGRAM_GROUPS, (2, 2, 1, 1) if step % 2 == 0 else (1, 1, 2, 2), strict=True
                    )
                ),
                "program_actual_step_strata",
            )
            require(
                Counter(s.questions[0].target.value for s in block[:6]) == {"yes": 3, "no": 3},
                "program_actual_step_polarity",
            )
            require(
                all(
                    d.get("program_stratum") == strata[s.id]
                    and d.get("target") == s.questions[0].target.value
                    for s, d in zip(
                        block[:6], recipe["draws"][12 * step : 12 * step + 6], strict=True
                    )
                ),
                "program_actual_draw_metadata",
            )
        else:
            for offset in (0, 2, 4):
                left, right = block[offset : offset + 2]
                require(
                    left.id in pairs and right.id in pairs and pairs[left.id] == pairs[right.id],
                    "matched_actual_pair_adjacency",
                )
    return bindings


PROGRAM_GROUPS = ("before_after", "event_motion", "endpoint_motion", "exist")


def program_strata_binding(prepared, prep, video, programs):
    from collections import Counter

    from scripts.s1_text_pools import canonical_sample_sha

    metadata = json.loads((prepared / "program-strata-metadata.json").read_bytes())
    require(
        metadata["metadata_only"] is True
        and len(metadata["rows"]) == 600
        and metadata["target_source"] == "original official gold only; exclusive program strata",
        "program_metadata600",
    )
    strata = {}
    counts = Counter()
    for row in metadata["rows"]:
        item = row["item_id"]
        require(item in video and item not in strata, "program_unique_member")
        sample = Sample.model_validate(video[item])
        program = programs[item]
        require(
            isinstance(program, list) and program and all(isinstance(p, str) for p in program),
            "program_official_postfix",
        )
        group = (
            "before_after"
            if "filter_before" in program or "filter_after" in program
            else "event_motion"
            if "events" in program and "get_frame" in program
            else "endpoint_motion"
            if "get_frame" in program
            else "exist"
        )
        require(group != "exist" or program[-1] == "exist", "program_exist_terminal")
        gold = sample.questions[0].target.value
        require(
            row
            == {
                "item_id": item,
                "family_id": sample.meta.family_id,
                "program_stratum": group,
                "official_program_sha256": hashlib.sha256(
                    json.dumps(program, sort_keys=True).encode()
                ).hexdigest(),
                "official_source_sha256": sample.meta.source_sha256,
                "media_sha256": sample.prefix.media[0].sha256,
                "sample_sha256": canonical_sample_sha(sample),
                "original_gold": gold,
                "has_collision_operator": "filter_collision" in program,
            },
            "program_exact_metadata",
        )
        strata[item] = group
        counts[group, gold] += 1
    require(
        set(strata) == set(video)
        and counts == Counter({(g, y): 75 for g in PROGRAM_GROUPS for y in ("yes", "no")}),
        "program_four_balanced_strata",
    )
    require(
        prep["stratum_counts"] == {g: {"yes": 75, "no": 75} for g in PROGRAM_GROUPS},
        "program_report_strata",
    )
    return strata


def program_exclusion_bindings(root, prep, video):
    """Reconstruct only the explicit source-bound exclusion scope, never global unseen claims."""
    import random

    bindings = {}

    def bound(path, expected):
        require(sha(path) == expected, "program_exclusion_input_sha")
        bindings[str(path)] = expected

    def pool(name, expected):
        directory = root / "artifacts" / name
        receipt = json.loads((directory / "report.json").read_bytes())
        bound(directory / "report.json", expected)
        bound(directory / "train.jsonl", receipt["output_sha256"]["train.jsonl"])
        return list(canonical_rows(directory / "train.jsonl").values())

    lineage_path = root / "reports/s4-ce0-video-exposure-001.json"
    bound(lineage_path, "cb9574e06fc9950198f703ab9107d6d2d42824a5d2399ded2a26a146dd88cbfa")
    lineage = json.loads(lineage_path.read_bytes())
    for stage in lineage["stages"]:
        for key in ("report", "data"):
            bound(Path(stage[key]), stage[key + "_sha256"])
    excluded = set(lineage["lineage_unique_video_families"])
    names = [
        "s4-temporal-coverage-pilot-001",
        "s4-train-learnability-001",
        "s4-native-event-condition-pilot-001",
        "s4-video-text-matched-pilot-001",
    ]
    prior_pins = {
        names[0]: "1e7f877e04ea311d29ebc1fe1711f31aa2cc91034f8794b666850bcf5f2fff19",
        names[1]: prep["exclusions"]["learnability_report_sha256"],
        names[2]: "401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696",
        names[3]: "3e764378147c06816a3f14fe71967f53d83fb50871237e094ceb19587419db93",
    }
    for name in names:
        excluded.update(
            r["meta"]["family_id"]
            for r in pool(name, prior_pins[name])
            if r["meta"]["modality"] == ["video"]
        )
    continuous = pool("s4-continuous-main-pool-001", CONTINUOUS_PREP_SHA)
    order = list(range(1200))
    random.Random(42).shuffle(order)
    excluded.update(
        continuous[i]["meta"]["family_id"]
        for i in order[:240]
        if continuous[i]["meta"]["modality"] == ["video"]
    )
    for row in prep["exclusions"]["specified_prior_pilots"]:
        excluded.update(
            r["meta"]["family_id"]
            for r in pool(row["artifact"], row["report_sha256"])
            if r["meta"]["modality"] == ["video"]
        )
    require(
        len(excluded) == prep["exclusions"]["final_union"] == 1430
        and not excluded.intersection(r["meta"]["family_id"] for r in video.values())
        and prep["exclusions"]["scope"]
        == "explicit source-bound pilot-family exclusions; not globally unseen",
        "program_explicit1430_exclusions",
    )
    return bindings


def program_training_bindings(root, launch, config, base, history, start, end):
    """Each invocation needs a fresh root decision inside the same frozen window."""
    profile = continuous_profile(program_stratified=True)
    bindings = {}

    def bound(path, expected):
        require(sha(path) == expected, "program_root_binding_sha")
        bindings[str(path)] = expected

    original = root / "artifacts/s4-continuous-matched-pool-001/config-ce.json"
    bound(original, MATCHED_CONTINUOUS_CONFIG_SHA)
    require(
        config == {**json.loads(original.read_bytes()), "checkpoint_every": 5},
        "program_ce_config_delta",
    )
    archive = root.parent / "omnijev-video-pair-code-7b75f9f.tar"
    archive_sha = "49b11d244dc483a59895695468697e05ee4d45ff8b491cad377724137b15d32d"
    bound(archive, archive_sha)
    require(
        launch["source_commit"] == "7b75f9f" and launch["source_archive_sha256"] == archive_sha,
        "program_runtime_source",
    )
    launcher = root.parent / "launch-program-stratified-pool-001.py"
    bound(launcher, "8ff309225773413e3c2a8b0494c891db77216229b36ee787c125d521bab3a4bc")
    helper = root.parent / "s4-native-event-cpu-001/prepare-program-stratified-pool.py"
    bound(helper, "fcd52c264ccabf0cb2120402836089aa988ff298b485dd272487f46aa92c893a")
    decision_path = root / f"reports/s4-program-stratified-decision-step{start + 10:04d}.json"
    bound(decision_path, launch["decision_sha256"])
    if not start:
        require(
            sha(decision_path)
            == "9a8bb1a6641df952cec0c7a55754aaa479d417a4d328874a205a8080ffa11b5a",
            "program_initial_decision_pin",
        )
    decision = json.loads(decision_path.read_bytes())
    expected = {
        "decision": "approve_next10",
        "start_step": start,
        "end_step": start + 10,
        "max_total_steps": 40,
        "invocation_steps": 10,
        "max_seconds": 240,
        "prepared_report_sha256": profile["prep_sha"],
        "initializer_sha256": CONTINUOUS_PARENT_SHA,
        "strategy_sha256": "43234bfcdc1cadbf80087cdc3fe55c238549000124f0ab35c0439dc9d1f3a657",
        "automatic_resume": False,
        "window_start_utc": "2026-10-16T06:06:49Z",
        "window_deadline_utc": "2026-10-16T06:26:49Z",
    }
    require(
        start in (0, 10)
        and end <= start + 10
        and all(decision.get(k) == v for k, v in expected.items()),
        "program_explicit_bounded_decision",
    )
    window = {
        k: decision[k]
        for k in (
            "window_start_utc",
            "window_deadline_utc",
            "prepared_report_sha256",
            "initializer_sha256",
            "max_total_steps",
        )
    }
    window_path = root / "reports/s4-program-stratified-window-001.json"
    bound(window_path, sha(window_path))
    require(
        json.loads(window_path.read_bytes()) == launch["window"] == window,
        "program_original_window",
    )
    if start:
        previous_launch = history[-1]["launch_report"]
        require(
            decision.get("learning_observed") is True
            and decision["previous_launch_sha256"] == previous_launch["sha256"]
            and decision["latest_exposure_report_sha256"]
            == launch["exposure_binding"]["base_exposure_report"]["sha256"]
            and decision["latest_exposure_report"]
            == launch["exposure_binding"]["base_exposure_report"]["path"],
            "program_fresh_resume_review",
        )
        screen_path = root / f"reports/s4-program-stratified-dev50-step{start:04d}.json"
        bound(screen_path, decision["screening_report_sha256"])
        screen = json.loads(screen_path.read_bytes())
        previous = json.loads(Path(history[-1]["runtime_report_snapshot"]["path"]).read_bytes())
        require(
            screen["status"] == "completed"
            and screen["exit_code"] == 0
            and screen["step"] == start
            and screen["paired_input_hashes_verified"] == 50
            and screen["adapter_sha256"] == previous["adapter_sha256"]
            and screen["candidate"]["count"] == 50
            and 0.64 <= screen["candidate"]["accuracy"] <= 1
            and 0 <= screen["candidate"]["brier"] <= 0.46748601999097937,
            "program_same_step_screen",
        )
        require(
            json.loads(Path(previous_launch["path"]).read_bytes())["window"] == window,
            "program_resume_window_unchanged",
        )
    return bindings


def continuous_profile(
    matched=False, rank64=False, video_objective=False, program_stratified=False
):
    """Frozen, independent recipes; objective changes cannot enter the CE profiles."""
    if program_stratified:
        return {
            "prep_sha": "ac2f17fc3f5417e93e04d15292763384858f5e3ba1c4de533ca6f5e57d0aaf99",
            "data_sha": "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f",
            "config_sha": "326b406eb23ce47c4854afe64d535a95ad91be94a68844109329e43872c6aaea",
            "base_sha": "960a6b661f6414a514fa3274979b3a979669544979632251f52838f1f882edcb",
            "history_key": "program_stratified_invocations",
            "name": "s4-program-stratified",
        }
    if video_objective:
        return {
            "prep_sha": ("371bb26d6d2dce38784c889cb58b2aa5cecc654233c859557959ae79442bd70e"),
            "data_sha": MATCHED_CONTINUOUS_DATA_SHA,
            "config_sha": ("f38c6672de3cd1b4d9e0dfff72a98c93c9c20ed94c0a17cd772bfe6b0310e621"),
            "base_sha": ("24b2bcb85e7067c3759cba3f007782c46b174695fd77399da15bc468cfb3a35e"),
            "history_key": "video_pair_objective_invocations",
            "name": "s4-video-pair-objective",
        }
    if rank64:
        return {
            "prep_sha": MATCHED_CONTINUOUS_PREP_SHA,
            "data_sha": MATCHED_CONTINUOUS_DATA_SHA,
            "config_sha": ("e01916029545aea72be7e03dc2cbb9a5b35530c26c4a71711ec1bd4518c6f871"),
            "base_sha": ("eb7502b939da6339d4944421cfafae297e2251101f88c9327a4466f9226cc81f"),
            "history_key": "rank64_invocations",
            "name": "s4-rank64-matched-pilot",
        }
    if matched:
        pins = {
            "prep_sha": MATCHED_CONTINUOUS_PREP_SHA,
            "data_sha": MATCHED_CONTINUOUS_DATA_SHA,
            "config_sha": MATCHED_CONTINUOUS_CONFIG_SHA,
        }
        require(
            all(
                isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef")
                for h in pins.values()
            ),
            "continuous_matched_profile_not_frozen",
        )
        return {
            **pins,
            "base_sha": MATCHED_CONTINUOUS_BASE_SHA,
            "history_key": "continuous_matched_invocations",
            "name": "s4-continuous-matched-pool",
        }
    return {
        "prep_sha": CONTINUOUS_PREP_SHA,
        "data_sha": CONTINUOUS_DATA_SHA,
        "config_sha": CONTINUOUS_CONFIG_SHA,
        "base_sha": CONTINUOUS_BASE_SHA,
        "history_key": "continuous_invocations",
        "name": "s4-continuous-main-pool",
    }


RANK64_POLICY_SHA = "0cc0d96a2f77edc857dc2904cff78af95d3c1fd5daa56110637bb0d695292ff1"
RANK64_WEIGHTS_SHA = "62920299c7a1ac33b2a874b586a9d7f3ea3eaa4e38198bc20388642482318fc6"
RANK64_CONFIG_SHA = "8d79bb86dc522bc6da2b3de170dc704e51b60e2d2518582b07e8ffbfcdd6bc0a"
RANK64_EVIDENCE_SHA = {
    "artifacts/s4-rank64-initializer-001/rank-expansion.json": (
        "05e97013ac5d474866fdf208611bfef618bdfb65f5770c18f2924d21d37b2be7"
    ),
    "artifacts/s4-rank64-initializer-gpu-parity-001/receipt.json": (
        "e248d3a907ff62cad6e2c23ead90ed78d741a55aaa5205c5443279928bda3011"
    ),
    "artifacts/s4-rank-runtime-control-001/receipt.json": (
        "5c7aecf6541122d240668a8bc4ebc0160fe4ceee50e61d66bbd59d4f24a18683"
    ),
}
RANK64_SOURCE_ARCHIVE_SHA = "636aca42044ab7d3e87b4928a99a3deb0fe10abd9a8fa60ac91d8061f9e64427"


def rank64_derivative_bindings(root: Path, policy_path: Path) -> tuple[dict, dict]:
    """Identity evidence only: failed parity remains failed; baseline/launch is a later gate."""
    require(root.is_absolute() and policy_path.is_absolute(), "rank64_absolute_paths")
    bindings = {}
    receipts = []
    for relative, expected in RANK64_EVIDENCE_SHA.items():
        path = root / relative
        payload, binding = read_json(path)
        require(binding["sha256"] == expected, "rank64_evidence_sha")
        bindings[str(path)] = expected
        receipts.append(payload)
    expansion, parity, control = receipts
    require(sha(policy_path) == RANK64_POLICY_SHA, "rank64_independent_policy_sha")
    bindings[str(policy_path)] = RANK64_POLICY_SHA
    require(
        expansion["status"] == "converted_research_initializer"
        and expansion["exit_code"] == 0
        and expansion["new_rank"] == 64
        and expansion["new_alpha"] == 128
        and expansion["old_rank"] == 16
        and expansion["old_alpha"] == 32
        and expansion["source_weights_sha256"] == CONTINUOUS_PARENT_SHA
        and expansion["training_authorized"] is False
        and expansion["production_replacement"] is False,
        "rank64_derivative_lineage",
    )
    require(
        parity["status"] == "failed"
        and parity["exit_code"] == 1
        and parity["parity_validated"] is False
        and parity["max_joint_probability_difference"] > parity["parity_tolerance"]
        and control["status"] == "completed"
        and control["exit_code"] == 0
        and control["parity_validated"] is False
        and control["same_process"] is True
        and control["source_repeat_bit_identical_logits"] is True,
        "rank64_failed_parity_preserved",
    )
    for receipt in (parity, control):
        require(
            receipt["expanded_weights_sha256"] == RANK64_WEIGHTS_SHA
            and receipt["expanded_config_sha256"] == RANK64_CONFIG_SHA
            and receipt["source_weights_sha256"] == CONTINUOUS_PARENT_SHA,
            "rank64_probe_derivative_identity",
        )
    adapter = root / "artifacts/s4-rank64-initializer-001"
    for name, expected in (
        ("adapter_model.safetensors", RANK64_WEIGHTS_SHA),
        ("adapter_config.json", RANK64_CONFIG_SHA),
    ):
        require(
            expansion["output_sha256"][name] == expected and sha(adapter / name) == expected,
            "rank64_actual_initializer_sha",
        )
        bindings[str(adapter / name)] = expected
    config = json.loads((adapter / "adapter_config.json").read_bytes())
    require(config["r"] == 64 and config["lora_alpha"] == 128, "rank64_actual_capacity_pair")
    source = Path(expansion["source_path"])
    require(source.is_absolute(), "rank64_source_absolute")
    for name, expected in (
        ("adapter_model.safetensors", CONTINUOUS_PARENT_SHA),
        ("adapter_config.json", expansion["source_config_sha256"]),
    ):
        require(sha(source / name) == expected, "rank64_original_source_sha")
        bindings[str(source / name)] = expected
    archive = root.parent / "omnijev-rank64-code-b97edc0.tar"
    require(sha(archive) == RANK64_SOURCE_ARCHIVE_SHA, "rank64_execution_source_archive")
    bindings[str(archive)] = RANK64_SOURCE_ARCHIVE_SHA
    return bindings, {
        "path": str(adapter),
        "weights_sha256": RANK64_WEIGHTS_SHA,
        "config_sha256": RANK64_CONFIG_SHA,
        "rank": 64,
        "alpha": 128,
        "mathematical_source_weights_sha256": CONTINUOUS_PARENT_SHA,
        "numerical_parity_validated": False,
        "parent_quality_inheritance": False,
        "candidate_role": "independent_capacity_candidate",
        "training_authorized": False,
        "own_baseline_required": True,
        "same_train_data_sha256": MATCHED_CONTINUOUS_DATA_SHA,
    }


def rank64_training_bindings(root, launch, config):
    """Root-reviewed independent first10; never reuse the parent's measured quality."""
    policy = root / "reports/explore/rank64-independent-candidate-policy-001.md"
    bindings, identity = rank64_derivative_bindings(root, policy)
    pins = {
        "reports/s4-rank64-matched-pilot-cpu-001.json": (
            "b67d684c05bb038b0fb0232a3bcb4360ca6b0f7e3a8731862de3fd4e30112311"
        ),
        "reports/s4-rank64-baseline-review-001.json": (
            "8468c583b4bcb52f02a7bcce4e3f86782a456e6ac1ca735e559af6a72873a421"
        ),
        "reports/s4-rank64-initializer-dev50-001.json": (
            "f05ae672360c869998120ac1452bff8ad1096e39e98d7a1acf7f533590f8aa51"
        ),
    }
    receipts = []
    for relative, expected in pins.items():
        path = root / relative
        payload, b = read_json(path)
        require(b["sha256"] == expected, "rank64_root_baseline_binding")
        bindings[str(path)] = expected
        receipts.append(payload)
    cpu, review, baseline = receipts
    require(
        cpu["status"] == "cpu_prepared_only"
        and cpu["exit_code"] == 0
        and cpu["derived_config_sha256"] == launch["derived_config_sha256"]
        and cpu["data_sha256"] == MATCHED_CONTINUOUS_DATA_SHA,
        "rank64_cpu_config_receipt",
    )
    require(
        review["decision"] == "approve_first10"
        and review["max_steps"] == 10
        and review["max_seconds"] == 360
        and review["policy_sha256"] == RANK64_POLICY_SHA
        and review["initializer_sha256"] == RANK64_WEIGHTS_SHA
        and review["baseline_report_sha256"]
        == launch["baseline_report_sha256"]
        == pins["reports/s4-rank64-initializer-dev50-001.json"]
        and launch["baseline_review_sha256"] == pins["reports/s4-rank64-baseline-review-001.json"],
        "rank64_root_first10_review",
    )
    require(
        baseline["status"] == "completed"
        and baseline["exit_code"] == 0
        and baseline["gpu_requested"] is True
        and baseline["paired_input_hashes_verified"] == 50
        and baseline["initializer_sha256"] == RANK64_WEIGHTS_SHA
        and baseline["initializer_config_sha256"] == RANK64_CONFIG_SHA
        and baseline["candidate"]["count"] == 50
        and 0.64 <= baseline["candidate"]["accuracy"] <= 1
        and 0 <= baseline["candidate"]["brier"] <= 0.46748601999097937,
        "rank64_actual_own_baseline",
    )
    for receipt in (launch, review, baseline):
        require(
            receipt["parity_validated"] is False and receipt["parent_quality_inherited"] is False,
            "rank64_independent_quality_boundary",
        )
    require(
        launch["source_commit"] == "b97edc0"
        and launch["source_archive_sha256"] == RANK64_SOURCE_ARCHIVE_SHA,
        "rank64_runtime_source",
    )
    original = root / "artifacts/s4-continuous-matched-pool-001/config-ce.json"
    require(sha(original) == MATCHED_CONTINUOUS_CONFIG_SHA, "rank64_original_recipe_config")
    old = json.loads(original.read_bytes())
    require(config == {**old, "rank": 64, "alpha": 128}, "rank64_capacity_only_config_delta")
    bindings[str(original)] = MATCHED_CONTINUOUS_CONFIG_SHA
    launcher = root.parent / "launch-rank64-matched-pilot-001.py"
    require(
        sha(launcher) == "9ac9cea6e9a52429cd41041784a84faaf2cf37d8144b16550304c2a9d765560a",
        "rank64_actual_launcher_source",
    )
    bindings[str(launcher)] = sha(launcher)
    baseline_artifact = root / "artifacts/s4-rank64-initializer-dev50-001/uniform"
    for name, field in (
        ("receipt.json", "candidate_receipt_sha256"),
        ("records.jsonl", "candidate_records_sha256"),
    ):
        path = baseline_artifact / name
        require(sha(path) == baseline[field], "rank64_baseline_artifact_sha")
        bindings[str(path)] = baseline[field]
    return bindings, {
        **identity,
        "own_baseline": bind(root / "reports/s4-rank64-initializer-dev50-001.json"),
        "root_first10_review": bind(root / "reports/s4-rank64-baseline-review-001.json"),
        "actual_authorization_scope": "root reviewed first10 only; no automatic resume",
    }


def video_objective_bindings(root, launch, config, prep, runtime_data):
    """Separately authorized ranking first10; the failed CE control remains immutable."""
    bindings, identity = rank64_derivative_bindings(
        root, root / "reports/explore/rank64-independent-candidate-policy-001.md"
    )
    pins = {
        "reports/s4-video-pair-objective-decision-001.json": (
            "6b7ecc322ccc001db63490811f04beea6d2b1bed81b96100a5215b1b34b8e51f"
        ),
        "reports/s4-rank64-initializer-dev50-001.json": (
            "f05ae672360c869998120ac1452bff8ad1096e39e98d7a1acf7f533590f8aa51"
        ),
        "reports/s4-rank64-matched-pilot-training-001.json": (
            "11e14016bfe5e546fe67bb0b7b1a21dbc5b89a082253ffbb743ee3bbd4e90a05"
        ),
        "reports/s4-rank64-matched-pilot-dev50-001.json": (
            "5a43069e962446387a5e7d171c3f17197b727b111644eaa482ec8ea968bc22df"
        ),
    }
    for relative, expected in pins.items():
        path = root / relative
        require(sha(path) == expected, "video_objective_root_evidence_sha")
        bindings[str(path)] = expected
    decision_path = root / "reports/s4-video-pair-objective-decision-001.json"
    decision = json.loads(decision_path.read_bytes())
    profile = continuous_profile(video_objective=True)
    expected = {
        "decision": "approve_first10",
        "max_steps": 10,
        "max_seconds": 240,
        "initializer_sha256": RANK64_WEIGHTS_SHA,
        "prepared_report_sha256": profile["prep_sha"],
        "config_sha256": profile["config_sha"],
        "baseline_report_sha256": pins["reports/s4-rank64-initializer-dev50-001.json"],
        "control_training_report_sha256": pins["reports/s4-rank64-matched-pilot-training-001.json"],
    }
    require(
        all(decision.get(k) == v for k, v in expected.items()), "video_objective_first10_decision"
    )
    require(
        launch.get("no_resume") is True
        and launch.get("invocation_steps") == 10
        and launch.get("training_hard_cap_seconds") == 240
        and launch.get("policy_sha256") == sha(decision_path),
        "video_objective_launch_authority",
    )
    for receipt in (decision, launch):
        require(
            receipt.get("parity_validated") is False
            and receipt.get("parent_quality_inherited") is False,
            "video_objective_independent_quality",
        )
    require(
        prep.get("status") == "cpu_prepared_only"
        and prep.get("exit_code") == 0
        and prep.get("training_authorized") is False
        and prep.get("audited_pairs") == 300,
        "video_objective_actual_preparation",
    )
    for b in prep["bindings"].values():
        path = Path(b["path"])
        require(path.is_absolute() and sha(path) == b["sha256"], "video_objective_prepared_binding")
        bindings[str(path)] = b["sha256"]
    require(
        launch.get("source_commit") == prep.get("source_commit") == "7b75f9f"
        and launch.get("source_archive_sha256")
        == "49b11d244dc483a59895695468697e05ee4d45ff8b491cad377724137b15d32d",
        "video_objective_runtime_source",
    )
    archive = root.parent / "omnijev-video-pair-code-7b75f9f.tar"
    require(sha(archive) == launch["source_archive_sha256"], "video_objective_archive_sha")
    bindings[str(archive)] = sha(archive)
    launcher = root.parent / "launch-video-pair-objective-pilot-001.py"
    require(
        sha(launcher) == "0ab1a071ed3e84e4347bf178212d7bcf1171b184249e98cf1f43c17dc9a7fa0d",
        "video_objective_launcher_sha",
    )
    bindings[str(launcher)] = sha(launcher)
    source = root.parent / "s4-video-pair-code-7b75f9f"
    for name, expected_sha in prep["source_sha256"].items():
        require(sha(source / name) == expected_sha, "video_objective_preparation_source_sha")
        bindings[str(source / name)] = expected_sha
    control_path = root / "artifacts/s4-rank64-matched-pilot-001/config-ce.json"
    require(
        sha(control_path) == continuous_profile(rank64=True)["config_sha"],
        "video_objective_control_config",
    )
    control = json.loads(control_path.read_bytes())
    sidecar = root / "artifacts/s4-video-pair-objective-pilot-001/video-pairs.json"
    ledger = root / "artifacts/s4-continuous-matched-pool-001/allocation-ledger.jsonl"
    delta = {
        "video_pair_loss_weight": 0.1,
        "video_pair_sidecar": str(sidecar),
        "video_pair_sidecar_sha256": (
            "c1ba3471df27ce6b7df7df954f7b3b5377727cf9e1902f1f811386453d18f3c8"
        ),
        "video_pair_ledger": str(ledger),
        "video_pair_ledger_sha256": (
            "1d18f75d395a6935b1fdececd7530d543144b4fa0e2a0974c31541b3d9c5fbb3"
        ),
    }
    require(config == {**control, **delta}, "video_objective_only_config_delta")
    for path, expected_sha in (
        (sidecar, delta["video_pair_sidecar_sha256"]),
        (ledger, delta["video_pair_ledger_sha256"]),
        (control_path, continuous_profile(rank64=True)["config_sha"]),
    ):
        require(sha(path) == expected_sha, "video_objective_sidecar_ledger_sha")
        bindings[str(path)] = expected_sha
    from train import video_pairs

    require(
        Path(video_pairs.__file__).resolve() == (source / "train/video_pairs.py").resolve(),
        "video_objective_imported_source_identity",
    )
    samples = [
        Sample.model_validate_json(line)
        for line in runtime_data.read_bytes().splitlines()
        if line.strip()
    ]
    pairs = video_pairs.load_video_pairs(LoopConfig.model_validate(config), samples, runtime_data)
    require(
        len(pairs) == 300
        and set(pairs) == {step * 12 + offset for step in range(100) for offset in (0, 2, 4)},
        "video_objective_actual_pair_offsets",
    )
    return bindings, {
        **identity,
        "candidate_role": "independent_conditional_ranking_candidate",
        "root_first10_decision": bind(decision_path),
        "sidecar": bind(sidecar),
        "own_baseline": bind(root / "reports/s4-rank64-initializer-dev50-001.json"),
        "ce_control": bind(root / "reports/s4-rank64-matched-pilot-training-001.json"),
        "audited_pairs": len(pairs),
        "loss_weight": 0.1,
        "actual_authorization_scope": "root reviewed first10 only; no resume",
    }


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


def continuous_fixed_identity(
    prep: dict, config: dict, final: dict, *, rank64=False, video_objective=False
) -> None:
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
        == (100, 1200, 12, 42, 64 if rank64 else 16, 128 if rank64 else 32, 16, 1120),
        "continuous_frozen_config",
    )
    require(
        not any(
            config.get(k, 0) for k in ("pair_loss_weight", "reference_kl_weight", "brier_weight")
        ),
        "continuous_ce_only",
    )
    require(
        config.get("video_pair_loss_weight", 0) == (0.1 if video_objective else 0),
        "continuous_video_objective_scope",
    )
    parent = final.get("init_adapter", {})
    require(
        parent.get("weights_sha256") == (RANK64_WEIGHTS_SHA if rank64 else CONTINUOUS_PARENT_SHA),
        "continuous_original_f2db_parent",
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
    root: Path,
    legacy_root: Path,
    output: Path,
    historical_index: Path,
    launch_path: Path,
    *,
    matched_pool: bool = False,
    rank64_candidate: bool = False,
    video_objective: bool = False,
    program_stratified: bool = False,
) -> dict:
    """One settled invocation over one immutable pool, based on the preceding inventory."""
    require(
        sum((matched_pool, rank64_candidate, video_objective, program_stratified)) <= 1,
        "continuous_profile_exclusive",
    )
    profile = continuous_profile(
        matched_pool, rank64_candidate, video_objective, program_stratified
    )
    derived_candidate = rank64_candidate or video_objective
    matched_pool = matched_pool or derived_candidate
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
        history = base.get(profile["history_key"], [])
        if history:
            previous = history[-1]
            require(
                previous["end_step"] == start and previous["status"] == "paused",
                "continuous_prior_interval",
            )
        else:
            require(
                start == 0 and base_binding["sha256"] == profile["base_sha"],
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
        require(sha(prep_path) == profile["prep_sha"], "continuous_preparation_sha")
        prep = json.loads(prep_path.read_bytes())
        for name, expected in prep["output_sha256"].items():
            require(Path(name).name == name, "continuous_safe_preparation_name")
            path = prep_path.with_name(name)
            require(sha(path) == expected, "continuous_preparation_output_sha")
            bindings[str(path)] = expected
        config_path = bound("config_file")
        require(sha(config_path) == profile["config_sha"], "continuous_global_config_sha")
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
        observed_runtimes = {str(raw_path): mutable_observation["sha256"]}
        for key in (
            "continuous_invocations",
            "continuous_matched_invocations",
            "rank64_invocations",
            "video_pair_objective_invocations",
            "program_stratified_invocations",
        ):
            for old_invocation in base.get(key, []):
                old = old_invocation["runtime_report_observation"]
                if old["path"] != str(raw_path):
                    observed_runtimes[old["path"]] = old["sha256"]
        require(
            all(sha(Path(p)) == h for p, h in observed_runtimes.items()),
            "continuous_other_profile_runtime_changed",
        )
        interval = continuous_interval(final, start, end)
        pool_path = prep_path
        pool_prep = prep
        if video_objective:
            pool_path = root / "artifacts/s4-continuous-matched-pool-001/report.json"
            require(
                sha(pool_path) == MATCHED_CONTINUOUS_PREP_SHA, "objective_original_pool_preparation"
            )
            bindings[str(pool_path)] = MATCHED_CONTINUOUS_PREP_SHA
            pool_prep = json.loads(pool_path.read_bytes())
        continuous_fixed_identity(
            pool_prep, config, final, rank64=derived_candidate, video_objective=video_objective
        )
        runtime_data = bound("runtime_data")
        require(
            sha(runtime_data) == final["data_sha256"] == profile["data_sha"],
            "continuous_runtime_data_sha",
        )
        comparison = equivalent(runtime_data, pool_path.with_name("train.jsonl"))
        require(
            comparison["rows"] == 1200 and all(v in mappings for v in comparison["data_versions"]),
            "continuous_pool_mapping",
        )
        if program_stratified:
            bindings.update(
                continuous_matched_bindings(
                    root, pool_path.parent, pool_prep, program_stratified=True
                )
            )
            bindings.update(
                program_training_bindings(root, launch, config, base, history, start, end)
            )
        elif matched_pool:
            bindings.update(continuous_matched_bindings(root, pool_path.parent, pool_prep))
        if rank64_candidate:
            require(start == 0 and end <= 10 and not history, "rank64_first_invocation_only")
            derivative_bindings, derivative_identity = rank64_training_bindings(
                root, launch, config
            )
            bindings.update(derivative_bindings)
        if video_objective:
            require(
                start == 0 and end <= 10 and not history, "video_objective_first_invocation_only"
            )
            objective_bindings, objective_identity = video_objective_bindings(
                root, launch, config, prep, runtime_data
            )
            bindings.update(objective_bindings)
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
            and (
                video_objective
                or program_stratified
                or launch.get("parent_selection")
                == "root_selected_f2db_balanced_retention_and_full_video_evidence"
            ),
            "continuous_root_selected_parent",
        )
        require(
            runtime_data == raw_path.parent / "data.jsonl", "continuous_actual_runtime_data_path"
        )
        parent_dir = Path(parent["path"])
        require(
            parent_dir.is_absolute()
            and sha(parent_dir / "adapter_model.safetensors")
            == (RANK64_WEIGHTS_SHA if derived_candidate else CONTINUOUS_PARENT_SHA),
            "continuous_parent_weights_sha",
        )
        bindings[str(parent_dir / "adapter_model.safetensors")] = (
            RANK64_WEIGHTS_SHA if derived_candidate else CONTINUOUS_PARENT_SHA
        )
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
        if program_stratified and start:
            previous_raw = json.loads(
                Path(history[-1]["runtime_report_snapshot"]["path"]).read_bytes()
            )
            require(
                final["losses"][:start] == previous_raw["losses"]
                and len(previous_raw["losses"]) == start,
                "program_resume_loss_prefix",
            )
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
            existing = [a for a in combined["samples"] if a["sha256"] == profile["data_sha"]]
            if existing:
                require(
                    len(existing) == 1
                    and existing[0]["expected_pool"] == "train"
                    and existing[0]["exposure"] == "current",
                    "continuous_single_current_pool",
                )
                if derived_candidate:
                    existing[0]["path"] = str(runtime_data)
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
                component_path = pool_path.with_name("components.jsonl")
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
            if str(path) not in observed_runtimes:
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
            and all(sha(Path(p)) == h for p, h in observed_runtimes.items()),
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
            profile["history_key"]: history,
            "continuous_profile": profile["name"],
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
        if matched_pool:
            for key in (
                "continuous_invocations",
                "continuous_actual_global_draws_seen",
                "native_actual_draws_seen",
                "specialist_actual_draws_seen",
                "specialist_storage_samples",
                "specialist_unique_families",
                "specialist_new_source_families",
                "text_matched_actual_draws_seen",
                "text_matched_pairs",
            ):
                if key in base:
                    report[key] = base[key]
            report["continuous_matched_actual_global_draws_seen"] = end * 12
            report["actual_global_draws_seen"] = base["actual_global_draws_seen"]
            report["continuous_matched_pool_included"] = True
        if rank64_candidate:
            report["continuous_matched_invocations"] = base["continuous_matched_invocations"]
            report["continuous_matched_actual_global_draws_seen"] = base[
                "continuous_matched_actual_global_draws_seen"
            ]
            report["rank64_actual_global_draws_seen"] = end * 12
            report["rank64_derivative_identity"] = derivative_identity
            report["rank64_independent_candidate_included"] = True
            report["parity_validated"] = False
            report["parent_quality_inherited"] = False
        if video_objective:
            for key in (
                "continuous_matched_invocations",
                "continuous_matched_actual_global_draws_seen",
                "rank64_invocations",
                "rank64_actual_global_draws_seen",
                "rank64_derivative_identity",
                "rank64_independent_candidate_included",
            ):
                report[key] = base[key]
            report["video_pair_objective_actual_global_draws_seen"] = end * 12
            report["video_pair_objective_identity"] = objective_identity
            report["video_pair_objective_included"] = True
            report["parity_validated"] = False
            report["parent_quality_inherited"] = False
        if program_stratified:
            for key in (
                "continuous_invocations",
                "continuous_actual_global_draws_seen",
                "continuous_matched_invocations",
                "continuous_matched_actual_global_draws_seen",
                "rank64_invocations",
                "rank64_actual_global_draws_seen",
                "rank64_derivative_identity",
                "rank64_independent_candidate_included",
                "video_pair_objective_invocations",
                "video_pair_objective_actual_global_draws_seen",
                "video_pair_objective_identity",
                "video_pair_objective_included",
                "native_actual_draws_seen",
                "specialist_actual_draws_seen",
                "specialist_storage_samples",
                "specialist_unique_families",
                "specialist_new_source_families",
                "text_matched_actual_draws_seen",
                "text_matched_pairs",
            ):
                if key in base:
                    report[key] = base[key]
            report["actual_global_draws_seen"] = base["actual_global_draws_seen"]
            report["program_stratified_actual_global_draws_seen"] = end * 12
            report["program_stratified_pool_included"] = True
            report["program_exclusion_scope"] = prep["exclusions"]["scope"]
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
    parser.add_argument("--video-text-matched", action="store_true")
    parser.add_argument("--video-specialist", action="store_true")
    parser.add_argument("--native-event-condition", action="store_true")
    parser.add_argument("--program-stratified-pool", action="store_true")
    parser.add_argument("--video-pair-objective", action="store_true")
    parser.add_argument("--rank64-independent-candidate", action="store_true")
    parser.add_argument("--continuous-matched-pool", action="store_true")
    parser.add_argument("--continuous-main-pool", action="store_true")
    parser.add_argument("--continuous-launch-report", type=Path)
    parser.add_argument("--continuous-base-exposure", type=Path)
    args = parser.parse_args()
    try:
        if args.continuous_base_exposure is not None:
            require(
                args.continuous_main_pool
                or args.continuous_matched_pool
                or args.rank64_independent_candidate
                or args.video_pair_objective
                or args.program_stratified_pool,
                "continuous_base_requires_branch",
            )
            launch_path = args.continuous_launch_report or args.root / "reports" / (
                (
                    "s4-program-stratified"
                    if args.program_stratified_pool
                    else "s4-video-pair-objective"
                    if args.video_pair_objective
                    else "s4-rank64-matched-pilot"
                    if args.rank64_independent_candidate
                    else "s4-continuous-matched-pool"
                    if args.continuous_matched_pool
                    else "s4-continuous-main-pool"
                )
                + ("-launch-step0010.json" if args.program_stratified_pool else "-launch-001.json")
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
                or args.root
                / "reports"
                / (
                    (
                        "s4-program-stratified"
                        if args.program_stratified_pool
                        else "s4-video-pair-objective"
                        if args.video_pair_objective
                        else "s4-rank64-matched-pilot"
                        if args.rank64_independent_candidate
                        else "s4-continuous-matched-pool"
                        if args.continuous_matched_pool
                        else "s4-continuous-main-pool"
                    )
                    + (
                        "-launch-step0010.json"
                        if args.program_stratified_pool
                        else "-launch-001.json"
                    )
                ),
                matched_pool=args.continuous_matched_pool,
                rank64_candidate=args.rank64_independent_candidate,
                video_objective=args.video_pair_objective,
                program_stratified=args.program_stratified_pool,
            )
            if args.continuous_main_pool
            or args.continuous_matched_pool
            or args.rank64_independent_candidate
            or args.video_pair_objective
            or args.program_stratified_pool
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
                native_event_condition=args.native_event_condition,
                video_specialist=args.video_specialist,
                video_text_matched=args.video_text_matched,
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
