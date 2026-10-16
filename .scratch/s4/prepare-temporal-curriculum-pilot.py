"""CPU-only fixed TRAIN temporal curriculum, preserving source gold and media."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import time
from collections import Counter
from pathlib import Path

from data.schema import HardTarget, Sample
from scripts.gold_inventory import Manifest, audit_manifest, bound_lines, digest
from train.trainer import LoopConfig, data_bytes, smoke_samples, verified_init_adapter

INIT_SHA = "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"
VIDEO_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def kind(sample: Sample) -> str:
    words = set(sample.questions[0].instruction.lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else "motion"
        if words & {"moving", "stationary"}
        else "existence"
    )


def select_video(
    samples: list[Sample],
    *,
    hash_namespace: str = "s4-temporal-curriculum:42:",
    excluded_families: frozenset[str] = frozenset(),
) -> dict[tuple[str, str], list[Sample]]:
    require(
        all(
            s.meta.pool == "train"
            and s.meta.modality == ["video"]
            and isinstance(s.questions[0].target, HardTarget)
            and s.questions[0].target.value in ["yes", "no"]
            for s in samples
        ),
        "video_train_gold",
    )
    selected: dict[tuple[str, str], list[Sample]] = {}
    families = set()
    for task, quotas in [
        ("temporal", {"yes": 25, "no": 25}),
        ("motion", {"yes": 2, "no": 3}),
        ("existence", {"yes": 3, "no": 2}),
    ]:
        for target, count in quotas.items():
            candidates = sorted(
                [s for s in samples if kind(s) == task and s.questions[0].target.value == target],
                key=lambda s: digest((hash_namespace + s.id).encode()),
            )
            bucket = []
            for s in candidates:
                if s.meta.family_id in families or s.meta.family_id in excluded_families:
                    continue
                families.add(s.meta.family_id)
                bucket.append(s)
                if len(bucket) == count:
                    break
            require(len(bucket) == count, "video_bucket_capacity")
            selected[(task, target)] = bucket
    require(len(families) == 60, "video_unique_family")
    return selected


def desired_draws(video: dict, pairs: list[tuple[Sample, Sample]], old: dict) -> list[Sample]:
    require(
        len(pairs) == 10
        and all(len(old[c]) == 10 for c in ["audio", "image", "screenshot", "text"]),
        "layout_capacity",
    )
    result = []
    for step in range(10):
        # Five temporal examples: even steps 3yes/2no, odd steps 2yes/3no.
        even_before = (step + 1) // 2
        odd_before = step // 2
        yes_start = 3 * even_before + 2 * odd_before
        no_start = 2 * even_before + 3 * odd_before
        yes_count = 3 if step % 2 == 0 else 2
        no_count = 5 - yes_count
        result.extend(video[("temporal", "yes")][yes_start : yes_start + yes_count])
        result.extend(video[("temporal", "no")][no_start : no_start + no_count])
        minor_index = step // 2
        if step % 2 == 0:
            target = "yes" if minor_index < 2 else "no"
            index = minor_index if target == "yes" else minor_index - 2
            result.append(video[("motion", target)][index])
        else:
            target = "yes" if minor_index < 3 else "no"
            index = minor_index if target == "yes" else minor_index - 3
            result.append(video[("existence", target)][index])
        result.extend(pairs[step])
        result.extend(old[c][step] for c in ["audio", "image", "screenshot", "text"])
    require(len(result) == 120 and len({s.id for s in result}) == 120, "unique_draw_layout")
    return result


def helper_module():
    helper = globals().get("PREP_HELPER")
    if helper is not None:
        return helper
    path = Path(__file__).with_name("prepare-claim-pair-pilot.py")
    spec = importlib.util.spec_from_file_location("claim_prep_helper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(
    root: Path,
    out: Path,
    exposure_sha: str,
    claim_sha: str,
    video_report_sha: str,
    *,
    video_hash_namespace: str = "s4-temporal-curriculum:42:",
    excluded_video_families: frozenset[str] = frozenset(),
    selection_provenance: dict | None = None,
    data_ready_only: bool = False,
) -> dict:
    started = time.monotonic()
    h = helper_module()
    for path in [root, out]:
        require(path.is_absolute(), "absolute_paths")
    require(not out.exists(), "output_exists")
    exposure = root / "artifacts/s4-pilot-exposure-004"
    claim = root / "artifacts/s4-claim-pair-pilot-001"
    source = root / "artifacts/s3-pools-video-001/run-a"
    for p, s in [
        (exposure / "report.json", exposure_sha),
        (claim / "report.json", claim_sha),
        (source / "report.json", video_report_sha),
    ]:
        require(h.sha(p) == s, "receipt_sha")
    er, cr, vr = (h.read(p / "report.json") for p in [exposure, claim, source])
    require(
        er["status"] == "known_pilot_exposures_reconciled"
        and er["exit_code"] == 0
        and er["unknown_update_bearing_inputs"] == 0,
        "exposure_status",
    )
    h.bound_outputs(exposure, er, ["inventory-manifest.json"])
    require(
        cr["status"] == "prepared_pending_pair_hashes"
        and cr["pilot_isolation"] == "isolated"
        and cr["exit_code"] == 0,
        "claim_preparation",
    )
    h.bound_outputs(
        claim, cr, ["train.jsonl", "allocation-ledger.jsonl", "config-ce0.json", "draw-recipe.json"]
    )
    require(vr["status"] == "passed" and vr["exit_code"] == 0, "video_receipt")
    require(
        h.sha(source / "train.jsonl") == vr["outputs"]["train_jsonl_sha256"] == VIDEO_SHA,
        "video_source_sha",
    )
    require(
        h.sha(source / "allocation-ledger.jsonl") == vr["outputs"]["allocation_ledger_sha256"],
        "video_ledger_sha",
    )
    inventory = Manifest.model_validate_json((exposure / "inventory-manifest.json").read_bytes())
    jointpath = root / "reports/s3-joint-preparation-001.json"
    joint = h.read(jointpath)
    require(
        joint["status"] == "passed"
        and joint["exit_code"] == 0
        and joint["total_rows"] == 21184
        and joint["ledger_check"]["verified_train"] == 21184,
        "joint_full_scope",
    )
    v = joint["inputs"]["video"]
    require(
        v["count"] == 8299
        and v["sha256"] == VIDEO_SHA
        and v["ledger_sha256"] == h.sha(source / "allocation-ledger.jsonl")
        and v["report_sha256"] == video_report_sha,
        "joint_video_binding",
    )
    require(
        h.sha(root / "artifacts/s3-joint-pool-001/joint-train.jsonl") == joint["joint_sha256"],
        "joint_full_data_sha",
    )
    evidence = globals().get("SOURCE_EVIDENCE")
    require(
        isinstance(evidence, dict) and bool(evidence.get("files")), "admission_evidence_required"
    )
    require(evidence["joint_preparation_sha256"] == h.sha(jointpath), "joint_evidence_sha")
    official = Path("/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files")
    require(
        h.sha(official / "README.txt")
        == "a4838cc68b79ba049b54f95def1714cdbbd1380193d7d2a5d1a85440ac862e7b",
        "official_license_sha",
    )
    require(
        h.sha(official / "questions/train.json")
        == "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4",
        "official_question_sha",
    )
    admission_trace = {
        **evidence,
        "joint_actual_sha256": h.sha(jointpath),
        "original_training_admitted": vr.get("training_admitted"),
        "source_scope": "full8299 TRAIN, not only previously exposed150",
    }
    video_source = [Sample.model_validate(r) for r in h.rows(source / "train.jsonl")]
    require(len(video_source) == 8299, "video_source_count")
    video = select_video(
        video_source, hash_namespace=video_hash_namespace, excluded_families=excluded_video_families
    )
    from scripts.gold_media_bundle import media_path

    for bucket in video.values():
        for sample in bucket:
            media = sample.prefix.media[0]
            actual = media_path([Path(vr["source"]["media_root"])], media.uri)
            require(h.sha(actual) == media.sha256, "selected_media_sha")
    admission_trace["selected_video_media_verified"] = 60
    claim_samples = [Sample.model_validate(r) for r in h.rows(claim / "train.jsonl")]
    require(
        all(s.meta.pool == "train" for s in claim_samples) and len(claim_samples) == 120,
        "claim_train",
    )
    mapping = h.cohorts(claim_samples, inventory)
    by_id = {s.id: s for s in claim_samples}
    recipe = h.read(claim / "draw-recipe.json")
    cross = {
        rule: sorted(
            [p for p in recipe["pairs"] if p["negative_rule"] == rule],
            key=lambda p: digest(("s4-temporal-cross:42:" + p["group_id"]).encode()),
        )[:5]
        for rule in ["image_false_audio_true", "image_true_audio_false"]
    }
    require(all(len(v) == 5 for v in cross.values()), "cross_rule_capacity")
    group_rows = [
        cross[rule][index]
        for index in range(5)
        for rule in ["image_false_audio_true", "image_true_audio_false"]
    ]
    require(len({p["group_id"] for p in group_rows}) == 10, "unique_cross_groups")
    pairs = []
    for p in group_rows:
        positive, negative = by_id[p["positive_id"]], by_id[p["negative_id"]]
        require(
            positive.meta.family_id == negative.meta.family_id
            and positive.questions[0].target.value == "yes"
            and negative.questions[0].target.value == "no",
            "cross_gold_family",
        )
        pairs.append((positive, negative))
    old = {
        c: sorted(
            [s for s in claim_samples if mapping[s.meta.data_version] == c],
            key=lambda s: digest(("s4-temporal-replay:42:" + s.id).encode()),
        )
        for c in ["audio", "image", "screenshot", "text"]
    }
    draws = desired_draws(video, pairs, old)
    versions = h.cohorts(draws, inventory)
    arranged = h.inverse_order(draws)
    # Only candidate-matching TRAIN allocation metadata is used; no reserve Sample loaded.
    ledger = h.exact_ledger(
        arranged,
        [r for r in h.rows(source / "allocation-ledger.jsonl") if r["pool"] == "train"]
        + h.rows(claim / "allocation-ledger.jsonl"),
    )
    out.mkdir(mode=0o700)
    h.write(out / "train.jsonl", data_bytes(arranged).decode())
    h.write(
        out / "allocation-ledger.jsonl",
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in ledger),
    )
    combined = inventory.model_dump(mode="json", exclude_none=True)
    wanted = {s.id for s in arranged}
    for index, artifact in enumerate(inventory.samples):
        if (
            artifact.expected_pool == "train"
            and artifact.exposure == "current"
            and any(json.loads(line)["id"] in wanted for line in bound_lines(artifact))
        ):
            combined["samples"][index]["exposure"] = "historical"
    combined["samples"].append(
        {
            "path": str(out / "train.jsonl"),
            "sha256": h.sha(out / "train.jsonl"),
            "expected_pool": "train",
            "exposure": "current",
            "cohorts_by_data_version": versions,
        }
    )
    h.write(out / "inventory-manifest.json", combined)
    audit = audit_manifest(out / "inventory-manifest.json")
    h.write(out / "inventory-audit.json", audit.model_dump(mode="json"))
    require(audit.outcome == "isolated", "inventory_isolation")
    oldconfig = h.read(claim / "config-ce0.json")
    # Keep all shared media/optimizer settings. Pair/ref fields are omitted/off.
    require(
        not oldconfig.get("pair_loss_weight", 0)
        and not oldconfig.get("reference_kl_weight", 0)
        and not oldconfig.get("brier_weight", 0),
        "ce_only",
    )
    config = {
        **oldconfig,
        "seed": 42,
        "sample_count": 120,
        "steps": 10,
        "accumulation": 12,
        "midway_step": 5,
        "checkpoint_every": 5,
        "expected_data_versions": sorted(versions),
    }
    validated = LoopConfig.model_validate(config)
    smoke_samples(
        validated, out / "train.jsonl", expected_data_versions=config["expected_data_versions"]
    )
    h.write(out / "config-ce.json", config)
    order = list(range(120))
    random.Random(42).shuffle(order)
    h.write(
        out / "draw-recipe.json",
        {
            "seed": 42,
            "steps": 10,
            "accumulation": 12,
            "video_heuristic_not_official": True,
            "historical_unseen": "not_claimed",
            "cross_groups": [
                {
                    "group_id": p["group_id"],
                    "negative_rule": p["negative_rule"],
                    "positive_id": p["positive_id"],
                    "negative_id": p["negative_id"],
                }
                for p in group_rows
            ],
            "draws": [
                {
                    "draw": n,
                    "step": n // 12,
                    "offset": n % 12,
                    "data_index": order[n],
                    "item_id": s.id,
                    "data_version": s.meta.data_version,
                    "sample_sha256": h.canonical_sample_sha(s),
                    "cohort": versions[s.meta.data_version],
                    "video_heuristic": kind(s) if s.meta.modality == ["video"] else None,
                    "target": s.questions[0].target.value,
                }
                for n, s in enumerate(draws)
            ],
        },
    )
    identity = verified_init_adapter(
        root / "artifacts/s4-claim-pair-ce0-train-001/adapter", INIT_SHA
    )
    init_report_path = root / "reports/s4-claim-pair-ce0-training-001.json"
    require(
        h.sha(init_report_path)
        == "c30ba04f01aae54dd24e816f7713e4c3bcbcc8e0e78706b3b985a65a30aa7675",
        "init_training_receipt_sha",
    )
    init_report = h.read(init_report_path)
    require(
        init_report["status"] == "completed"
        and init_report["exit_code"] == 0
        and init_report["steps_completed"] == 10
        and init_report["adapter_sha256"] == INIT_SHA,
        "init_training_receipt",
    )
    adapter_config = h.read(Path(identity["path"]) / "adapter_config.json")
    require(
        adapter_config["r"] == init_report["config"]["rank"] == 16
        and adapter_config["lora_alpha"] == init_report["config"]["alpha"] == 32,
        "init_config_training_binding",
    )
    result = {
        "status": "ready_for_preflight",
        "data_ready_for_preflight": True,
        "exit_code": 0,
        "gpu_used": False,
        "training_started": False,
        "inventory_isolation": audit.outcome,
        "init_adapter": identity,
        "init_training_receipt": {"path": str(init_report_path), "sha256": h.sha(init_report_path)},
        "init_selection": "root_selected_ce0",
        "arm_wallclock_limit_seconds": 240,
        "total_draws": 120,
        "per_step": {
            "video": 6,
            "crossmodal": 2,
            "audio": 1,
            "image": 1,
            "screenshot": 1,
            "text": 1,
        },
        "video_selected": {
            task: {t: len(video[(task, t)]) for t in ["yes", "no"]}
            for task in ["temporal", "motion", "existence"]
        },
        "video_unique_families": 60,
        "cross_groups": 10,
        "cross_negative_rules": dict(Counter(p["negative_rule"] for p in group_rows)),
        "historical_unseen": "not_claimed; no full history exclusion",
        "source_original_training_admitted_field": vr.get("training_admitted"),
        "source_admission": "verified_existing_scope",
        "source_admission_boundary": (
            "Existing ADR0005/S3 frozen full-pool scope; original builder False retained."
        ),
        "admission_trace": admission_trace,
        "quota_decision": {
            "original_suggestion": {"temporal": 40, "motion": 10, "existence": 10},
            "actual": {"temporal": 50, "motion": 5, "existence": 5},
            "ratio_sweep": False,
        },
        "input_report_sha256": {
            "exposure004": exposure_sha,
            "claim_preparation": claim_sha,
            "video_source": video_report_sha,
        },
        "source_train_sha256": VIDEO_SHA,
        "output_sha256": {p.name: h.sha(p) for p in out.iterdir() if p.is_file()},
        "config_changes": {
            k: {"old": oldconfig.get(k), "new": v}
            for k, v in config.items()
            if oldconfig.get(k) != v
        },
        "duration_seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "cost": None,
        "script_sha256": globals().get("SOURCE_SHA") or h.sha(Path(__file__)),
        "helper_sha256": globals().get("HELPER_SHA")
        or h.sha(Path(__file__).with_name("prepare-claim-pair-pilot.py")),
    }
    if video_hash_namespace != "s4-temporal-curriculum:42:" or excluded_video_families:
        result["video_selection"] = {
            "hash_namespace": video_hash_namespace,
            "excluded_prior_family_count": len(excluded_video_families),
            "excluded_prior_families": sorted(excluded_video_families),
            "selected_families": sorted(
                s.meta.family_id for bucket in video.values() for s in bucket
            ),
            "historical_exclusion_scope": (
                "prior temporal curriculum60families only; not all historical exposures"
            ),
            "provenance": selection_provenance,
        }
    if data_ready_only:
        result.update(
            status="data_ready_only",
            data_ready=True,
            data_ready_for_preflight=False,
            init_selection="pending_parent_selection",
            pending_parent_selection=True,
            init_adapter_role="verified CE0 placeholder; next parent not selected",
            training_authorized=False,
        )
    h.write(out / "provenance.json", result)
    h.write(out / "report.json", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    for name in ["exposure", "claim", "video"]:
        p.add_argument("--" + name + "-report-sha256", required=True)
    a = p.parse_args()
    result = prepare(
        a.root, a.output_dir, a.exposure_report_sha256, a.claim_report_sha256, a.video_report_sha256
    )
    print(
        json.dumps(
            {k: result[k] for k in ["status", "exit_code", "total_draws", "inventory_isolation"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
