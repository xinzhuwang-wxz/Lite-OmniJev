"""Fixed sealed holdout evaluation: aggregate exports only, never temperature fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, FiniteFloat, model_validator

from data.schema import Contract, HardTarget, Sample, answer_tokens
from eval.gate import evaluate_gate
from eval.metrics import evaluate
from scripts import private_calibration as cal
from scripts import private_stage_eval as stage
from scripts import s0_development_baselines as baseline
from scripts.check_tokens import probe
from scripts.s0_public_baselines import sha256_file
from train.calibration import Binding, Cohort, TemperatureArtifact, load_temperature
from train.readout import read_probabilities

GROUP_COUNTS = {key: 500 if key.startswith("audio/") else 1000 for key in cal.GROUP_COUNTS}


class TemperatureReference(Contract):
    artifact: stage.Artifact
    binding: Binding


class DevelopmentReference(Contract):
    cohort: Cohort
    report: stage.Artifact
    runtime_report: stage.Artifact | None = None
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class JobSpec(Contract):
    schema_version: Literal["private-holdout-job-v1"] = "private-holdout-job-v1"
    datasets: list[stage.Spec] = Field(min_length=7, max_length=7)
    producer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_python: stage.Artifact
    inventory: stage.Artifact
    inventory_receipt: stage.Artifact
    root_traverse_acl: bool = False
    max_snapshot_bytes: int = Field(default=64 * 1024**3, gt=0, le=128 * 1024**3)
    temperatures: list[TemperatureReference] = Field(default_factory=list, max_length=7)
    calibration_receipt: stage.Artifact | None = None
    calibration_raw_receipt: stage.Artifact | None = None
    development: list[DevelopmentReference] = Field(default_factory=list, max_length=6)
    gates: stage.Artifact

    @model_validator(mode="after")
    def frozen_scope(self) -> JobSpec:
        first = self.datasets[0]
        totals: Counter[str] = Counter()
        for source in self.datasets:
            cal.require(
                source.mode in ("S4", "S5")
                and source.pool == "holdout"
                and source.subset_limit is None,
                "full_holdout_only",
            )
            for field in (
                "model_dir",
                "model_files",
                "adapter_dir",
                "adapter_files",
                "base_revision",
                "selected_manifest",
                "data_admission",
                "evidence_root",
                "mode",
            ):
                cal.require(getattr(source, field) == getattr(first, field), "one_frozen_identity")
            totals[source.cohort] += source.expected_rows
        cal.require(
            dict(totals)
            == {c: 1000 for c in ("text", "image", "screenshot", "audio", "video", "crossmodal")},
            "six_full_holdout_cohorts_required",
        )
        cal.require(
            first.adapter_dir is not None and first.adapter_files is not None, "adapter_required"
        )
        keys = [f"{r.binding.cohort}/{r.binding.primitive}" for r in self.temperatures]
        cal.require(
            not keys or (len(keys) == 7 and set(keys) == set(GROUP_COUNTS)),
            "seven_real_temperatures_only",
        )
        cal.require(
            bool(keys) == (self.calibration_receipt is not None)
            and bool(keys) == (self.calibration_raw_receipt is not None),
            "fit_receipt_required",
        )
        cal.require(
            len({d.cohort for d in self.development}) == len(self.development),
            "duplicate_dev_cohort",
        )
        return self


def snapshot_extras(
    spec: JobSpec, inputs: Path, copy_bound: Callable[[Path, str, Path], None]
) -> dict[str, Any]:
    """Snapshot public aggregate/temperature metadata; no private rows are parsed here."""
    first = spec.datasets[0]
    assert first.selected_manifest is not None
    selected = json.loads(stage.binding(first.selected_manifest).read_text())
    for ref in spec.development:
        cal.require(
            any(
                item["cohort"] == ref.cohort
                and item["pool"] == "dev"
                and item["sha256"] == ref.data_sha256
                and item["frozen_original_dev"] is True
                for item in selected["selected"]
            ),
            "not_original_selected_dev",
        )
    copied: dict[str, str] = {}
    refs = [("gates", spec.gates)]
    refs += [(f"temperature-{i}", ref.artifact) for i, ref in enumerate(spec.temperatures)]
    refs += [(f"dev-{i}", ref.report) for i, ref in enumerate(spec.development)]
    refs += [
        (f"dev-runtime-{i}", ref.runtime_report)
        for i, ref in enumerate(spec.development)
        if ref.runtime_report is not None
    ]
    if spec.calibration_receipt is not None:
        refs.append(("calibration-receipt", spec.calibration_receipt))
    if spec.calibration_raw_receipt is not None:
        refs.append(("calibration-raw-receipt", spec.calibration_raw_receipt))
    for key, artifact_ref in refs:
        path = inputs / "metadata" / (key + ".json")
        copy_bound(Path(artifact_ref.path), artifact_ref.sha256, path)
        copied[key] = str(path)
    return {"metadata": copied}


def private_samples(state: dict[str, Any]) -> tuple[JobSpec, list[tuple[str, Sample, str]]]:
    spec = JobSpec.model_validate(state["spec"])
    counts: Counter[str] = Counter()
    rows: list[tuple[str, Sample, str]] = []
    identifiers: set[str] = set()
    for source, snapshot in zip(spec.datasets, state["datasets"], strict=True):
        path = Path(snapshot)
        cal.require(sha256_file(path) == source.data.sha256, "snapshot_data_changed")
        samples = [
            Sample.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        cal.require(len(samples) == source.expected_rows, "wrong_holdout_count")
        for sample in samples:
            cal.require(sample.id not in identifiers, "duplicate_holdout_sample")
            identifiers.add(sample.id)
            cal.require(
                sample.meta.pool == "holdout" and len(sample.questions) == 1,
                "wrong_pool_or_question_count",
            )
            question = sample.questions[0]
            cal.require(
                isinstance(question.target, HardTarget)
                and question.target.value in answer_tokens(question),
                "official_known_hard_gold_required",
            )
            media = sorted(m.type for m in sample.prefix.media)
            expected = (
                []
                if source.cohort == "text"
                else ["audio", "image"]
                if source.cohort == "crossmodal"
                else ["image" if source.cohort == "screenshot" else source.cohort]
            )
            cal.require(media == expected, "cohort_media_mismatch")
            key = f"{source.cohort}/{question.type}"
            cal.require(key in GROUP_COUNTS, "unsupported_holdout_group")
            counts[key] += 1
            rows.append((source.cohort, sample, source.media_root))
    cal.require(dict(counts) == GROUP_COUNTS, "actual_holdout_group_counts_mismatch")
    return spec, rows


def plan_worker(state_path: Path) -> None:
    state = json.loads(state_path.read_text())
    _, samples = private_samples(state)
    media: dict[str, dict[str, str]] = {}
    for _, sample, root in samples:
        for item in sample.prefix.media:
            path = cal.lexical_media(root, item.uri)
            entry = {"sha256": item.sha256, "type": item.type}
            cal.require(path not in media or media[path] == entry, "conflicting_media")
            media[path] = entry
    cal.write_private(
        Path(state["output"]) / "media-plan.json", {"groups": GROUP_COUNTS, "media": media}
    )


def temperatures(
    spec: JobSpec, state: dict[str, Any], official: dict[str, Any]
) -> dict[str, TemperatureArtifact]:
    if not spec.temperatures:
        return {}
    first = spec.datasets[0]
    assert first.adapter_files is not None and spec.calibration_receipt is not None
    path = Path(state["metadata"]["calibration-receipt"])
    cal.require(sha256_file(path) == spec.calibration_receipt.sha256, "fit_receipt_changed")
    receipt = json.loads(path.read_text())
    cal.require(
        receipt["status"] == "fitted"
        and receipt["exit_code"] == 0
        and receipt["ordinary_identity_boundary"] is True
        and receipt["boundary"] == "fresh_ordinary_uid_fixed_calibration_job",
        "untrusted_fit_receipt",
    )
    assert spec.calibration_raw_receipt is not None
    raw_path = Path(state["metadata"]["calibration-raw-receipt"])
    cal.require(
        sha256_file(raw_path) == spec.calibration_raw_receipt.sha256
        and spec.calibration_raw_receipt.sha256 == receipt["raw_logits_receipt_sha256"],
        "calibration_raw_receipt_binding_mismatch",
    )
    raw = json.loads(raw_path.read_text())
    cal.require(
        raw["runtime_recipe"]
        == {"engine": "transformers", "dtype": "bfloat16", "adapter_mode": "native_lora"}
        and raw["model_files"] == first.model_files
        and raw["adapter_files"] == first.adapter_files
        and raw["official_template_sha256"] == official["template_sha256"]
        and raw["answer_token_map_sha256"]
        == hashlib.sha256(
            json.dumps(official["answer_token_ids"], sort_keys=True).encode()
        ).hexdigest()
        and raw["spec_sha256"] == receipt["spec_sha256"]
        and raw["code_sha256"] == receipt["code_sha256"]
        and raw["logits_kind"] == "raw_logits"
        and raw["complete_legal_logits"] is True,
        "calibration_runtime_provenance_mismatch",
    )
    entries = {e["group"]: e for e in receipt["artifacts"]}
    cal.require(
        set(entries) == set(GROUP_COUNTS) and len(receipt["artifacts"]) == 7, "fit_groups_mismatch"
    )
    mapping_sha = hashlib.sha256(
        json.dumps(official["answer_token_ids"], sort_keys=True).encode()
    ).hexdigest()
    result = {}
    for index, ref in enumerate(spec.temperatures):
        key = f"{ref.binding.cohort}/{ref.binding.primitive}"
        entry = entries[key]
        cal.require(
            entry["sha256"] == ref.artifact.sha256 and entry["binding"] == ref.binding.model_dump(),
            "temperature_not_in_fit_receipt",
        )
        cal.require(
            ref.binding.base_revision == first.base_revision
            and ref.binding.adapter_weights_sha256
            == first.adapter_files["adapter_model.safetensors"]
            and ref.binding.official_template_sha256 == official["template_sha256"]
            and ref.binding.answer_token_map_sha256 == mapping_sha
            and ref.binding.raw_logits_receipt_sha256 == receipt["raw_logits_receipt_sha256"],
            "temperature_model_identity_mismatch",
        )
        path = Path(state["metadata"][f"temperature-{index}"])
        cal.require(sha256_file(path) == ref.artifact.sha256, "temperature_changed")
        artifact = load_temperature(path, ref.binding)
        cal.require(artifact.rows == cal.GROUP_COUNTS[key], "calibration_count_mismatch")
        result[key] = artifact
    return result


def summarize(rows: list[tuple[list[float], int, float, float]]) -> dict[str, Any]:
    values = evaluate([r[0] for r in rows], [r[1] for r in rows], max_error=0.05)
    return {
        **values,
        "correct": sum(max(range(len(r[0])), key=r[0].__getitem__) == r[1] for r in rows),
        "mean_p_insufficient": math.fsum(r[2] for r in rows) / len(rows),
        "joint_legal_plus_u_nll": math.fsum(r[3] for r in rows) / len(rows),
    }


def append_readout(
    bucket: list[Any], logits: dict[str, float], options: list[str], target: str, temperature: float
) -> None:
    readout = read_probabilities(logits, options, temperature=temperature)
    maximum = max(logits.values())
    nll = (maximum - logits[target]) / temperature + math.log(
        math.fsum(math.exp((v - maximum) / temperature) for v in logits.values())
    )
    bucket.append(
        (
            [readout.probabilities[o] for o in options],
            options.index(target),
            readout.p_insufficient,
            nll,
        )
    )


def dev_runtime_categories(receipt: dict[str, Any], cohort: str, base_revision: str) -> list[str]:
    """Recognize actual report layouts; no inferred default budgets or precision."""
    errors = []
    if receipt.get("model_revision") != base_revision:
        errors.append("base_revision_missing_or_mismatched")
    # This exact checked-in evaluator loads dtype=torch.bfloat16 and has no dtype option.
    known_bf16_source = "12d387e778a685a8f49ef223f5fa2d26538e3e6a052c7c8e5013677186ad9c68"
    source = receipt.get("script_sha256") or receipt.get("source_file_sha256", {}).get(
        "scripts/s1_dev_eval.py"
    )
    dtype = receipt.get("dtype")
    if dtype not in (None, "bfloat16") or (dtype is None and source != known_bf16_source):
        errors.append("bf16_evidence_missing_or_mismatched")
    budget = receipt.get("image_max_soft_tokens")
    effective = budget.get("effective") if isinstance(budget, dict) else budget
    if cohort != "text" and effective != 1120:
        errors.append("image_budget_not_proven_1120")
    if cohort == "video" and receipt.get("video_num_frames") != 16:
        errors.append("video_frames_not_proven_16")
    return errors


def development_metrics(
    spec: JobSpec, state: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    result: dict[str, dict[str, Any]] = {}
    statuses: dict[str, list[str]] = {}
    first = spec.datasets[0]
    assert first.adapter_files is not None
    for index, ref in enumerate(spec.development):
        path = Path(state["metadata"][f"dev-{index}"])
        cal.require(sha256_file(path) == ref.report.sha256, "dev_receipt_changed")
        receipt = json.loads(path.read_text())
        cal.require(
            receipt["status"] in ("completed", "passed")
            and receipt["exit_code"] == 0
            and receipt["adapter_sha256"] == first.adapter_files["adapter_model.safetensors"]
            and receipt["data_sha256"] == ref.data_sha256
            and receipt["metrics"]["count"] == 500,
            "unmatched_full_dev_receipt",
        )
        runtime = receipt
        if ref.runtime_report is not None:
            runtime_path = Path(state["metadata"][f"dev-runtime-{index}"])
            cal.require(
                sha256_file(runtime_path) == ref.runtime_report.sha256,
                "dev_runtime_receipt_changed",
            )
            if "report_sha256" in receipt:
                cal.require(
                    receipt["report_sha256"] == ref.runtime_report.sha256,
                    "dev_nested_receipt_mismatch",
                )
            runtime = json.loads(runtime_path.read_text())
            cal.require(
                runtime.get("exit_code") == 0
                and runtime.get("adapter_sha256") == receipt["adapter_sha256"]
                and runtime.get("data_sha256") == receipt["data_sha256"]
                and runtime.get("metrics") == receipt["metrics"],
                "dev_runtime_identity_mismatch",
            )
        statuses[ref.cohort] = dev_runtime_categories(runtime, ref.cohort, first.base_revision)
        if statuses[ref.cohort]:
            continue  # Raw holdout remains valid; unmatched DEV recipe cannot justify a gap pass.
        # Explicit typed subset; never propagate public row/record fields to sealed output.
        result[ref.cohort] = stage.Metrics.model_validate(
            {key: receipt["metrics"][key] for key in stage.Metrics.model_fields}
        ).model_dump()
    return result, statuses


def metric_deltas(raw: dict[str, Any], scaled: dict[str, Any] | None) -> dict[str, Any]:
    return (
        {
            section: {
                key: {
                    metric: values[metric] - raw[section][key][metric]
                    for metric in (
                        "accuracy",
                        "brier",
                        "ece15",
                        "coverage_at_error",
                        "joint_legal_plus_u_nll",
                    )
                }
                for key, values in scaled[section].items()
            }
            for section in ("cohorts", "groups")
        }
        if scaled
        else {}
    )


def quality_observations(
    raw: dict[str, dict[str, Any]],
    calibrated: dict[str, dict[str, Any]],
    development: dict[str, dict[str, Any]],
    gates: dict[str, Any],
    development_runtime: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    fixed = {
        "S4": {"max_holdout_dev_gap": {"max": 0.03}},
        "S5": {
            "max_modality_ece": {"max": 0.03},
            "coverage_at_5pct_error": {"min": 0.65},
            "accuracy_regression": {"max": 0.005},
        },
    }
    for stage_name, bounds in fixed.items():
        cal.require(
            all(gates["stages"][stage_name][key] == value for key, value in bounds.items()),
            "original_stage_thresholds_required",
        )
    gaps = {c: abs(raw[c]["accuracy"] - d["accuracy"]) for c, d in development.items()}
    s4 = evaluate_gate(
        {"max_holdout_dev_gap": max(gaps.values())} if len(gaps) == 6 else {},
        {"max_holdout_dev_gap": gates["stages"]["S4"]["max_holdout_dev_gap"]},
    )
    s5_values = (
        {
            "max_modality_ece": max(m["ece15"] for m in calibrated.values()),
            "coverage_at_5pct_error": min(m["coverage_at_error"] for m in calibrated.values()),
            "accuracy_regression": max(
                raw[c]["accuracy"] - m["accuracy"] for c, m in calibrated.items()
            ),
        }
        if len(calibrated) == 6
        else {}
    )
    s5 = evaluate_gate(
        s5_values,
        {
            key: gates["stages"]["S5"][key]
            for key in ("max_modality_ece", "coverage_at_5pct_error", "accuracy_regression")
        },
    )
    return {
        "holdout_dev_absolute_accuracy_gaps": gaps,
        "development_runtime_checks": development_runtime or {},
        "development_gap_status": "ready"
        if len(gaps) == 6
        else "pending_runtime_or_missing_evidence",
        "s4_gap_observation": s4,
        "s5_probability_observations": s5,
        "stage_acceptance": "not_evaluated",
        "missing_stage_evidence": [
            "teacher ratios",
            "DecisionIndex",
            "JevBench",
            "permutation consistency",
            "noul pair consistency",
            "R1 conclusion",
        ],
        "source_comparability": (
            "selected holdout and original DEV source distributions differ; no IID claim"
        ),
        "coverage_scope": (
            "minimum of six cohort coverage values; diagnostic only, no threshold fit"
        ),
    }


def evaluate_worker(state_path: Path) -> None:
    started = time.monotonic()
    state = json.loads(state_path.read_text())
    spec, samples = private_samples(state)
    first = spec.datasets[0]
    assert first.adapter_files is not None
    model_dir, adapter_dir = Path(state["model"]), Path(state["adapter"])
    stage.bind_files(str(model_dir), first.model_files)
    stage.bind_files(str(adapter_dir), first.adapter_files, adapter=True)
    official = probe(model_dir)
    cal.require(
        official["status"] == "passed" and len(official["answer_token_ids"]) == 33,
        "official_probe_failed",
    )
    fitted = temperatures(spec, state, official)
    development, dev_runtime = development_metrics(spec, state)
    gates_path = Path(state["metadata"]["gates"])
    cal.require(sha256_file(gates_path) == spec.gates.sha256, "gates_changed")
    gates = json.loads(gates_path.read_text())
    processor, forward, model = cal.raw_backend(model_dir, adapter_dir)
    media_map = json.loads(Path(state["media_map"]).read_text())
    raw_groups: dict[str, list[Any]] = {key: [] for key in GROUP_COUNTS}
    scaled_groups: dict[str, list[Any]] = {key: [] for key in fitted}
    try:
        cal.worker_progress(state, "inference", 0, started)
        for index, (cohort, original, root) in enumerate(samples):
            media = [
                m.model_copy(update={"uri": media_map[cal.lexical_media(root, m.uri)]})
                for m in original.prefix.media
            ]
            sample = original.model_copy(
                update={"prefix": original.prefix.model_copy(update={"media": media})}
            )
            prepared = baseline.prepare_sample(sample, processor, Path(state["media_root"]))
            question = sample.questions[0]
            assert isinstance(question.target, HardTarget)
            options = list(answer_tokens(question))
            cal.require(
                prepared.answer_token_ids
                == {o: official["answer_token_ids"][o] for o in [*options, "U"]},
                "answer_map_changed",
            )
            logits = cal.forward_private(forward, prepared, state, index, started, cohort)
            key = f"{cohort}/{question.type}"
            append_readout(raw_groups[key], logits, options, str(question.target.value), 1.0)
            if key in fitted:
                append_readout(
                    scaled_groups[key],
                    logits,
                    options,
                    str(question.target.value),
                    fitted[key].temperature,
                )
            if (
                (index + 1) % 100 == 0
                or index + 1 == len(samples)
                or samples[index + 1][0] != cohort
            ):
                cal.worker_progress(state, "inference", index + 1, started, cohort)
        raw = {key: summarize(rows) for key, rows in raw_groups.items()}
        scaled = {key: summarize(rows) for key, rows in scaled_groups.items()}
        cohorts = sorted({key.split("/")[0] for key in GROUP_COUNTS})

        def combine(groups: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
            return (
                {
                    c: summarize(
                        [
                            row
                            for key, rows in groups.items()
                            if key.startswith(c + "/")
                            for row in rows
                        ]
                    )
                    for c in cohorts
                }
                if groups
                else {}
            )

        raw_cohorts, scaled_cohorts = combine(raw_groups), combine(scaled_groups)
        result = {
            "schema_version": "private-holdout-receipt-v1",
            "status": "completed",
            "exit_code": 0,
            "spec_sha256": state["source_spec_sha256"],
            "code_sha256": state["code_sha256"],
            "producer_revision": spec.producer_revision,
            "base_revision": first.base_revision,
            "adapter_sha256": first.adapter_files["adapter_model.safetensors"],
            "adapter_config_sha256": first.adapter_files["adapter_config.json"],
            "official_template_sha256": official["template_sha256"],
            "answer_token_map_sha256": hashlib.sha256(
                json.dumps(official["answer_token_ids"], sort_keys=True).encode()
            ).hexdigest(),
            "runtime_recipe": {
                "engine": "transformers",
                "dtype": "bfloat16",
                "adapter_mode": "native_lora",
            },
            "logits_kind": "raw_logits",
            "dtype": "bfloat16",
            "pool": "holdout",
            "total_count": len(samples),
            "raw": {"cohorts": raw_cohorts, "groups": raw},
            "calibrated": {"cohorts": scaled_cohorts, "groups": scaled} if fitted else None,
            "calibration_receipt_sha256": spec.calibration_receipt.sha256
            if spec.calibration_receipt
            else None,
            "unpopulated_groups": sorted(cal.ALL_GROUPS - GROUP_COUNTS.keys()),
            "raw_calibrated_deltas": metric_deltas(
                {"cohorts": raw_cohorts, "groups": raw},
                {"cohorts": scaled_cohorts, "groups": scaled} if fitted else None,
            ),
            "quality": quality_observations(
                raw_cohorts, scaled_cohorts, development, gates, dev_runtime
            ),
            "family_isolation_source": {
                "inventory_sha256": spec.inventory.sha256,
                "receipt_sha256": spec.inventory_receipt.sha256,
            },
            "dataset_bindings": [
                {"cohort": s.cohort, "sha256": s.data.sha256, "rows": s.expected_rows}
                for s in spec.datasets
            ],
            "gates_sha256": spec.gates.sha256,
            "wall_seconds": time.monotonic() - started,
        }
        cal.write_private(Path(state["output"]) / "receipt.json", result)
    finally:
        del model


class AggregateMetrics(stage.Metrics):
    correct: int = Field(ge=0)
    mean_p_insufficient: FiniteFloat = Field(ge=0, le=1)
    joint_legal_plus_u_nll: FiniteFloat = Field(ge=0)


def export_receipt(
    spec: JobSpec, output: Path, receipt: dict[str, Any], expected: str, code_sha: str
) -> None:
    """Allowlist aggregate structure; never export worker-selected files or raw records."""
    first = spec.datasets[0]
    assert first.adapter_files is not None and first.adapter_dir is not None
    required = {
        "schema_version",
        "status",
        "exit_code",
        "spec_sha256",
        "code_sha256",
        "producer_revision",
        "base_revision",
        "adapter_sha256",
        "adapter_config_sha256",
        "official_template_sha256",
        "answer_token_map_sha256",
        "runtime_recipe",
        "logits_kind",
        "dtype",
        "pool",
        "total_count",
        "raw",
        "calibrated",
        "calibration_receipt_sha256",
        "raw_calibrated_deltas",
        "unpopulated_groups",
        "quality",
        "family_isolation_source",
        "dataset_bindings",
        "gates_sha256",
        "wall_seconds",
    }
    cal.require(set(receipt) == required, "unexpected_holdout_export_fields")
    cal.require(
        receipt["status"] == "completed"
        and receipt["exit_code"] == 0
        and receipt["total_count"] == 6000
        and receipt["spec_sha256"] == expected
        and receipt["code_sha256"] == code_sha
        and receipt["producer_revision"] == spec.producer_revision
        and receipt["pool"] == "holdout"
        and receipt["logits_kind"] == "raw_logits"
        and receipt["dtype"] == "bfloat16",
        "invalid_holdout_receipt",
    )
    cal.require(stage.code_hash() == code_sha, "code_changed_during_job")
    stage.bind_files(first.model_dir, first.model_files)
    stage.bind_files(first.adapter_dir, first.adapter_files, adapter=True)
    official = probe(Path(first.model_dir))
    cal.require(
        receipt["base_revision"] == first.base_revision
        and receipt["adapter_sha256"] == first.adapter_files["adapter_model.safetensors"]
        and receipt["adapter_config_sha256"] == first.adapter_files["adapter_config.json"]
        and receipt["official_template_sha256"] == official["template_sha256"]
        and receipt["answer_token_map_sha256"]
        == hashlib.sha256(
            json.dumps(official["answer_token_ids"], sort_keys=True).encode()
        ).hexdigest(),
        "holdout_model_identity_mismatch",
    )
    for branch in ("raw", "calibrated"):
        values = receipt[branch]
        if values is None:
            cal.require(branch == "calibrated" and not spec.temperatures, "missing_holdout_metrics")
            continue
        cal.require(
            set(values) == {"cohorts", "groups"} and set(values["groups"]) == set(GROUP_COUNTS),
            "invalid_holdout_metric_groups",
        )
        cal.require(
            set(values["cohorts"]) == {k.split("/")[0] for k in GROUP_COUNTS},
            "invalid_holdout_cohorts",
        )
        for key, metric in {**values["groups"], **values["cohorts"]}.items():
            allowed = {
                *stage.Metrics.model_fields,
                "correct",
                "mean_p_insufficient",
                "joint_legal_plus_u_nll",
            }
            cal.require(set(metric) == allowed, "unexpected_metric_fields")
            AggregateMetrics.model_validate(metric)
            cal.require(
                type(metric["correct"]) is int
                and metric["correct"] <= metric["count"]
                and math.isclose(metric["accuracy"], metric["correct"] / metric["count"]),
                "invalid_correct_count",
            )
            cal.require(
                metric["count"] == GROUP_COUNTS.get(key, 1000), "incomplete_holdout_metrics"
            )
    cal.require(
        receipt["unpopulated_groups"] == sorted(cal.ALL_GROUPS - GROUP_COUNTS.keys()),
        "invalid_unpopulated_groups",
    )
    cal.require(
        bool(receipt["calibrated"]) == bool(spec.temperatures), "calibration_presence_mismatch"
    )
    cal.require(
        receipt["schema_version"] == "private-holdout-receipt-v1"
        and receipt["runtime_recipe"]
        == {"engine": "transformers", "dtype": "bfloat16", "adapter_mode": "native_lora"}
        and receipt["calibration_receipt_sha256"]
        == (spec.calibration_receipt.sha256 if spec.calibration_receipt else None)
        and receipt["family_isolation_source"]
        == {
            "inventory_sha256": spec.inventory.sha256,
            "receipt_sha256": spec.inventory_receipt.sha256,
        }
        and receipt["dataset_bindings"]
        == [
            {"cohort": s.cohort, "sha256": s.data.sha256, "rows": s.expected_rows}
            for s in spec.datasets
        ]
        and receipt["gates_sha256"] == spec.gates.sha256,
        "holdout_source_binding_mismatch",
    )
    cal.require(
        type(receipt["wall_seconds"]) in (float, int)
        and math.isfinite(receipt["wall_seconds"])
        and receipt["wall_seconds"] >= 0,
        "invalid_duration",
    )
    metadata = {
        f"dev-{i}": str(output.parent / "inputs" / "metadata" / f"dev-{i}.json")
        for i in range(len(spec.development))
    }
    metadata.update(
        {
            f"dev-runtime-{i}": str(output.parent / "inputs" / "metadata" / f"dev-runtime-{i}.json")
            for i, ref in enumerate(spec.development)
            if ref.runtime_report is not None
        }
    )
    dev, dev_runtime = development_metrics(spec, {"metadata": metadata})
    gates = json.loads(stage.binding(spec.gates).read_text())
    scaled = receipt["calibrated"]
    expected_quality = quality_observations(
        receipt["raw"]["cohorts"], scaled["cohorts"] if scaled else {}, dev, gates, dev_runtime
    )
    cal.require(receipt["quality"] == expected_quality, "invalid_quality_export")
    cal.require(
        receipt["raw_calibrated_deltas"] == metric_deltas(receipt["raw"], scaled),
        "invalid_delta_export",
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("phase", choices=["_runtime_probe", "_plan", "_evaluate"])
    parser.add_argument("--state", type=Path, required=True)
    args = None
    try:
        args = parser.parse_args()
        cal.verify_child_identity()
        if args.phase == "_runtime_probe":
            import peft
            import torch
            import transformers

            cal.require(
                bool(peft.__version__ and torch.__version__ and transformers.__version__),
                "runtime_unavailable",
            )
        elif args.phase == "_plan":
            plan_worker(args.state)
        else:
            evaluate_worker(args.state)
        return 0
    except BaseException as error:
        if args is not None:
            try:
                cal.record_worker_failure(
                    args.state,
                    {"_plan": "plan", "_evaluate": "inference", "_runtime_probe": "runtime"}[
                        args.phase
                    ],
                    error,
                )
            except BaseException:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
