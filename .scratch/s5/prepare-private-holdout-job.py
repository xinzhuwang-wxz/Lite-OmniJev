"""Metadata-only holdout job assembly; no private dataset bytes are opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import private_calibration as cal
from scripts import private_holdout as holdout
from scripts import private_stage_eval as stage
from scripts.s0_public_baselines import sha256_file
from train.calibration import Binding


def artifact(path: Path) -> stage.Artifact:
    return stage.Artifact(path=str(path), sha256=sha256_file(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-job", type=Path, required=True)
    parser.add_argument("--calibration-job-sha256", required=True)
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument(
        "--development", action="append", default=[], help="cohort=/absolute/aggregate-report.json"
    )
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--development-runtime", action="append", default=[])
    parser.add_argument("--video-input-parity", type=Path)
    args = parser.parse_args()
    cal.require(
        sha256_file(args.calibration_job) == args.calibration_job_sha256, "calibration_job_changed"
    )
    prior = cal.JobSpec.model_validate_json(args.calibration_job.read_bytes())
    first = prior.datasets[0]
    assert first.selected_manifest is not None and first.data_admission is not None
    selected = json.loads(stage.binding(first.selected_manifest).read_text())
    admission = json.loads(stage.binding(first.data_admission).read_text())
    inventory = json.loads(stage.binding(prior.inventory).read_text())
    audit = json.loads(stage.binding(prior.inventory_receipt).read_text())
    cal.require(
        audit["outcome"] == "isolated"
        and audit["ordinary_identity_boundary"] is True
        and audit["source_manifest_sha256"] == prior.inventory.sha256,
        "inventory_binding_mismatch",
    )
    registered = {
        (r["path"], r["sha256"])
        for r in inventory["samples"]
        if r["expected_pool"] == "holdout" and r["exposure"] == "current"
    }
    entries = [r for r in selected["selected"] if r["pool"] == "holdout"]
    cal.require(len(entries) == 7, "seven_full_holdout_files_required")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    datasets = []
    for index, row in enumerate(entries):
        cal.require((row["path"], row["sha256"]) in registered, "holdout_not_registered")
        cal.require(Path(row["path"]).is_file(), "missing_holdout_file")  # stat only
        template = next(s for s in prior.datasets if s.cohort == row["cohort"])
        declaration = stage.Admission(
            schema_version="private-stage-data-admission-v1",
            status="admitted",
            data_sha256=row["sha256"],
            selected_manifest_sha256=first.selected_manifest.sha256,
            evidence_bundle_sha256=admission["evidence_bundle_sha256"],
            pool="holdout",
            cohort=row["cohort"],
            expected_rows=row["expected_questions"],
        )
        path = args.output / f"admission-{index:02d}.json"
        cal.write_private(path, declaration.model_dump())
        values = template.model_dump()
        values.update(
            mode="S5" if args.calibration_receipt else "S4",
            pool="holdout",
            data={"path": row["path"], "sha256": row["sha256"]},
            expected_rows=row["expected_questions"],
            admission=artifact(path).model_dump(),
        )
        source = stage.Spec.model_validate(values)
        stage.verify_catalog(source)
        datasets.append(source)
    runtime_reports = dict(value.split("=", 1) for value in args.development_runtime)
    development = []
    for value in args.development:
        cohort, raw_path = value.split("=", 1)
        report_path = Path(raw_path)
        report = json.loads(report_path.read_text())
        cal.require(
            report["exit_code"] == 0
            and report["metrics"]["count"] == 500
            and any(
                r["cohort"] == cohort
                and r["pool"] == "dev"
                and r["frozen_original_dev"]
                and r["sha256"] == report["data_sha256"]
                for r in selected["selected"]
            ),
            "full_original_dev_required",
        )
        development.append(
            holdout.DevelopmentReference.model_validate(
                {
                    "cohort": cohort,
                    "report": artifact(report_path).model_dump(),
                    "data_sha256": report["data_sha256"],
                    "video_input_parity": artifact(args.video_input_parity).model_dump()
                    if cohort == "video" and args.video_input_parity
                    else None,
                    "runtime_report": artifact(Path(runtime_reports[cohort])).model_dump()
                    if cohort in runtime_reports
                    else None,
                }
            )
        )
    temperatures = []
    fit_receipt = None
    raw_receipt = None
    if args.calibration_receipt:
        fit_receipt = artifact(args.calibration_receipt)
        report = json.loads(args.calibration_receipt.read_text())
        cal.require(
            report["status"] == "fitted" and report["exit_code"] == 0, "completed_fit_required"
        )
        raw_receipt = artifact(Path(report["export_directory"]) / "raw-receipt.json")
        cal.require(
            raw_receipt.sha256 == report["raw_logits_receipt_sha256"], "raw_receipt_changed"
        )
        for row in report["artifacts"]:
            path = Path(report["export_directory"]) / row["path"]
            cal.require(sha256_file(path) == row["sha256"], "temperature_changed")
            temperatures.append(
                holdout.TemperatureReference(
                    artifact=artifact(path), binding=Binding.model_validate(row["binding"])
                )
            )
    job = holdout.JobSpec(
        datasets=datasets,
        producer_revision=args.producer_revision,
        runtime_python=prior.runtime_python,
        inventory=prior.inventory,
        inventory_receipt=prior.inventory_receipt,
        root_traverse_acl=prior.root_traverse_acl,
        max_snapshot_bytes=prior.max_snapshot_bytes,
        temperatures=temperatures,
        calibration_receipt=fit_receipt,
        calibration_raw_receipt=raw_receipt,
        development=development,
        gates=artifact(args.gates),
    )
    path = args.output / "job.json"
    cal.write_private(path, job.model_dump())
    result = {
        "status": "metadata_prepared",
        "job": artifact(path).model_dump(),
        "holdout_count": 6000,
        "dataset_files": 7,
        "audio_counts": {"choice": 500, "noul": 500},
        "development_cohorts": len(development),
        "temperature_groups": len(temperatures),
        "private_contents_read": False,
        "gpu_started": False,
        "stage_acceptance": "not_evaluated",
        "producer_revision": args.producer_revision,
        "producer_sha256": sha256_file(Path(holdout.__file__)),
        "source_code_sha256": stage.code_hash(),
    }
    cal.write_private(args.output / "preparation-report.json", result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
