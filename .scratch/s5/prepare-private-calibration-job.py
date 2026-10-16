"""Prepare only metadata bindings; never open private sample, logits, or log files."""

from __future__ import annotations

import argparse
import json
import stat
import sys
import time
from pathlib import Path

from scripts import private_calibration as cal
from scripts import private_gold_job as broker
from scripts import private_stage_eval as stage
from scripts.s0_public_baselines import sha256_file

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
PRIVATE = ROOT / "artifacts/s4-gold-inventory-001"
EXPECTED = {
    "selected": "e073a601eab235644f91f058a99dd331c568de2560660441192778553429876f",
    "admission": "fd5a42f6a311a2b6cb5d91cf473c46ee532c65b6abef13cd463fd118e7713c7a",
    "inventory": "807003d0dfc3443f69f3d992566c62d60e7e7824db2655e1347e8bb9d546d6b8",
    "inventory_receipt": "e14f7dd48d07bffe779ea1998e0e17bdb044a70438255d930173aa5241aa6ea0",
    "training": "421b950611a8e8db3fa1b0ae75c960790ae56c29a80a218976c171fbdf67ae97",
    "adapter": "e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd",
}


def bind(path: Path, expected: str | None = None) -> dict[str, str]:
    # Called only for public metadata, source, runtime and model/adapter assets.
    digest = sha256_file(path)
    cal.require(expected is None or digest == expected, "metadata_binding_changed")
    return {"path": str(path), "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/s5-calibration-job-001")
    parser.add_argument("--producer-revision", default="daeaa7f5a5fccaeb055fc54ea5c163025b3690ac")
    parser.add_argument("--timeout", type=int, default=14400)
    args = parser.parse_args()
    start = time.monotonic()
    selected_path = PRIVATE / "selected-gold-manifest-001.json"
    admission_path = ROOT / "reports/s4-gold-data-admission-001.json"
    inventory_path = ROOT / "artifacts/s4-pilot-exposure-028/inventory-manifest.json"
    inventory_receipt_path = ROOT / "reports/s5-calibration-inventory-001.json"
    training_path = ROOT / "reports/s4-program-stratified-training-step0100.json"
    bindings = {
        key: bind(path, EXPECTED[key])
        for key, path in [
            ("selected", selected_path),
            ("admission", admission_path),
            ("inventory", inventory_path),
            ("inventory_receipt", inventory_receipt_path),
            ("training", training_path),
        ]
    }
    selected = json.loads(selected_path.read_text())
    admission = json.loads(admission_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    audit = json.loads(inventory_receipt_path.read_text())
    training = json.loads(training_path.read_text())
    cal.require(
        audit["outcome"] == "isolated"
        and audit["ordinary_identity_boundary"] is True
        and audit["source_manifest_sha256"] == EXPECTED["inventory"],
        "inventory_not_current",
    )
    cal.require(
        training["exit_code"] == 0
        and training["steps_completed"] == 100
        and training["adapter_sha256"] == EXPECTED["adapter"],
        "candidate_not_frozen",
    )
    model = (
        Path("/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it")
        / stage.FROZEN_REVISION
    )
    adapter = ROOT / "artifacts/s4-program-stratified-pool-train-001/adapter-invocation-step-0100"
    model_files = {
        name: digest
        for name, digest in training["asset_hashes"].items()
        if Path(name).suffix in (".json", ".safetensors", ".jinja", ".model")
    }
    stage.bind_files(str(model), model_files)
    adapter_files = {
        path.name: sha256_file(path)
        for path in adapter.iterdir()
        if path.suffix in (".json", ".safetensors")
    }
    stage.bind_files(str(adapter), adapter_files, adapter=True)
    cal.require(
        adapter_files["adapter_model.safetensors"] == EXPECTED["adapter"], "adapter_changed"
    )
    entries = [item for item in selected["selected"] if item["pool"] == "calibration"]
    cal.require(len(entries) == 7, "seven_selected_sources_required")
    registered = {
        (item["path"], item["sha256"])
        for item in inventory["samples"]
        if item["expected_pool"] == "calibration" and item["exposure"] == "current"
    }
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    datasets = []
    roots = {
        "text": Path(entries[0]["path"]).parent,
        "video": PRIVATE / "clevrer-validation-gold-001",
        "screenshot": PRIVATE / "gui-gold-002",
        "audio": PRIVATE / "crossmodal-source-bundle-001",
        "image": PRIVATE / "crossmodal-source-bundle-001",
        "crossmodal": PRIVATE / "crossmodal-source-bundle-001",
    }
    for index, item in enumerate(entries):
        cal.require((item["path"], item["sha256"]) in registered, "not_in_current_inventory")
        # Only stat is permitted on private datasets in this metadata preparer.
        cal.require(stat.S_ISREG(Path(item["path"]).stat().st_mode), "missing_private_artifact")
        declaration = stage.Admission(
            schema_version="private-stage-data-admission-v1",
            status="admitted",
            data_sha256=item["sha256"],
            selected_manifest_sha256=EXPECTED["selected"],
            evidence_bundle_sha256=admission["evidence_bundle_sha256"],
            pool="calibration",
            cohort=item["cohort"],
            expected_rows=item["expected_questions"],
        )
        declaration_path = args.output / f"admission-{index:02d}.json"
        cal.write_private(declaration_path, declaration.model_dump())
        spec = stage.Spec(
            mode="S5",
            cohort=item["cohort"],
            data=stage.Artifact(path=item["path"], sha256=item["sha256"]),
            pool="calibration",
            media_root=str(roots[item["cohort"]]),
            model_dir=str(model),
            base_revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
            model_files=model_files,
            adapter_dir=str(adapter),
            adapter_files=adapter_files,
            admission=stage.Artifact(**bind(declaration_path)),
            selected_manifest=stage.Artifact(**bindings["selected"]),
            data_admission=stage.Artifact(**bindings["admission"]),
            evidence_root=str(ROOT),
            expected_rows=item["expected_questions"],
            output_root=str(args.output / "private-unused"),
            timeout_seconds=args.timeout,
        )
        stage.verify_catalog(spec)  # metadata/evidence only; never stage.validate or load_samples
        datasets.append(spec)
    job = cal.JobSpec(
        datasets=datasets,
        producer_revision=args.producer_revision,
        runtime_python=stage.Artifact(**bind(Path(sys.executable))),
        inventory=stage.Artifact(**bindings["inventory"]),
        inventory_receipt=stage.Artifact(**bindings["inventory_receipt"]),
        root_traverse_acl=True,
    )
    job_path = args.output / "job.json"
    cal.write_private(job_path, job.model_dump())
    cal.JobSpec.model_validate_json(job_path.read_bytes())
    uid_available = all(broker.unused_uid(uid) for uid in (64201, 64202))
    broker.check_process_ids({64201, 64202})
    media_metadata = []
    for name in ("clevrer-validation-gold-001", "gui-gold-002", "crossmodal-source-bundle-001"):
        directory = PRIVATE / name
        paths = [p for p in (directory / "media").rglob("*") if p.is_file()]
        media_metadata.append(
            {
                "directory": str(directory),
                "directory_mode": oct(stat.S_IMODE(directory.stat().st_mode)),
                "asset_count": len(paths),
                "all_pool_asset_bytes": sum(p.stat().st_size for p in paths),
                "non_public_assets": sum(not p.stat().st_mode & 0o004 for p in paths),
            }
        )
    report = {
        "status": "metadata_prepared"
        if hasattr(cal, "snapshot_media")
        else "metadata_prepared_runtime_blocked",
        "exit_code": 0,
        "job": bind(job_path),
        "bindings": bindings,
        "adapter_sha256": EXPECTED["adapter"],
        "producer_revision": args.producer_revision,
        "producer_sha256": sha256_file(Path(cal.__file__)),
        "source_code_sha256": stage.code_hash(),
        "runtime": job.runtime_python.model_dump(),
        "dataset_sources": [
            {
                "cohort": s.cohort,
                "rows": s.expected_rows,
                "path": s.data.path,
                "sha256": s.data.sha256,
                "media_root": s.media_root,
            }
            for s in datasets
        ],
        "audio_separate_files": True,
        "selected_rows": sum(s.expected_rows for s in datasets),
        "private_contents_opened": False,
        "gpu_started": False,
        "uid64201_64202_unused": uid_available,
        "permission_metadata": media_metadata,
        "blockers": (
            []
            if hasattr(cal, "snapshot_media")
            else ["Private media snapshot fallback absent; root must deploy reviewed producer fix."]
        ),
        "admission_scope": (
            "Per-source metadata declarations derived from unchanged admitted selection; "
            "no new quality or isolation claim."
        ),
        "wall_seconds": time.monotonic() - start,
    }
    cal.write_private(args.output / "preparation-report.json", report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
