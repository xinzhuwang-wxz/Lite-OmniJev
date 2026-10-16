"""Apply root-reviewed batch005 to a new directory; --validate-only writes nothing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

from scripts.decision_index_offline import CATALOG_IDS, PreparationManifest

BASE_SHA = "6373c2f0f72f88c129b197c9f2d1cf4e4442274a815e8d3b5732b08ab232f2e8"
PROPOSAL_SHA = "2f3f8c69bb70377d3979f5e4fc76208c4f12b3dbaefd20acf1d38f473ec374f5"
METADATA_SHA = "946a18dbbcbdb61794a0280944af375291955a732f458963e60961e0530a6b4c"
VALIDATOR_SHA = "90287d7c1c1da12d049f892df62b81fa8d461886d344a3367708218c0ff0eeaa"
ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
RAW = Path("/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3")
# Paths were verified by read-only SSH on 2026-10-16; no guessed HTTP-body cache.
DOCS = {
    "repos/isarcasm/LICENSE": "f956625f01a15c08af3be7d32480d39d8302fc36f52075c484a004e8e849835f",
    "repos/isarcasm/README.md": "98c79b76a4b5dc0e3cc2fb4e9653048e6c9a7156d6e2b1eacacf296b7acebb30",
    "raw/simplebench/LICENSE": "6d346288cf5db3f2b4bcd468894f4679d4c798d06e95ed865338b2e6bbfc6aa6",
    "raw/simplebench/README.md": "866c1a9692c04713953e5853c9e898f2a60fbaab21e11485463815db954d06a1",
    "raw/cladder/LICENSE": "5729188929e6873096ebe9623ae6ecb2164cb5d0acec10805cb4a463ff8fd05a",
    "raw/cladder/README.md": "ded750c842b056aecd45887551218e5af19a30d95f12ee02aff654f9c8e0843a",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    require(
        path.is_absolute() and path.is_file() and not path.is_symlink(), "regular_absolute_input"
    )
    with path.open("rb") as h:
        return hashlib.file_digest(h, "sha256").hexdigest()


def reviewed_manifest(base: dict, proposal: dict, verified: dict) -> dict:
    """Pure reviewed-extension contract, also used for bounded CPU validation."""
    require(proposal["base_manifest_sha256"] == BASE_SHA, "proposal_base_identity")
    require(
        proposal["status"] == "proposal_pending_root_review"
        and proposal["applied"] is False
        and proposal["ready"] is False,
        "proposal_state",
    )
    require(proposal["admissions"] == verified["admissions"], "live_verification_matches_proposal")
    prior = PreparationManifest.model_validate(base)
    require(
        len(prior.admissions) == 21
        and len({a.catalog_id for a in prior.admissions}) == 21
        and all(a.evaluation_admitted for a in prior.admissions),
        "prior21",
    )
    additions = []
    require(
        {a["catalog_id"] for a in proposal["admissions"]} == {20, 25, 28, 34, 40, 44}
        and len(proposal["admissions"]) == 6,
        "exact_six",
    )
    for item in proposal["admissions"]:
        require(item["evaluation_admitted"] is True, "evaluation_only_addition")
        addition = dict(item)
        evidence = json.loads(addition["evidence"])
        require(
            evidence["approved_base_manifest_sha256"] == BASE_SHA
            and evidence["metadata_sha256"] == METADATA_SHA,
            "evidence_identity",
        )
        evidence["root_review_required"] = False
        evidence["review"] = "root_reviewed_publisher_work_inferences_and_unspecified_ccby_version"
        evidence["reviewed_proposal_sha256"] = PROPOSAL_SHA
        evidence["purpose"] = (
            "Offline local evaluation/rebuild only; no training or redistribution. "
            "Preserve attribution, copyright/permission notices, change notices, "
            "GPQA canary and prohibition on online plaintext/image examples. "
            "No API execution, model generation or terms acceptance."
        )
        addition["evidence"] = json.dumps(evidence, sort_keys=True)
        additions.append(addition)
    result = {**base, "admissions": base["admissions"] + additions}
    new = PreparationManifest.model_validate(result)
    require(
        new.files == prior.files
        and new.source_archive == prior.source_archive
        and new.admissions[:21] == prior.admissions,
        "original_fields_unchanged",
    )
    require(len({a.catalog_id for a in new.admissions}) == 27, "unique27")
    return result


def main() -> int:
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=ROOT / "artifacts/s4-index-evaluation-admission-004/manifest.json",
    )
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="reviewed derived metadata snapshot, not raw HTTP body cache",
    )
    p.add_argument(
        "--validator", type=Path, required=True, help="pinned batch005 proposal validator"
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/s4-index-evaluation-admission-005"
    )
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    require(
        a.output_dir.is_absolute() and not a.output_dir.exists(), "new_absolute_output_required"
    )
    require(
        sha(a.base) == BASE_SHA
        and sha(a.proposal) == PROPOSAL_SHA
        and sha(a.metadata) == METADATA_SHA
        and sha(a.validator) == VALIDATOR_SHA,
        "reviewed_input_sha",
    )
    original = a.base.read_bytes()
    base = json.loads(original)
    proposal = json.loads(a.proposal.read_bytes())
    metadata = json.loads(a.metadata.read_bytes())
    spec = importlib.util.spec_from_file_location("reviewed_batch005", a.validator)
    require(spec is not None and spec.loader is not None, "validator_import")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    manifest = PreparationManifest.model_validate(base)
    require(
        sha(manifest.source_archive.path) == manifest.source_archive.sha256, "source_tar_identity"
    )
    actual = []
    for f in manifest.files:
        if {20, 25, 28, 34, 40, 44} & set(f.catalog_ids):
            require(sha(f.path) == f.sha256, "actual_input_sha")
            actual.append(validator.observe(f.path, f.destination, f.catalog_ids))
    verified = validator.build(original, actual, metadata)
    checks = {str(RAW / k): sha(RAW / k) for k in DOCS}
    require(
        all(checks[str(RAW / k)] == expected for k, expected in DOCS.items()),
        "publisher_local_docs_sha",
    )
    new = reviewed_manifest(base, proposal, verified)
    ids = {x["catalog_id"] for x in new["admissions"]}
    report = {
        "status": "validated_not_applied"
        if a.validate_only
        else "partial_evaluation_admission_applied",
        "exit_code": 0,
        "ready": False,
        "training_admitted": False,
        "redistribution_admitted": False,
        "base_manifest_sha256": BASE_SHA,
        "proposal_sha256": PROPOSAL_SHA,
        "metadata_sha256": METADATA_SHA,
        "validator_sha256": VALIDATOR_SHA,
        "runner_sha256": sha(Path(__file__).resolve()),
        "admitted_catalog_ids": sorted(ids),
        "remaining_catalog_ids": sorted(CATALOG_IDS - ids),
        "verified_inputs_by_catalog": validator.COUNTS,
        "publisher_local_document_checks": checks,
        "publisher_snapshot_boundary": (
            "Approved derived metadata snapshot verified; raw API "
            "HTTP bodies are not cached/revalidated. No network fetch."
        ),
        "original_manifests_unchanged": True,
        "gpu_used": False,
        "cost": None,
        "python": platform.python_version(),
        "duration_seconds": time.monotonic() - started,
    }
    require(a.base.read_bytes() == original and sha(a.proposal) == PROPOSAL_SHA, "inputs_unchanged")
    if not a.validate_only:
        os.umask(0o077)
        a.output_dir.mkdir(exist_ok=False)
        with (a.output_dir / "manifest.json").open("x") as h:
            h.write(json.dumps(new, indent=2) + "\n")
        report["output_manifest_sha256"] = sha(a.output_dir / "manifest.json")
        with (a.output_dir / "report.json").open("x") as h:
            h.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
