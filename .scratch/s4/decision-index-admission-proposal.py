"""Draft only the verified subset of16 evaluation candidates; never apply admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.decision_index_offline import (
    CATALOG_IDS,
    Admission,
    PreparationManifest,
    relative_path,
)

CANDIDATES = {10, 11, 12, 24, 26, 27, 29, 30, 31, 36, 43, 48, 50, 57, 62, 64}
LICENSES = {
    10: "CC-BY-SA-4.0",
    24: "MIT",
    26: "CC-BY-SA-4.0",
    27: "CC-BY-SA-4.0",
    29: "MIT",
    30: "MIT",
    31: "CC0-1.0 and CC-BY-4.0 (dataset portions)",
    36: "CC-BY-4.0",
    43: "MIT",
    48: "CC-BY-SA-4.0",
    50: "CC-BY-4.0",
    57: "MIT",
    62: "CC-BY-4.0",
}
PENDING = {
    11: (
        "Pinned data ZIP contains independent TERMS stating access/use agreement; "
        "no new acceptance made. Root must review existing terms authority."
    ),
    12: (
        "CC-BY-NC-4.0; current project purpose is not explicitly established as "
        "non-commercial. No inference from evaluation-only."
    ),
    64: (
        "Publisher explicitly licenses contributed annotations; underlying New "
        "Yorker caption text rights remain unclosed."
    ),
}
ROOTS = {
    10: "sgd",
    31: "searchless_chess",
    43: "cruxeval",
    48: "forecastbench-datasets",
    50: "habermas_machine",
}
SCOPE = (
    "offline local evaluation/rebuild only; no training, commercial release "
    "or dataset redistribution; preserve source attribution/license/changes "
    "notices; proposal pending root review"
)


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build(evidence_path: Path, observed_path: Path) -> tuple[list[dict], dict]:
    evidence = json.loads(evidence_path.read_bytes())
    observed = json.loads(observed_path.read_bytes())
    require(
        evidence["source_revision"] == "f8552a13b643d39f61185f5951bb5fed16bedf51", "source_revision"
    )
    require(
        observed["manifest_sha256"] == evidence["remote_manifest_sha256"]
        and observed["report_sha256"] == evidence["remote_report_sha256"],
        "input_receipt_identity",
    )
    manifest = PreparationManifest.model_validate(observed["manifest"])
    require(not manifest.admissions and len(manifest.files) == 1530, "baseline_changed")
    require(manifest.source_archive.sha256 == evidence["source_archive_sha256"], "archive_identity")
    catalog = {c["catalog_id"]: c for c in evidence["catalog"]}
    require(
        {
            c["catalog_id"]
            for c in evidence["catalog"]
            if c["evidence_category"] == "publisher_dataset_statement"
        }
        == CANDIDATES,
        "candidate_scope",
    )
    git = {r["name"]: r for r in observed["git"]}
    proposed, details = [], []
    for cid, license_name in sorted(LICENSES.items()):
        c = catalog[cid]
        files = sorted(
            [
                {"source": str(f.path), "sha256": f.sha256, "destination": f.destination}
                for f in manifest.files
                if cid in f.catalog_ids
            ],
            key=lambda f: f["destination"],
        )
        require(
            files and canonical_hash(files) == c["input_binding_sha256"], "catalog_file_identity"
        )
        for file in files:
            relative_path(file["destination"])
        applicable = files
        if cid in [26, 27]:
            marker = "ARC-Easy/" if cid == 26 else "ARC-Challenge/"
            applicable = [f for f in files if marker in f["destination"]]
            require(len(applicable) == 1, "arc_recipe_file")
        publishers = []
        if cid in ROOTS:
            root = git[ROOTS[cid]]
            pin = next(
                a["revision_or_content_sha256"] for a in c["acquisition"] if a["kind"] == "git"
            )
            require(root["head"] == pin, "publisher_git_pin")
            doc = next(e for e in root["evidence"] if Path(e["path"]).name == "README.md")
            if cid in [10, 43, 48]:
                require(
                    any(e["sha256"] == doc["sha256"] for e in c["local_evidence"]),
                    "publisher_document_identity",
                )
            else:
                expected = {
                    31: "df0800f57ee673873ddc7a3f6681a4e583c4d42fdc68b33b6666eccbcfed1e37",
                    50: "b967e89dc753aec404e80bc9291dfd37ed391f0b5361ec50b47aa5df4a80d4c2",
                }[cid]
                require(doc["sha256"] == expected, "publisher_document_identity")
            repo = next(
                a["url_or_repo"] for a in c["acquisition"] if a["kind"] == "git"
            ).removesuffix(".git")
            publishers.append(
                {
                    **doc,
                    "url": repo + "/blob/" + pin + "/README.md",
                    "revision": pin,
                    "line_range": {
                        10: [242, 247],
                        31: [286, 291],
                        43: [89, 90],
                        48: [18, 20],
                        50: [76, 78],
                    }[cid],
                }
            )
            if cid in [10, 43, 48]:
                for license_doc in root["evidence"]:
                    if Path(license_doc["path"]).name.startswith("LICENSE"):
                        publishers.append(
                            {
                                **license_doc,
                                "url": repo + "/blob/" + pin + "/" + Path(license_doc["path"]).name,
                                "revision": pin,
                            }
                        )
        else:
            urls = set(c["publisher_metadata_urls"])
            for doc in reversed(evidence["publisher_metadata"]):
                if (
                    doc.get("url") in urls
                    and "sha256" in doc
                    and not any(d["sha256"] == doc["sha256"] for d in publishers)
                ):
                    publishers.append(
                        {k: doc[k] for k in ["url", "sha256", "license_lines"] if k in doc}
                    )
            require(bool(publishers), "publisher_metadata_missing")
        acquisition = c["acquisition"]
        if cid in [26, 27]:
            acquisition = [
                {
                    **a,
                    "recipe_revision": None,
                    "resolved_acquisition_revision": "210d026faf9955653af8916fad021475a3f00453",
                }
                for a in acquisition
            ]
        item = {
            "catalog_id": cid,
            "name": c["name"],
            "license": license_name,
            "acquisition": acquisition,
            "publisher_evidence": publishers,
            "applicable_files": applicable,
            "input_catalog_binding_sha256": c["input_binding_sha256"],
            "applicable_files_sha256": canonical_hash(applicable),
            "scope": SCOPE,
            "obligations": [
                "retain attribution/source/license and identify mechanical adaptation",
                "no training or dataset/row redistribution authorized by this proposal",
            ],
            "note": {
                31: (
                    "Dataset CC0/CC-BY split is retained; Apache code license is not used for data."
                ),
                43: (
                    "Publisher names CRUXEval MIT; includes frozen released model-generated "
                    "distractors in the pinned repository. No CodeLlama model is downloaded "
                    "or executed."
                ),
                48: (
                    "Retired from edition0.3 scoring; admission draft still covers fixed "
                    "rebuild input closure."
                ),
                50: "All other materials CC-BY; Apache applies only to software.",
            }.get(
                cid,
                (
                    "Dataset card or explicit dataset README statement, "
                    "not inferred from code license."
                ),
            ),
        }
        evidence_string = json.dumps(
            {
                "proposal": "root_review_required",
                "source_revision": evidence["source_revision"],
                "source_archive_sha256": evidence["source_archive_sha256"],
                "input_manifest_sha256": observed["manifest_sha256"],
                "applicable_files_sha256": item["applicable_files_sha256"],
                "publisher_evidence": publishers,
                "scope": SCOPE,
            },
            sort_keys=True,
        )
        proposed.append(
            Admission(
                catalog_id=cid,
                evaluation_admitted=True,
                license=license_name,
                evidence=evidence_string,
            ).model_dump()
        )
        details.append(item)
    draft = PreparationManifest.model_validate({**observed["manifest"], "admissions": proposed})
    remaining = sorted(
        CATALOG_IDS - {a.catalog_id for a in draft.admissions if a.evaluation_admitted}
    )
    require(len(remaining) == 31 and not (set(LICENSES) & set(PENDING)), "full44_gate_preserved")
    report = {
        "status": "proposal_pending_root_review_not_applied",
        "evaluation_only": True,
        "training_admitted": False,
        "redistribution_admitted": False,
        "ready": False,
        "source_revision": evidence["source_revision"],
        "source_archive_sha256": evidence["source_archive_sha256"],
        "original_manifest_sha256": observed["manifest_sha256"],
        "original_report_sha256": observed["report_sha256"],
        "prior_evidence_file": {"path": str(evidence_path), "sha256": sha(evidence_path)},
        "observed_metadata_sha256": sha(observed_path),
        "candidate_scope": sorted(CANDIDATES),
        "evidence_closed_proposed_count": len(proposed),
        "evidence_closed_proposed_ids": sorted(LICENSES),
        "pending_within16": {str(k): v for k, v in PENDING.items()},
        "pending_evidence": {
            "11": {
                "source_archive": catalog[11]["file_examples"][0],
                "member": "contract-nli/TERMS",
                "sha256": "b86e5b9bd7d2c49fd0d60e1d0842b7f64746abc8b206451acbe9080027cc49c8",
                "fact": (
                    "TERMS says accessing/downloading/using is agreement; "
                    "root authority not inferred."
                ),
            },
            "12": {
                "url": catalog[12]["publisher_metadata_urls"][0],
                "license": "CC-BY-NC-4.0",
                "fact": "Non-commercial purpose not established by evaluation-only.",
            },
            "64": {
                "url": catalog[64]["publisher_metadata_urls"][0],
                "fact": (
                    "License explicitly covers contributed annotations; underlying captions "
                    "remain pending."
                ),
            },
        },
        "remaining_if_root_accepts_proposal": remaining,
        "current_applied_admission_count": 0,
        "full44_condition_unchanged": True,
        "details": details,
        "policy_boundary": SCOPE,
        "script_sha256": sha(Path(__file__)),
    }
    return proposed, report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--observed-metadata", type=Path, required=True)
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    require(not a.proposal.exists() and not a.report.exists(), "output_exists")
    proposal, report = build(a.evidence, a.observed_metadata)
    a.proposal.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n")
    report["proposal_sha256"] = sha(a.proposal)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "proposed": len(proposal),
                "pending_within16": len(PENDING),
                "remaining_full44": len(report["remaining_if_root_accepts_proposal"]),
                "applied": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
