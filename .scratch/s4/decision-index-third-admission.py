"""Draft only BFCL/API-Bank/BBH license facts; never apply admissions or read rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = "e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f"
IDS = {1, 3, 58}


def build(base: Path, observed: Path, metadata: Path) -> dict:
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE:
        raise ValueError("approved base SHA changed")
    manifest = json.loads(base.read_bytes())
    observations = json.loads(observed.read_bytes())
    publisher = json.loads(metadata.read_bytes())
    files = [f for f in manifest["files"] if IDS & set(f["catalog_ids"])]
    by_path = {o["path"]: o for o in observations}
    if len(by_path) != len(files) or len(observations) != len(files):
        raise ValueError("input metadata closure changed")
    for f in files:
        o = by_path[f["path"]]
        if any(o[k] != f[k] for k in ("sha256", "destination", "catalog_ids")):
            raise ValueError("input SHA binding changed")
    for p in publisher:
        if "text" in p and hashlib.sha256(p["text"].encode()).hexdigest() != p["sha256"]:
            raise ValueError("publisher metadata SHA changed")
    tree = next(
        p for p in publisher if p["key"] == "publisher_tree" and p["repo"].startswith("gorilla")
    )
    entries = {e["path"]: e for e in tree["entries"]}
    matches, mismatches = [], []
    for o in observations:
        if 1 in o["catalog_ids"]:
            name = o["destination"].split("/data/")[-1]
            e = entries.get(name, {})
            (matches if e.get("oid") == o["git_blob_sha1"] else mismatches).append(name)
    required_scope = {
        "22234ad32a0bb921809278d01ed65e932b4e611690dfa17d1e0eb6ba74ab4e27",
        "2273d9fd5fc2fd175deb939b99ed52e2945b0225dd1f290a836308ca36efec18",
    }
    if not required_scope <= {p["sha256"] for p in publisher}:
        raise ValueError("fixed data README scope evidence missing")
    admissions = []
    for cid, license_name in [(1, "Apache-2.0"), (58, "MIT")]:
        evidence = {
            "base_manifest_sha256": BASE,
            "purpose": "offline evaluation only; never train or redistribute; preserve notices",
            "input_bindings": [o for o in observations if cid in o["catalog_ids"]],
            "scope_basis": "fixed data README license"
            if cid == 1
            else "fixed BBH MIT plus data README canary; retain parent notices",
            "publisher_metadata": publisher,
            "root_review_required": True,
        }
        admissions.append(
            {
                "catalog_id": cid,
                "evaluation_admitted": True,
                "license": license_name,
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        )
    return {
        "status": "proposal_pending_root_review",
        "applied": False,
        "base_manifest_sha256": BASE,
        "catalog_ids": sorted(IDS),
        "additional_closed_count": 2,
        "admissions": admissions,
        "scope": "metadata-only rights audit; no rows, execution, terms acceptance or training",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(observed.read_bytes()).hexdigest(),
        "publisher_metadata": publisher,
        "input_bindings": observations,
        "decisions": [
            {
                "catalog_id": 1,
                "name": "BFCL",
                "evaluation_admitted": None,
                "recommendation": "evaluation_only_pending_root_review",
                "publisher_dataset_license": "Apache-2.0",
                "fixed_input_count": 41,
                "matching_publisher_blobs": matches,
                "different_publisher_blobs": mismatches,
                "gap": (
                    "Fixed GitHub data/README.md itself declares Apache-2.0 for this dataset. "
                    "It describes authored functions and enterprise/OSS live contributions. "
                    "HF blob differences are version facts, not a licensing rejection. "
                    "No external API execution or provider terms acceptance is proposed."
                ),
            },
            {
                "catalog_id": 3,
                "name": "API-Bank",
                "evaluation_admitted": False,
                "publisher_dataset_license": "MIT",
                "fixed_subproject_license": "Apache-2.0",
                "fixed_input_count": 319,
                "gap": (
                    "Fixed README describes author-annotated evaluation dialogues and authored "
                    "APIs; subproject Apache LICENSE is not explicitly restricted to code, but "
                    "README has no explicit dataset-license statement. Root must decide whether "
                    "the subproject Work grant covers released samples or seek publisher scope "
                    "confirmation. Later HF MIT packaging is corroboration, not a required "
                    "bijection and not automatic retroactive authorization."
                ),
            },
            {
                "catalog_id": 58,
                "name": "BBH",
                "evaluation_admitted": None,
                "recommendation": "evaluation_only_pending_root_review",
                "fixed_repo_license": "MIT",
                "fixed_input_count": 27,
                "parent_repo_license": "Apache-2.0",
                "mandatory_boundaries": [
                    "evaluation only",
                    "never train",
                    "preserve canary",
                    "preserve original notices and citations",
                    "no redistribution",
                ],
                "gap": (
                    "Fixed BBH README identifies task materials and BIG-bench derivation; "
                    "fixed bbh/README.md expressly prohibits training. Propose local evaluation "
                    "under BBH MIT while retaining all parent/task notices and canaries. "
                    "Task-level notices were not read; absence of inspected extra terms is "
                    "neither a prohibition nor proof that notices can be removed."
                ),
            },
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = build(a.base, a.observed, a.metadata)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "additional_closed_count", "catalog_ids")}))


if __name__ == "__main__":
    main()
