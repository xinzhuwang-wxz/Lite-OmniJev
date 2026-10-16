"""Hash-bound metadata-only evaluation proposal; never mutates a source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASE = "e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f"
METADATA_SHA = "453705d4e5c346e81a32eed91f17aceae25c4255803505df7245778f38de148e"
EXPECTED_COUNTS = {3: 319, 32: 3, 37: 2}
LICENSES = {3: "Apache-2.0", 32: "CC-BY-4.0", 37: "Apache-2.0"}
SCOPE = {
    3: "Fixed api-bank README explicitly publishes code and author-annotated evaluation data. "
    "The nearest subproject Apache Work grant is not code-only; no separate license "
    "exception appears in the fixed metadata tree. Applying it to the published sample "
    "Work is an explicit scope inference for root review, not an independent warranty "
    "of all third-party rights. Excludes training corpus, model weights and API execution.",
    32: "Fixed README identifies authored GPT-4 synthetic datasets and all three domain JSONs. "
    "Publisher TAUR-Lab/MuSR dataset card explicitly licenses the dataset CC-BY-4.0 and "
    "links the same Git source and three domains. Use the explicit data license rather "
    "than Git code MIT. Current card is corroborating publisher authorization, not a "
    "claim that later CSV exports are byte-identical to fixed JSON inputs.",
    37: "Fixed publisher README is specifically a dataset release, identifies examples and "
    "products parquet files, and licenses the project Apache-2.0. Retain Amazon NOTICE. "
    "This is the publisher's dataset-project grant, not a license borrowed from a "
    "separate implementation repository. No separate data exception in metadata tree.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base: Path, observed: Path, metadata: Path) -> dict:
    if sha(base) != BASE:
        raise ValueError("approved_base_identity")
    if sha(metadata) != METADATA_SHA:
        raise ValueError("publisher_metadata_identity")
    manifest = json.loads(base.read_bytes())
    publisher = json.loads(metadata.read_bytes())
    rows = json.loads(observed.read_bytes())
    selected = [f for f in manifest["files"] if set(f["catalog_ids"]) & set(EXPECTED_COUNTS)]
    lookup = {r["path"]: r for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(selected):
        raise ValueError("selected_input_closure")
    if len(manifest["admissions"]) != 16:
        raise ValueError("base_admissions_changed")
    for f in selected:
        if f["path"] not in lookup or any(
            lookup[f["path"]][k] != f[k] for k in ["sha256", "destination", "catalog_ids"]
        ):
            raise ValueError("input_SHA_binding")
    for cid, count in EXPECTED_COUNTS.items():
        if sum(r["catalog_ids"] == [cid] for r in rows) != count:
            raise ValueError("catalog_count")
        if not any(p["catalog_id"] == cid and p["path"].endswith("README.md") for p in publisher):
            raise ValueError("publisher_scope_missing")
    card = next(p for p in publisher if p["repo"] == "TAUR-Lab/MuSR")
    if card.get("license_frontmatter") != "cc-by-4.0":
        raise ValueError("explicit_MuSR_data_license")
    admissions = []
    for cid in EXPECTED_COUNTS:
        evidence = {
            "approved_base_sha256": BASE,
            "purpose": "local offline evaluation/rebuild only; no training or redistribution",
            "scope_basis": SCOPE[cid],
            "input_bindings": [r for r in rows if r["catalog_ids"] == [cid]],
            "publisher_metadata": [p for p in publisher if p["catalog_id"] == cid],
            "obligations": "retain original attribution/license/NOTICE and mark changes",
            "root_review_required": True,
        }
        admissions.append(
            {
                "catalog_id": cid,
                "evaluation_admitted": True,
                "license": LICENSES[cid],
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        )
    return {
        "status": "proposal_pending_root_review",
        "applied": False,
        "base_manifest_sha256": BASE,
        "admissions": admissions,
        "additional_closed_count": 3,
        "catalog_ids": sorted(EXPECTED_COUNTS),
        "metadata_sha256": sha(metadata),
        "observed_sha256": sha(observed),
        "source_rows_read": False,
        "gpu_used": False,
        "cost": None,
        "scope_inference_boundary": "Publisher Work-scope interpretation requires root review; "
        "no demand for a new bespoke legal confirmation imposed.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "observed", "metadata", "output"]:
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    result = build(a.base, a.observed, a.metadata)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "catalog_ids", "additional_closed_count"]}))


if __name__ == "__main__":
    main()
