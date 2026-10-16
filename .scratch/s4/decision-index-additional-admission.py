"""Draft three evaluation admissions from metadata; never mutate source manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.decision_index_offline import Admission, PreparationManifest

REV = "f8552a13b643d39f61185f5951bb5fed16bedf51"
BASE = "b40157e74b88f411161d3905a75ff4077f9f5b6cf2ec1d1ba0ad887b04842ac1"
ARCHIVE = "12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082"
SOURCE = {
    "LICENSE": "3445fa3344098a37d88d2d0c24e39cf6c8a1d63a9478d52668db2acfda21f6ab",
    "decision_index/suite/build/home_appliance.py": (
        "89666f0c5af54cc7d35addcb1feb29cb6ed022dfa6de8f2419ee96325e3c6bc3"
    ),
    "decision_index/suite/build/normalize_direct.py": (
        "98f7fcc03edc25fa03a62d48fdc467748ee23cbbf8ba4442086bd0840df60431"
    ),
    "decision_index/suite/build/acquire.py": (
        "94bc9997f9c95ae33ed9d2d00e6c715bd1408b69ecb64de9e31e06317b761c16"
    ),
}
EXPECTED = {
    "banking77/categories.json": (
        4,
        "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
        "cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4",
        "banking_data/categories.json",
    ),
    "banking77/test.csv": (
        4,
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        "799687a8367359432985b8b13d85a2baf73f92dd",
        "banking_data/test.csv",
    ),
    "clinc150/data_full.json": (
        5,
        "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
        "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa",
        "data/data_full.json",
    ),
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(observed: dict, publishers: list[dict], source: Path, clinc_receipt: Path) -> dict:
    require(observed["manifest_sha256"] == BASE, "approved_base_identity")
    require(
        hashlib.sha256(
            (json.dumps(observed["manifest"], indent=2, ensure_ascii=False) + "\n").encode()
        ).hexdigest()
        == BASE,
        "base_manifest_bytes",
    )
    m = PreparationManifest.model_validate(observed["manifest"])
    require(m.source_archive.sha256 == ARCHIVE, "source_archive_identity")
    old = {a.catalog_id for a in m.admissions if a.evaluation_admitted}
    require(old == {10, 24, 26, 27, 29, 30, 31, 36, 43, 48, 50, 57, 62}, "prior_admissions_changed")
    for name, expected in SOURCE.items():
        require(digest(source / name) == expected, "fixed_source_identity:" + name)
    selected = [f for f in m.files if {4, 5, 9} & set(f.catalog_ids)]
    require(
        len(selected) == 3 and not any(9 in f.catalog_ids for f in selected), "recipe_input_closure"
    )
    bindings = {4: [], 5: [], 9: []}
    for suffix, (cid, sha, blob, upstream_path) in EXPECTED.items():
        f = next(f for f in selected if f.destination.endswith("/" + suffix))
        o = next(o for o in observed["observed_files"] if o["destination"] == f.destination)
        p = next(p for p in publishers if p.get("path") == upstream_path)
        require(f.catalog_ids == [cid] and f.sha256 == sha == o["sha256"], "data_sha_identity")
        require(
            str(f.path) == o["path"]
            and o["git_blob_sha1"] == blob == p["git_blob_sha1"]
            and o["bytes"] == p["bytes"],
            "publisher_blob_identity",
        )
        bindings[cid].append({**o, "publisher": p})
    receipts = json.loads(clinc_receipt.read_bytes())
    require(
        receipts["source_provenance"]["full_source"]
        == "clinc/clinc_oos@155b9c710419136e17307b80d0a13e68cd46b4ec:plus/train.parquet",
        "clinc_prior_scope",
    )
    require(receipts["test_accessed"] is False, "clinc_prior_test_boundary")
    # Metadata snapshots are immutable evidence, not instructions or dataset rows.
    required = {
        "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
        "7b9ce9069931cc2de89dce8c7ffb53d07806cf413601ccf133b9af5227b3a373",
        "e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76",
        "e28704a3c04f6b05c286ca2a8891bbf32c69c000dedfc5825d6e1713369b7a89",
    }
    require(required <= {p.get("sha256") for p in publishers}, "publisher_license_identity")
    admissions = []
    for cid, license_name in [(4, "CC-BY-4.0"), (5, "CC-BY-3.0"), (9, "MIT")]:
        evidence = {
            "source_revision": REV,
            "source_archive_sha256": ARCHIVE,
            "approved_base_sha256": BASE,
            "purpose": (
                "offline evaluation/rebuild only; no training or redistribution; preserve "
                "attribution, license and change notices; root review required"
            ),
            "inputs": bindings[cid],
            "source_files_sha256": SOURCE,
        }
        if cid in [4, 5]:
            repo = "PolyAI-LDN/task-specific-datasets" if cid == 4 else "clinc/oos-eval"
            evidence["publisher_metadata"] = [
                p
                for p in publishers
                if p.get("repo") == repo or (cid == 5 and "huggingface" in p.get("url", ""))
            ]
        if cid == 5:
            evidence["prior_train_receipt_sha256"] = digest(clinc_receipt)
            evidence["boundary"] = (
                "HF train approval is not test approval; publisher GitHub LICENSE plus "
                "exact blob equality independently covers Index test/oos_test input. HF "
                "fixed card links same repository; its LICENSE URL returns404 and is not "
                "used as evidence."
            )
        if cid == 9:
            evidence["boundary"] = (
                "Fixed generator imports only copy/hashlib/json/random; all content "
                "constructed from local literals; NEEDS[9]=[]; reads only its freshly "
                "written output to hash. Root MIT covers authored generator/content; not "
                "borrowed external datasets. Generator not executed."
            )
        admissions.append(
            Admission(
                catalog_id=cid,
                evaluation_admitted=True,
                license=license_name,
                evidence=json.dumps(evidence, sort_keys=True),
            ).model_dump()
        )
    PreparationManifest.model_validate(
        {
            **m.model_dump(mode="json"),
            "admissions": [a.model_dump() for a in m.admissions] + admissions,
        }
    )
    return {
        "status": "proposal_pending_root_review",
        "applied": False,
        "source_revision": REV,
        "base_manifest_sha256": BASE,
        "additional_closed_count": 3,
        "admissions": admissions,
        "total_if_approved": 16,
        "remaining_of44": 28,
        "publisher_metadata": publishers,
        "input_bindings": bindings,
        "clinc_prior_receipt_sha256": digest(clinc_receipt),
        "source_files_sha256": SOURCE,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--publishers", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--clinc-receipt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = build(
        json.loads(args.observed.read_bytes()),
        json.loads(args.publishers.read_bytes()),
        args.source,
        args.clinc_receipt,
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "additional_closed_count",
                    "total_if_approved",
                    "remaining_of44",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
