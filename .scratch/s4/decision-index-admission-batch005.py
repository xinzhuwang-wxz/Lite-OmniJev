"""Draft six evaluation-only admissions from pinned publisher metadata, never apply."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from scripts.decision_index_offline import Admission, PreparationManifest

BASE = "6373c2f0f72f88c129b197c9f2d1cf4e4442274a815e8d3b5732b08ab232f2e8"
ARCHIVE = "12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082"
REV = "f8552a13b643d39f61185f5951bb5fed16bedf51"
METADATA = "946a18dbbcbdb61794a0280944af375291955a732f458963e60961e0530a6b4c"
PRIOR = {10, 24, 26, 27, 29, 30, 31, 36, 43, 48, 50, 57, 62, 4, 5, 9, 1, 58, 3, 32, 37}
REMAINING = {
    2,
    6,
    11,
    12,
    20,
    21,
    22,
    23,
    25,
    28,
    33,
    34,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    56,
    59,
    61,
    64,
}
COUNTS = {20: 3, 25: 1, 28: 1, 34: 1, 40: 5, 44: 1}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def observe(path: Path, destination: str, catalog_ids: list[int]) -> dict:
    # Read bytes solely for hashes. Never parse benchmark records or archives.
    b = path.read_bytes()
    return {
        "path": str(path),
        "destination": destination,
        "catalog_ids": catalog_ids,
        "bytes": len(b),
        "sha256": sha(b),
        "md5": hashlib.md5(b).hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest(),
    }


def build(manifest_bytes: bytes, observed: list[dict], metadata: dict) -> dict:
    require(
        sha((json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode()) == METADATA,
        "frozen_metadata_identity",
    )
    require(sha(manifest_bytes) == BASE, "approved_manifest_bytes")
    m = PreparationManifest.model_validate_json(manifest_bytes)
    require(m.source_archive.sha256 == ARCHIVE, "source_archive_identity")
    require(
        {a.catalog_id for a in m.admissions if a.evaluation_admitted} == PRIOR,
        "approved_admissions_identity",
    )
    require({r["catalog_id"] for r in metadata["candidates"]} == set(COUNTS), "candidate_set")
    additions = []
    total_bytes = 0
    for row in metadata["candidates"]:
        cid = row["catalog_id"]
        files = [f for f in m.files if cid in f.catalog_ids]
        require(len(files) == COUNTS[cid] == len(row["inputs"]), "recipe_file_closure")
        for f in files:
            expected = next((x for x in row["inputs"] if x["destination"] == f.destination), None)
            actual = [x for x in observed if x["destination"] == f.destination]
            require(expected is not None and len(actual) == 1, "missing_duplicate_observation")
            actual = actual[0]
            require(
                str(f.path) == expected["path"] == actual["path"]
                and f.catalog_ids == expected["catalog_ids"] == actual["catalog_ids"]
                and f.sha256 == expected["sha256"] == actual["sha256"],
                "input_binding",
            )
            for k in ["bytes", "md5", "git_blob_sha1"]:
                require(actual[k] == expected[k], "actual_file_identity:" + k)
            if cid in [25, 34, 40, 44]:
                blob = expected["publisher_blob"]
                require(
                    blob["sha"] == actual["git_blob_sha1"] and blob["size"] == actual["bytes"],
                    "publisher_blob_binding",
                )
            if cid == 20:
                part = f.destination.rsplit("_p", 1)[1].split(".")[0]
                published = next(z for z in row["release"]["files"] if f"_p{part}_" in z["key"])
                require(
                    published["checksum"] == "md5:" + actual["md5"]
                    and published["size"] == actual["bytes"],
                    "zenodo_file_binding",
                )
            if cid == 28:
                published = row["hf_release"]["file"]["lfs"]
                require(
                    published["sha256"] == actual["sha256"]
                    and published["size"] == actual["bytes"],
                    "hf_lfs_binding",
                )
            total_bytes += actual["bytes"]
        if cid == 25:
            card = row["dataset_card"]["evidence"]
            require(
                card["id"] == "Idavidrein/gpqa"
                and card["sha"] == "83022cefff930aea54f654c0b282e74b9eeda5c6"
                and card["cardData"]["license"] == "cc-by-4.0",
                "dataset_license_binding",
            )
            require(
                sha(json.dumps(card, sort_keys=True, separators=(",", ":")).encode())
                == row["dataset_card_projection_sha256"],
                "dataset_card_projection",
            )
        evidence = {
            "source_revision": REV,
            "source_archive_sha256": ARCHIVE,
            "approved_base_manifest_sha256": BASE,
            "metadata_sha256": METADATA,
            "purpose": "offline evaluation/rebuild only; no training or redistribution",
            "boundary": (
                "preserve attribution, copyright/permission notices, change notices "
                "and dataset-specific restrictions; root review required"
            ),
            "publisher_release_evidence": row,
        }
        additions.append(
            Admission(
                catalog_id=cid,
                evaluation_admitted=True,
                license=row["license"],
                evidence=json.dumps(evidence, sort_keys=True),
            ).model_dump()
        )
    require(len(observed) == sum(COUNTS.values()), "unexpected_observed_input")
    PreparationManifest.model_validate(
        {
            **m.model_dump(mode="json"),
            "admissions": [a.model_dump() for a in m.admissions] + additions,
        }
    )
    return {
        "schema": "decision-index-admission-proposal-v1",
        "status": "proposal_pending_root_review",
        "applied": False,
        "ready": False,
        "cost": None,
        "base_manifest_sha256": BASE,
        "source_revision": REV,
        "source_archive_sha256": ARCHIVE,
        "metadata_sha256": METADATA,
        "previous_admitted_count": len(PRIOR),
        "additional_closed_count": len(additions),
        "proposed_total_count": len(PRIOR) + len(additions),
        "required_count": 44,
        "remaining_catalog_ids": sorted(REMAINING - set(COUNTS)),
        "input_files_verified": sum(COUNTS.values()),
        "input_bytes_verified": total_bytes,
        "admissions": additions,
        "specific_pending": metadata["pending"],
    }


def main() -> int:
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).with_name("decision-index-admission-batch005-metadata.json"),
    )
    p.add_argument(
        "--observed",
        type=Path,
        help="previous metadata-only hash receipt; omit for live byte hashing",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "output_exists")
    raw = a.metadata.read_bytes()
    require(sha(raw) == METADATA, "frozen_metadata_identity")
    mb = a.manifest.read_bytes()
    m = PreparationManifest.model_validate_json(mb)
    if a.observed:
        observed = json.loads(a.observed.read_bytes())["observed_files"]
    else:
        observed = [
            observe(f.path, f.destination, f.catalog_ids)
            for f in m.files
            if set(COUNTS) & set(f.catalog_ids)
        ]
    result = build(mb, observed, json.loads(raw))
    result["execution"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "observed_receipt_sha256": sha(a.observed.read_bytes()) if a.observed else None,
        "observation_mode": "bound metadata receipt"
        if a.observed
        else "live file bytes hashed, not parsed",
        "duration_seconds": time.monotonic() - started,
        "exit_code": 0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as h:
        h.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ["admissions", "specific_pending"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
