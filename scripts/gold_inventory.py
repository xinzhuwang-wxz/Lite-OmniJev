"""Private, supplied-inventory audit; never a gold or security attestation.

Run as ``python -m scripts.gold_inventory --manifest ABS --report ABS``.
Only manifest-listed JSONL files are opened: media URIs are metadata, not reads.
The launcher must supply the actual private execution/log permission boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, Never, Self

from pydantic import Field, model_validator

from data.schema import Contract, Nonempty, Pool, Sample, Sha256
from eval.gold_gate import Cohort, IsolationRow, IsolationSummary, audit_pool_isolation
from scripts.s1_text_pools import canonical_family_hash

Failure = Literal[
    "invalid_manifest",
    "input_io",
    "hash_mismatch",
    "invalid_sample",
    "invalid_component",
    "pool_mismatch",
    "modality_mismatch",
    "missing_component",
    "conflicting_identity",
    "pool_overlap",
    "invalid_metadata",
    "report_io",
    "job_failed",
    "invalid_arguments",
]


class Artifact(Contract):
    path: Nonempty
    sha256: Sha256

    @model_validator(mode="after")
    def absolute(self) -> Self:
        if not Path(self.path).is_absolute():
            raise ValueError("absolute path required")
        return self


class SampleArtifact(Artifact):
    cohort: Cohort | None = None
    cohorts_by_data_version: dict[Nonempty, Cohort] | None = None
    exposure: Literal["current", "historical"]
    expected_pool: Pool

    @model_validator(mode="after")
    def cohort_mapping(self) -> Self:
        if (self.cohort is None) == (self.cohorts_by_data_version is None):
            raise ValueError("exactly one cohort declaration required")
        if self.cohorts_by_data_version == {}:
            raise ValueError("nonempty mapping required")
        return self


class Manifest(Contract):
    schema_version: Literal["gold-inventory-manifest-v1"]
    samples: list[SampleArtifact] = Field(min_length=1)
    component_ledgers: list[Artifact]


class Component(Contract):
    item_id: Nonempty
    family_id: Nonempty
    family_hash: Sha256
    source_hash: Sha256
    media_sha256: Sha256
    proposal: Nonempty
    proposal_option_text_sha256: Sha256 | None = None


class ComponentRecord(Contract):
    item_id: Nonempty
    crossmodal_family_id: Nonempty
    crossmodal_family_hash: Sha256
    pool: Pool
    proposal_rule: Nonempty
    program_truth: Literal["yes", "no"]
    image: Component
    audio: Component
    component_family_separate_ledger: Literal[True]


class Receipt(Contract):
    schema_version: Literal["gold-inventory-receipt-v1"] = "gold-inventory-receipt-v1"
    scope: Literal["supplied_inventory_only"] = "supplied_inventory_only"
    outcome: Literal["isolated", "rejected"]
    categories: list[Failure] = Field(default_factory=list)
    manifest_sha256: Sha256 | None = None
    verified_artifact_sha256: list[Sha256] = Field(default_factory=list)
    isolation: IsolationSummary | None = None
    inventory_completeness: Literal["not_validated"] = "not_validated"
    lineage_completeness: Literal["not_validated"] = "not_validated"
    gold_provenance: Literal["not_validated"] = "not_validated"
    private_job_permissions: Literal["not_validated"] = "not_validated"
    external_security_attestation: Literal["not_validated"] = "not_validated"


class Rejected(Exception):
    def __init__(self, category: Failure):
        self.category = category


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise Rejected("input_io") from None


def bound_lines(artifact: Artifact) -> list[str]:
    content = read_bytes(Path(artifact.path))
    if digest(content) != artifact.sha256:
        raise Rejected("hash_mismatch")
    try:
        lines = content.decode("utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError
        return lines
    except (UnicodeError, ValueError):
        raise Rejected("invalid_metadata") from None


def extract(sample: Sample, artifact: SampleArtifact) -> IsolationRow:
    cohort = artifact.cohort
    if cohort is None:
        cohort = (artifact.cohorts_by_data_version or {}).get(sample.meta.data_version)
    if cohort is None:
        raise Rejected("invalid_sample")
    if sample.meta.pool != artifact.expected_pool:
        raise Rejected("pool_mismatch")
    expected = {
        "text": set(),
        "image": {"image"},
        "screenshot": {"image"},
        "audio": {"audio"},
        "video": {"video"},
        "crossmodal": {"image", "audio"},
    }[cohort]
    if {media.type for media in sample.prefix.media} != expected or set(sample.meta.modality) - {
        "text"
    } != expected:
        raise Rejected("modality_mismatch")
    # Crossmodal references are populated only after checking the bound ledger.
    return IsolationRow.model_construct(
        item_sha256=digest(sample.id.encode()),
        cohort=cohort,
        pool=sample.meta.pool,
        exposure=artifact.exposure,
        original_family_sha256=[canonical_family_hash(sample.meta.family_id)],
        family_alias_sha256=[],
        media_sha256=[m.sha256 for m in sample.prefix.media],
        component_item_sha256=[],
    )


def link_components(
    samples: list[Sample], rows: list[IsolationRow], records: list[ComponentRecord]
) -> None:
    by_id: dict[str, list[tuple[Sample, IsolationRow]]] = {}
    for sample, row in zip(samples, rows, strict=True):
        by_id.setdefault(sample.id, []).append((sample, row))
    linked: dict[tuple[str, Pool], ComponentRecord] = {}
    for record in records:
        key = (record.item_id, record.pool)
        if key in linked and linked[key] != record:
            raise Rejected("conflicting_identity")
        linked[key] = record
        composites = [
            pair
            for pair in by_id.get(record.item_id, [])
            if pair[1].pool == record.pool and pair[1].cohort == "crossmodal"
        ]
        if not composites:
            raise Rejected("missing_component")
        for sample, row in composites:
            if (
                record.crossmodal_family_id != sample.meta.family_id
                or record.crossmodal_family_hash != canonical_family_hash(sample.meta.family_id)
            ):
                raise Rejected("invalid_component")
            refs: list[str] = []
            hashes: set[str] = set()
            for modality, component in (("image", record.image), ("audio", record.audio)):
                candidates = [
                    pair
                    for pair in by_id.get(component.item_id, [])
                    if pair[1].pool == record.pool and pair[1].cohort == modality
                ]
                if not candidates:
                    raise Rejected("missing_component")
                for original, original_row in candidates:
                    if (
                        component.family_id != original.meta.family_id
                        or component.family_hash != canonical_family_hash(original.meta.family_id)
                        or component.source_hash != original.meta.source_sha256
                        or set(original_row.media_sha256) != {component.media_sha256}
                    ):
                        raise Rejected("invalid_component")
                refs.extend(original_row.item_sha256 for _, original_row in candidates)
                hashes.add(component.media_sha256)
            if hashes != set(row.media_sha256):
                raise Rejected("invalid_component")
            row.component_item_sha256 = refs
    if any(row.cohort == "crossmodal" and not row.component_item_sha256 for row in rows):
        raise Rejected("missing_component")


def audit_manifest(path: Path) -> Receipt:
    """All exceptions stay private; only a schema-allowlisted receipt escapes."""
    manifest_hash: str | None = None
    verified: list[str] = []
    try:
        content = read_bytes(path)
        manifest_hash = digest(content)
        try:
            manifest = Manifest.model_validate_json(content)
        except ValueError:
            raise Rejected("invalid_manifest") from None
        samples: list[Sample] = []
        rows: list[IsolationRow] = []
        current_identities: dict[str, str] = {}
        for artifact in manifest.samples:
            lines = bound_lines(artifact)
            verified.append(artifact.sha256)
            for line in lines:
                try:
                    sample = Sample.model_validate_json(line)
                except ValueError:
                    raise Rejected("invalid_sample") from None
                # Pool/exposure changes remain visible to the isolation audit.
                payload = sample.model_dump(mode="json")
                del payload["meta"]["pool"]
                # Supervision/version changes do not erase exposure to the same input.
                # Target legality was validated above; gold correctness is not this audit.
                del payload["meta"]["data_version"]
                for question in payload["questions"]:
                    del question["target"]
                fingerprint = digest(json.dumps(payload, sort_keys=True).encode())
                if artifact.exposure == "current":
                    if sample.id in current_identities:
                        raise Rejected("conflicting_identity")
                    current_identities[sample.id] = fingerprint
                samples.append(sample)
                # Historical revisions remain separate observations, joined by a
                # stable original ID alias even if their declared family changed.
                # These observation counts are not independent gold-item quotas.
                row = extract(sample, artifact)
                rows.append(
                    row.model_copy(
                        update={
                            "item_sha256": digest((sample.id + ":" + fingerprint).encode()),
                            "family_alias_sha256": [
                                digest(("original-item:" + sample.id).encode())
                            ],
                        }
                    )
                )
        records: list[ComponentRecord] = []
        for ledger in manifest.component_ledgers:
            lines = bound_lines(ledger)
            verified.append(ledger.sha256)
            try:
                records.extend(ComponentRecord.model_validate_json(line) for line in lines)
            except ValueError:
                raise Rejected("invalid_component") from None
        link_components(samples, rows, records)
        summary = audit_pool_isolation(
            {
                "schema_version": "gold-isolation-metadata-v1",
                "rows": [row.model_dump() for row in rows],
            }
        )
        return Receipt(
            outcome=summary.outcome,
            categories=list(summary.categories),
            manifest_sha256=manifest_hash,
            verified_artifact_sha256=verified,
            isolation=summary,
        )
    except Rejected as error:
        category = error.category
    except Exception:
        category = "job_failed"
    return Receipt(
        outcome="rejected",
        categories=[category],
        manifest_sha256=manifest_hash,
        verified_artifact_sha256=verified,
    )


def publish(receipt: Receipt, path: Path) -> None:
    """Publish a complete receipt atomically without replacing any existing path."""
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(receipt.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class PrivateParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise Rejected("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = PrivateParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--report", type=Path, required=True)
        args = parser.parse_args(argv)
        if not args.manifest.is_absolute() or not args.report.is_absolute():
            raise Rejected("invalid_arguments")
        receipt = audit_manifest(args.manifest)
        try:
            publish(receipt, args.report)
        except Exception:
            raise Rejected("report_io") from None
    except Rejected as error:
        receipt = Receipt(outcome="rejected", categories=[error.category])
    except Exception:
        receipt = Receipt(outcome="rejected", categories=["job_failed"])
    print(receipt.model_dump_json())
    return 0 if receipt.outcome == "isolated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
