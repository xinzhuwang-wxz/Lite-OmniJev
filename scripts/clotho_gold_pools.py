"""Private Clotho gold expansion. Only aggregate receipts may leave this CPU job.

--manifest binds train/val/test, metadata, license_plan, archive, inventory,
quarantine and explicit current_reserves (all {path, sha256}); audio_root is an
absolute directory. Output inventory retains every input observation, demoting
only explicitly bound replaced reserves to historical. Canary is never real gold.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from data.schema import Contract, Sample
from scripts.clevrer_gold_pools import (
    BuildError,
    Quarantine,
    bound_json,
    encode,
    file_digest,
    rank,
    verify,
    write_private,
    zip_members,
)
from scripts.gold_inventory import (
    Artifact,
    ComponentRecord,
    Manifest,
    PrivateParser,
    Rejected,
    SampleArtifact,
    audit_manifest,
    bound_lines,
    digest,
    extract,
)
from scripts.s0_audio_dev import (
    FIXED_ARCHIVE_ALIASES,
    METADATA_SHA256,
    TRAIN_SHA256,
    candidate_rows,
    decode_wav_full,
    load_license_plan,
    load_metadata,
    load_train_groups,
    sample_for,
    source_hash_for,
)
from scripts.s1_text_pools import canonical_family_hash

PINS = {
    "train": TRAIN_SHA256,
    "val": "c69b32095d70740cee49d66dc33690d0267cf7bc821d71d834725f9d24c0a491",
    "test": "6b72ea4722e2a4dde53465f9a0c165c901b16a2fb4ef16356d37891f71496737",
    "metadata": METADATA_SHA256,
    "license_plan": "e5821f612665a846309c884784007adf19c48fc56cfa7b87a15fcfa72e28e1f1",
    "archive": "4793d97781689f8bdb1a6669056386825bb0ec8022f2217acc8083abc4774d66",
}


class BuildManifest(Contract):
    schema_version: Literal["clotho-gold-build-v1"]
    train: Artifact
    val: Artifact
    test: Artifact
    metadata: Artifact
    license_plan: Artifact
    archive: Artifact
    inventory: Artifact
    quarantine: Artifact
    current_reserves: list[Artifact] = Field(min_length=1)
    audio_root: str


class Receipt(Contract):
    schema_version: Literal["clotho-gold-receipt-v1"] = "clotho-gold-receipt-v1"
    outcome: Literal["built", "rejected"] = "rejected"
    mode: Literal["real", "canary"] = "real"
    real_source_verified: bool = False
    full_gold_gate: Literal["not_validated"] = "not_validated"
    private_job_permissions: Literal["not_validated"] = "not_validated"
    inventory_completeness: Literal["not_validated"] = "not_validated"
    source_revision: str = "zenodo-6473207"
    seed: Literal[42] = 42
    category: str = "job_failed"
    counts: dict[str, int] = Field(default_factory=dict)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    output_sha256: dict[str, str] = Field(default_factory=dict)
    code_sha256: str = ""
    code_commit: str = "unavailable"
    runtime: str = ""
    elapsed_seconds: float = 0
    cost_usd: None = None
    exit_code: int = 1


def subset_sum(sizes: dict[str, int], target: int) -> set[str]:
    """First reachable exact sum in seed-42 order; each component is indivisible."""
    if target < 0:
        raise BuildError("invalid_quota")
    reachable: dict[int, tuple[int, str] | None] = {0: None}
    for family in sorted(sizes, key=lambda value: (rank("holdout", value), value)):
        for total in sorted(list(reachable), reverse=True):
            new = total + sizes[family]
            if new <= target and new not in reachable:
                reachable[new] = (total, family)
        if target in reachable:
            break
    if target not in reachable:
        raise BuildError("insufficient_capacity")
    selected = set()
    while target:
        previous = reachable[target]
        assert previous is not None
        target, family = previous
        selected.add(family)
    return selected


def inventory_state(
    manifest: BuildManifest, receipt: Receipt
) -> tuple[Manifest, dict[str, set[str]], dict[str, set[str]], dict[tuple[str, str], Sample]]:
    inventory = Manifest.model_validate(bound_json(manifest.inventory))
    replaced = {(a.path, a.sha256) for a in manifest.current_reserves}
    if len(replaced) != len(manifest.current_reserves):
        raise BuildError("invalid_reserve_binding")
    found = set()
    families: dict[str, set[str]] = defaultdict(set)
    media: dict[str, set[str]] = defaultdict(set)
    existing: dict[tuple[str, str], Sample] = {}
    ids: set[str] = set()
    for index, artifact in enumerate(inventory.samples):
        replace = (artifact.path, artifact.sha256) in replaced
        if replace:
            if artifact.exposure != "current" or artifact.expected_pool not in {
                "calibration",
                "holdout",
            }:
                raise BuildError("invalid_reserve_binding")
            found.add((artifact.path, artifact.sha256))
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            row = extract(sample, artifact)
            families[canonical_family_hash(sample.meta.family_id)].add(sample.meta.pool)
            for item in sample.prefix.media:
                media[item.sha256].add(sample.meta.pool)
            current_clotho = (
                artifact.exposure == "current"
                and row.cohort == "audio"
                and sample.meta.source_dataset == "Clotho-AQA"
                and sample.meta.pool in {"calibration", "holdout"}
            )
            if current_clotho != replace:
                raise BuildError("invalid_reserve_binding")
            if replace:
                if len(sample.questions) != 1 or len(sample.prefix.media) != 1:
                    raise BuildError("invalid_existing_gold")
                key = (sample.meta.family_id, sample.questions[0].instruction)
                if key in existing or sample.id in ids:
                    raise BuildError("duplicate_existing_identity")
                ids.add(sample.id)
                existing[key] = sample
        receipt.input_sha256[f"inventory_sample_{index}"] = artifact.sha256
    if found != replaced:
        raise BuildError("invalid_reserve_binding")
    for index, component_artifact in enumerate(inventory.component_ledgers):
        for line in bound_lines(component_artifact):
            record = ComponentRecord.model_validate_json(line)
            for component in (record.image, record.audio):
                if canonical_family_hash(component.family_id) != component.family_hash:
                    raise BuildError("invalid_component")
                families[component.family_hash].add(record.pool)
                media[component.media_sha256].add(record.pool)
            if canonical_family_hash(record.crossmodal_family_id) != record.crossmodal_family_hash:
                raise BuildError("invalid_component")
            families[record.crossmodal_family_hash].add(record.pool)
        receipt.input_sha256[f"inventory_components_{index}"] = component_artifact.sha256
    if any(len(pools) > 1 for pools in [*families.values(), *media.values()]):
        raise BuildError("historical_pool_conflict")
    return inventory, families, media, existing


def collect(manifest: BuildManifest) -> dict[tuple[str, str], dict[str, Any]]:
    approved = load_license_plan(
        Path(manifest.license_plan.path), metadata_sha256=manifest.metadata.sha256
    )
    metadata = load_metadata(
        Path(manifest.metadata.path), metadata_sha256=manifest.metadata.sha256, approved=approved
    )
    result: dict[tuple[str, str], dict[str, Any]] = {}
    media_families: dict[str, str] = {}
    checked: set[str] = set()
    with zipfile.ZipFile(manifest.archive.path) as archive:
        members = {Path(name).name: info for name, info in zip_members(archive).items()}
        for split in ("train", "val", "test"):
            artifact = getattr(manifest, split)
            groups = load_train_groups(Path(artifact.path), train_sha256=artifact.sha256)
            rows, _, _ = candidate_rows(
                groups=groups,
                metadata=metadata,
                audio_root=Path(manifest.audio_root),
                aliases=FIXED_ARCHIVE_ALIASES,
            )
            for row in rows:
                family = str(row["sound_id"])
                sha = row["header"]["sha256"]
                if sha in media_families and media_families[sha] != family:
                    raise BuildError("ambiguous_media_alias")
                media_families[sha] = family
                if row["file_name"] not in checked:
                    name = FIXED_ARCHIVE_ALIASES.get(row["file_name"], row["file_name"])
                    if name not in members or digest(archive.read(members[name])) != sha:
                        raise BuildError("media_source_mismatch")
                    checked.add(row["file_name"])
                key = (family, row["question"])
                if key in result:
                    if (
                        result[key]["answer"] != row["answer"]
                        or result[key]["header"]["sha256"] != sha
                    ):
                        raise BuildError("ambiguous_question_identity")
                    result[key]["source_splits"].append(split)
                else:
                    result[key] = {**row, "split": split, "source_splits": [split]}
    return result


def allocate(
    rows: dict[tuple[str, str], dict[str, Any]],
    existing: dict[tuple[str, str], Sample],
    families: dict[str, set[str]],
    media: dict[str, set[str]],
    quarantine: Quarantine,
    calibration_count: int,
    holdout_count: int,
) -> dict[str, list[tuple[str, str]]]:
    eligible: dict[str, list[tuple[str, str]]] = defaultdict(list)
    assigned: dict[str, str] = {}
    for key, row in rows.items():
        family, _ = key
        fh = canonical_family_hash(family)
        sha = row["header"]["sha256"]
        pools = families.get(fh, set()) | media.get(sha, set())
        if len(pools) > 1:
            raise BuildError("historical_pool_conflict")
        if fh in quarantine.families or sha in quarantine.media or pools & {"train", "dev"}:
            continue
        eligible[family].append(key)
        pool = next(iter(pools)) if pools else "unallocated"
        if family in assigned and assigned[family] != pool:
            raise BuildError("ambiguous_family_alias")
        assigned[family] = pool
    for key, sample in existing.items():
        original = rows.get(key)
        target = sample.questions[0].target
        if (
            original is None
            or key not in eligible.get(key[0], [])
            or assigned[key[0]] != sample.meta.pool
            or target is None
            or target.kind != "hard"
            or target.value != original["answer"]
            or sample.prefix.media[0].sha256 != original["header"]["sha256"]
            or sample.meta.source_revision != "zenodo-6473207"
        ):
            raise BuildError("invalid_existing_gold")
    output: dict[str, list[tuple[str, str]]] = {"calibration": [], "holdout": []}
    for family, keys in eligible.items():
        if assigned[family] in output:
            output[assigned[family]].extend(keys)
    extra = subset_sum(
        {f: len(keys) for f, keys in eligible.items() if assigned[f] == "unallocated"},
        holdout_count - len(output["holdout"]),
    )
    for family, keys in eligible.items():
        if assigned[family] == "unallocated":
            output["holdout" if family in extra else "calibration"].extend(keys)
    mandatory = [key for key in output["calibration"] if key in existing]
    additions = sorted(
        (key for key in output["calibration"] if key not in existing),
        key=lambda key: (rank("calibration", json.dumps(key)), key),
    )
    needed = calibration_count - len(mandatory)
    if needed < 0 or len(additions) < needed or len(output["holdout"]) != holdout_count:
        raise BuildError("insufficient_capacity")
    output["calibration"] = mandatory + additions[:needed]
    return output


def build(args: argparse.Namespace, receipt: Receipt) -> None:
    build_started = time.monotonic()
    payload = args.manifest.read_bytes()
    receipt.input_sha256["build_manifest"] = digest(payload)
    manifest = BuildManifest.model_validate_json(payload)
    if not Path(manifest.audio_root).is_absolute():
        raise BuildError("invalid_arguments")
    real = not args.canary
    if min(args.calibration_count, args.holdout_count) < 1 or (
        real and (args.calibration_count, args.holdout_count) != (1400, 500)
    ):
        raise BuildError("invalid_quota")
    for name in (*PINS, "inventory", "quarantine"):
        artifact = getattr(manifest, name)
        if real and name in PINS and artifact.sha256 != PINS[name]:
            raise BuildError("source_not_pinned")
        verify(artifact)
        receipt.input_sha256[name] = artifact.sha256
    inventory, families, media, existing = inventory_state(manifest, receipt)
    rows = collect(manifest)
    selected = allocate(
        rows,
        existing,
        families,
        media,
        Quarantine.model_validate(bound_json(manifest.quarantine)),
        args.calibration_count,
        args.holdout_count,
    )
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    media_dir = args.output_dir / "media"
    media_dir.mkdir(mode=0o700)
    derived: dict[str, tuple[Path, str]] = {}
    ledger = []
    artifacts = []
    for pool, keys in selected.items():
        samples = []
        for key in sorted(keys):
            row = rows[key]
            original_sha = row["header"]["sha256"]
            if original_sha not in derived:
                if file_digest(row["media_path"]) != original_sha:
                    raise BuildError("media_changed")
                info = decode_wav_full(row["media_path"])
                destination = media_dir / f"{original_sha}.wav"
                if info["sample_rate"] == 16000 and info["channels"] == 1:
                    write_private(destination, row["media_path"].read_bytes())
                else:
                    # Reserve output with private mode before ffmpeg writes it.
                    write_private(destination, b"")
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-nostdin",
                            "-y",
                            "-i",
                            str(row["media_path"]),
                            "-ar",
                            "16000",
                            "-ac",
                            "1",
                            "-c:a",
                            "pcm_s16le",
                            str(destination),
                        ],
                        check=True,
                        capture_output=True,
                    )
                decoded = decode_wav_full(destination)
                if (
                    decoded["sample_rate"] != 16000
                    or decoded["channels"] != 1
                    or decoded["duration_seconds"] > 30
                ):
                    raise BuildError("invalid_derived_media")
                derived[original_sha] = (destination, file_digest(destination))
            destination, derived_sha = derived[original_sha]
            if key in existing:
                sample = existing[key]
            else:
                raw = sample_for(row).model_dump(mode="json")
                source_hash = digest(
                    encode(
                        {
                            "original_source_hash": source_hash_for(row),
                            "question_artifacts": {
                                split: getattr(manifest, split).sha256
                                for split in row["source_splits"]
                            },
                            "original_media_sha256": original_sha,
                            "derived_media_sha256": derived_sha,
                        }
                    )
                )
                raw["id"] = f"clotho-aqa-{row['split']}-gold-{row['sound_id']}-{source_hash[:16]}"
                raw["prefix"]["media"][0].update(uri=destination.as_uri(), sha256=derived_sha)
                raw["meta"].update(
                    pool=pool,
                    source_sha256=source_hash,
                    data_version="clotho-original-gold-v1"
                    if real
                    else "clotho-canary-not-real-gold-v1",
                    usage="commercial" if real else "research",
                )
                sample = Sample.model_validate(raw)
            samples.append(sample)
            ledger.append(
                {
                    "item_id": sample.id,
                    "family_id": key[0],
                    "family_hash": canonical_family_hash(key[0]),
                    "pool": pool,
                    "preserved": key in existing,
                    "original_file_name": row["file_name"],
                    "archive_member_name": FIXED_ARCHIVE_ALIASES.get(
                        row["file_name"], row["file_name"]
                    ),
                    "question_sha256": digest(key[1].encode()),
                    "source_splits": row["source_splits"],
                    "original_source_sha256": source_hash_for(row),
                    "original_media_sha256": original_sha,
                    "derived_media_sha256": derived_sha,
                    "sample_media_sha256": sample.prefix.media[0].sha256,
                    "attribution": row["media"],
                }
            )
        path = args.output_dir / f"{pool}.jsonl"
        write_private(path, b"".join((s.model_dump_json() + "\n").encode() for s in samples))
        artifacts.append(
            SampleArtifact(
                path=str(path),
                sha256=file_digest(path),
                cohort="audio",
                exposure="current",
                expected_pool="calibration" if pool == "calibration" else "holdout",
            )
        )
        receipt.counts[pool] = len(samples)
    replaced = {(a.path, a.sha256) for a in manifest.current_reserves}
    combined = Manifest(
        schema_version="gold-inventory-manifest-v1",
        samples=[
            a.model_copy(update={"exposure": "historical"}) if (a.path, a.sha256) in replaced else a
            for a in inventory.samples
        ]
        + artifacts,
        component_ledgers=inventory.component_ledgers,
    )
    write_private(args.output_dir / "inventory-manifest.json", combined.model_dump_json().encode())
    write_private(args.output_dir / "allocation.jsonl", b"".join(encode(row) for row in ledger))
    write_private(args.output_dir / "build-manifest.json", payload)
    for name in (
        "calibration.jsonl",
        "holdout.jsonl",
        "inventory-manifest.json",
        "allocation.jsonl",
        "build-manifest.json",
    ):
        receipt.output_sha256[name] = file_digest(args.output_dir / name)
    if audit_manifest(args.output_dir / "inventory-manifest.json").outcome != "isolated":
        raise BuildError("output_isolation_failed")
    receipt.counts["preserved_items"] = len(existing)
    receipt.counts["fully_decoded_selected_media"] = len(derived)
    receipt.real_source_verified = real
    provenance = {
        "schema_version": "clotho-gold-provenance-v1",
        "mode": receipt.mode,
        "inputs": manifest.model_dump(mode="json"),
        "build_manifest_sha256": receipt.input_sha256["build_manifest"],
        "outputs": {
            name: {"path": str(args.output_dir / name), "sha256": sha}
            for name, sha in receipt.output_sha256.items()
        },
        "code_sha256": receipt.code_sha256,
        "code_commit": receipt.code_commit,
        "runtime": receipt.runtime,
        "elapsed_seconds": round(time.monotonic() - build_started, 6),
        "cost_usd": None,
        "seed": 42,
        "exit_code": 0,
    }
    write_private(args.output_dir / "provenance.json", encode(provenance))
    receipt.output_sha256["provenance.json"] = file_digest(args.output_dir / "provenance.json")
    receipt.outcome, receipt.category, receipt.exit_code = "built", "none", 0


def main(argv: list[str] | None = None) -> int:
    start = time.monotonic()
    receipt = Receipt(runtime=f"Python {platform.python_version()} / {platform.system()}")
    args = None
    try:
        receipt.code_sha256 = file_digest(Path(__file__))
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            receipt.code_commit = commit.stdout.strip()
        parser = PrivateParser(description=__doc__)
        for name in ("manifest", "output-dir", "report"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--canary", action="store_true")
        parser.add_argument("--calibration-count", type=int, default=1400)
        parser.add_argument("--holdout-count", type=int, default=500)
        args = parser.parse_args(argv)
        receipt.mode = "canary" if args.canary else "real"
        if args.canary:
            receipt.source_revision = "synthetic-canary"
        paths = [args.manifest, args.output_dir, args.report]
        if any(not path.is_absolute() for path in paths) or len({p.resolve() for p in paths}) != 3:
            raise BuildError("invalid_arguments")
        if any(p.exists() or p.is_symlink() for p in [args.output_dir, args.report]):
            raise BuildError("output_exists")
        build(args, receipt)
    except BuildError as error:
        receipt.category = str(error)
    except Rejected as error:
        receipt.category = error.category
    except zipfile.BadZipFile:
        receipt.category = "invalid_archive"
    except (ValueError, KeyError, TypeError):
        receipt.category = "invalid_metadata"
    except OSError:
        receipt.category = "io_error"
    except Exception:
        receipt.category = "job_failed"
    receipt.elapsed_seconds = round(time.monotonic() - start, 6)
    if args is not None and args.report.is_absolute():
        try:
            write_private(args.report, receipt.model_dump_json().encode() + b"\n")
        except Exception:
            receipt.outcome, receipt.category, receipt.exit_code = "rejected", "report_io", 1
    print(receipt.model_dump_json())
    return receipt.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
