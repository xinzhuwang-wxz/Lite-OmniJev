"""Freeze S1 CLINC text pools: train, calibration-reserve, holdout-reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample, validate_pool_isolation

SOURCE_DATA_VERSION = "clinc-oos-plus-train-v1-text-dev-excluded"
EXPECTED_SOURCE_SHA256 = "12a42005478ff1cd2f618fbe4951ced1af878b9e115a9a055c57d9d79578cb11"
EXPECTED_SOURCE_ROWS = 14_750
EXPECTED_DEV_SHA256 = "8c7809e8b07673904b71bc4676a49a5e31abb22ee7126f2dc09e8fde45b4e070"
SOURCE_DATASET = "clinc/clinc_oos"
SOURCE_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
LICENSE = "CC-BY-3.0"
S1_POOLS_VERSION = "clinc-oos-plus-train-v1-s1-pools-v1"
SEED = 42
HOLDOUT_RESERVE_COUNT = 1_000
CALIBRATION_RESERVE_COUNT = 3_000
ALLOCATION_PREFIX = f"s1-text-pools-v1:{SEED}:"


class TextPoolsError(ValueError):
    """User-correctable input or output contract failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sample_sha(sample: Sample) -> str:
    return sha256_bytes(canonical_json(sample.model_dump(mode="json")).encode())


def canonical_family_hash(family_id: str) -> str:
    lowered = family_id.lower()
    if len(lowered) == 64 and all(char in "0123456789abcdef" for char in lowered):
        return lowered
    return sha256_bytes(family_id.encode())


def allocation_key(family_id: str) -> str:
    return sha256_bytes(f"{ALLOCATION_PREFIX}{family_id}".encode())


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                samples.append(Sample.model_validate_json(stripped))
            except ValueError as exc:
                raise TextPoolsError(f"invalid sample JSON at line {line_no}") from exc
    return samples


def load_dev_families(path: Path, expected_sha256: str) -> set[str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise TextPoolsError("unexpected dev500 sha256")
    return {sample.meta.family_id for sample in load_samples(path)}


def load_exclusion_families(path: Path) -> set[str]:
    """Family nodes from the historical dev-exclusion union; format-guarded."""
    families: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except ValueError as exc:
                raise TextPoolsError(f"invalid exclusion JSON at line {line_no}") from exc
            family = row.get("family_id") or row.get("family_hash") or row.get("family")
            if not isinstance(family, str) or not family:
                raise TextPoolsError(f"exclusion row without family at line {line_no}")
            families.add(canonical_family_hash(family))
    return families


def validate_source_samples(samples: list[Sample], *, expected_count: int) -> dict[str, Any]:
    if len(samples) != expected_count:
        raise TextPoolsError(f"expected {expected_count} source samples, found {len(samples)}")
    ids: set[str] = set()
    families: set[str] = set()
    for sample in samples:
        if sample.id in ids:
            raise TextPoolsError("duplicate sample id")
        ids.add(sample.id)
        meta = sample.meta
        if meta.source_dataset != SOURCE_DATASET:
            raise TextPoolsError("unexpected source_dataset")
        if meta.source_revision != SOURCE_REVISION:
            raise TextPoolsError("unexpected source_revision")
        if meta.license != LICENSE:
            raise TextPoolsError("unexpected license")
        if meta.pool != "train":
            raise TextPoolsError("source samples must all be train pool")
        if meta.data_version != SOURCE_DATA_VERSION:
            raise TextPoolsError("unexpected source data_version")
        if meta.modality != ["text"]:
            raise TextPoolsError("source samples must be text-only")
        if len(sample.questions) != 1 or sample.questions[0].type != "choice":
            raise TextPoolsError("source samples must be single Choice questions")
        target = sample.questions[0].target
        if not isinstance(target, HardTarget) or type(target.value) is not str:
            raise TextPoolsError("source samples must use hard answer-token targets")
        if target.source != "clinc_human_label":
            raise TextPoolsError("unexpected target source")
        family = meta.family_id
        if family in families:
            raise TextPoolsError("source contains a repeated family; per-row split is invalid")
        families.add(family)
    return {"families": len(families), "rows": len(samples)}


def split_pools(
    samples: list[Sample],
    *,
    holdout_count: int = HOLDOUT_RESERVE_COUNT,
    calibration_count: int = CALIBRATION_RESERVE_COUNT,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    ordered = sorted(
        samples, key=lambda sample: (allocation_key(sample.meta.family_id), sample.meta.family_id)
    )
    holdout = [with_pool_and_version(sample, pool="holdout") for sample in ordered[:holdout_count]]
    calibration = [
        with_pool_and_version(sample, pool="calibration")
        for sample in ordered[holdout_count : holdout_count + calibration_count]
    ]
    train = [
        with_pool_and_version(sample, pool="train")
        for sample in ordered[holdout_count + calibration_count :]
    ]
    if not train:
        raise TextPoolsError("allocation leaves an empty train pool")
    validate_pool_isolation(holdout + calibration + train)
    return holdout, calibration, train


def with_pool_and_version(sample: Sample, *, pool: str) -> Sample:
    payload = sample.model_dump(mode="json")
    payload["meta"]["pool"] = pool
    payload["meta"]["data_version"] = S1_POOLS_VERSION
    return Sample.model_validate(payload)


def allocation_rows(
    pools: dict[str, list[Sample]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for samples in pools.values():
        for sample in samples:
            original = with_pool_and_version(sample, pool="train")
            rows.append(
                {
                    "item_id": sample.id,
                    "pool": sample.meta.pool,
                    "family_id": sample.meta.family_id,
                    "family_hash": canonical_family_hash(sample.meta.family_id),
                    "allocation_key_sha256": allocation_key(sample.meta.family_id),
                    "original_sample_sha256": canonical_sample_sha(original),
                    "derived_sample_sha256": canonical_sample_sha(sample),
                    "source_dataset": sample.meta.source_dataset,
                    "source_revision": sample.meta.source_revision,
                    "license": sample.meta.license,
                }
            )
    return sorted(rows, key=lambda row: (str(row["pool"]), str(row["item_id"])))


def write_jsonl(path: Path, rows: list[Sample] | list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if isinstance(row, Sample):
            lines.append(row.model_dump_json())
        else:
            lines.append(canonical_json(row))
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode())


def validate_output_paths(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise TextPoolsError("output paths must be unique")
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise TextPoolsError("refusing to overwrite existing outputs")


def config_payload(*, expected_source_sha256: str, expected_count: int) -> dict[str, Any]:
    return {
        "protocol": "s1-text-pools-v1",
        "seed": SEED,
        "allocation_prefix": ALLOCATION_PREFIX,
        "expected_source_sha256": expected_source_sha256,
        "expected_source_rows": expected_count,
        "source_data_version": SOURCE_DATA_VERSION,
        "pools_data_version": S1_POOLS_VERSION,
        "holdout_reserve_count": HOLDOUT_RESERVE_COUNT,
        "calibration_reserve_count": CALIBRATION_RESERVE_COUNT,
    }


def build_report(
    *,
    status: str,
    exit_code: int,
    command: list[str],
    started_at: float,
    blocker: str | None = None,
    source: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finished_at = time.time()
    return {
        "status": status,
        "exit_code": exit_code,
        "blocker": blocker,
        "command": command,
        "seed": SEED,
        "protocol": "s1-text-pools-v1",
        "config": config,
        "training_admitted": False,
        "quality_gate": "not_run",
        "gpu_used": False,
        "holdout_content_accessed": False,
        "cost": None,
        "source": source,
        "split": split,
        "outputs": outputs,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": git_head(),
            "script_sha256": script_sha256(),
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": finished_at - started_at,
        },
    }


def run(
    *,
    source_path: Path,
    dev_path: Path,
    exclusion_path: Path | None,
    train_out: Path,
    calibration_out: Path,
    holdout_out: Path,
    allocation_ledger_out: Path,
    report_out: Path,
    expected_source_sha256: str,
    expected_count: int,
    holdout_count: int,
    calibration_count: int,
    command: list[str],
) -> int:
    started_at = time.time()
    report: dict[str, Any]
    output_paths = [train_out, calibration_out, holdout_out, allocation_ledger_out]
    if report_out.exists() or report_out.is_symlink():
        return 1
    if report_out.resolve(strict=False) in {path.resolve(strict=False) for path in output_paths}:
        return 1
    try:
        validate_output_paths([*output_paths, report_out])
        actual_source_sha256 = sha256_file(source_path)
        if actual_source_sha256 != expected_source_sha256:
            raise TextPoolsError("unexpected source JSONL sha256")
        samples = load_samples(source_path)
        validation = validate_source_samples(samples, expected_count=expected_count)
        dev_families = {
            canonical_family_hash(family)
            for family in load_dev_families(dev_path, EXPECTED_DEV_SHA256)
        }
        exclusion_families = load_exclusion_families(exclusion_path) if exclusion_path else set()
        held, calibration, train = split_pools(
            samples, holdout_count=holdout_count, calibration_count=calibration_count
        )
        pools = {
            "holdout_reserve": held,
            "calibration_reserve": calibration,
            "train": train,
        }
        reserved_families = {
            canonical_family_hash(sample.meta.family_id) for sample in [*held, *calibration, *train]
        }
        dev_overlap = reserved_families & dev_families
        if dev_overlap:
            raise TextPoolsError(f"pools overlap dev500 families: {len(dev_overlap)}")
        exclusion_overlap = reserved_families & exclusion_families
        if exclusion_overlap:
            raise TextPoolsError(
                f"pools overlap exclusion union families: {len(exclusion_overlap)}"
            )
        ledger = allocation_rows(pools)
        train_hash = write_jsonl(train_out, train)
        calibration_hash = write_jsonl(calibration_out, calibration)
        holdout_hash = write_jsonl(holdout_out, held)
        ledger_hash = write_jsonl(allocation_ledger_out, ledger)
        source = {
            "source_jsonl_sha256": actual_source_sha256,
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "license": LICENSE,
            **validation,
            "dev_families": len(dev_families),
            "exclusion_union_families": len(exclusion_families),
            "exclusion_checked": exclusion_path is not None,
        }
        split = {
            "train_items": len(train),
            "calibration_reserve_items": len(calibration),
            "holdout_reserve_items": len(held),
            "dev_family_overlap": 0,
            "exclusion_family_overlap": 0 if exclusion_path else None,
        }
        outputs = {
            "train_jsonl_sha256": train_hash,
            "calibration_reserve_jsonl_sha256": calibration_hash,
            "holdout_reserve_jsonl_sha256": holdout_hash,
            "allocation_ledger_sha256": ledger_hash,
        }
        report = build_report(
            status="passed",
            exit_code=0,
            command=command,
            started_at=started_at,
            source=source,
            split=split,
            outputs=outputs,
            config={
                **config_payload(
                    expected_source_sha256=expected_source_sha256, expected_count=expected_count
                ),
                "config_sha256": sha256_bytes(
                    canonical_json(
                        config_payload(
                            expected_source_sha256=expected_source_sha256,
                            expected_count=expected_count,
                        )
                    ).encode()
                ),
            },
        )
        code = 0
    except (OSError, TextPoolsError, ValueError) as exc:
        report = build_report(
            status="blocked",
            exit_code=1,
            command=command,
            started_at=started_at,
            blocker=str(exc) if isinstance(exc, TextPoolsError) else type(exc).__name__,
            config={
                **config_payload(
                    expected_source_sha256=expected_source_sha256, expected_count=expected_count
                ),
                "config_sha256": sha256_bytes(
                    canonical_json(
                        config_payload(
                            expected_source_sha256=expected_source_sha256,
                            expected_count=expected_count,
                        )
                    ).encode()
                ),
            },
        )
        code = 1
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--exclusion-union", type=Path, default=None)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--calibration-out", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--allocation-ledger-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SOURCE_ROWS)
    parser.add_argument("--holdout-count", type=int, default=HOLDOUT_RESERVE_COUNT)
    parser.add_argument("--calibration-count", type=int, default=CALIBRATION_RESERVE_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        source_path=args.source,
        dev_path=args.dev,
        exclusion_path=args.exclusion_union,
        train_out=args.train_out,
        calibration_out=args.calibration_out,
        holdout_out=args.holdout_out,
        allocation_ledger_out=args.allocation_ledger_out,
        report_out=args.report,
        expected_source_sha256=args.expected_source_sha256,
        expected_count=args.expected_count,
        holdout_count=args.holdout_count,
        calibration_count=args.calibration_count,
        command=[sys.executable, "-m", "scripts.s1_text_pools", *(argv or sys.argv[1:])],
    )


if __name__ == "__main__":
    raise SystemExit(main())
