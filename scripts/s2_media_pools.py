"""Freeze S2 media cohort pools: train, calibration-reserve, holdout-reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from data.schema import HardTarget, Sample, validate_pool_isolation
from scripts.s0_media_dev import AOK_MEDIA_LICENSE, AOK_QA_LICENSE, IMAGE_DATA_VERSION

SEED = 42
COHORT_SPECS: dict[str, dict[str, Any]] = {
    "image": {
        "source_dataset": "A-OKVQA",
        "source_revision": "42e8c44ff40d13cf272776a3197365f582137772",
        "license": f"QA:{AOK_QA_LICENSE};media:{AOK_MEDIA_LICENSE}",
        "modality": "image",
        "media_type": "image",
        "source_data_version": IMAGE_DATA_VERSION,
        "pools_data_version": "aokvqa-train-image-choice-s2-pools-v1",
        "expected_dev_sha256": "38efa78efe62ebdd8523c7372b8c0ac87a9030a18d429db488f44e5aad08e7de",
        "holdout_reserve_count": 200,
        "calibration_reserve_count": 600,
    },
    "screenshot": {
        "source_dataset": "hflqf88888/GUIOdyssey",
        "source_revision": "61632d0f3f4d51d7e9561ce4f84347dd06b2019d",
        "license": "CC-BY-4.0",
        "modality": "image",
        "media_type": "image",
        "source_data_version": None,
        "pools_data_version": "gui-v2-screenshot-s2-pools-v1",
        "expected_dev_sha256": None,
        "holdout_reserve_count": 110,
        "calibration_reserve_count": 330,
    },
}


class MediaPoolsError(ValueError):
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


def protocol_name(cohort: str) -> str:
    return f"s2-{cohort}-pools-v1"


def allocation_prefix(cohort: str) -> str:
    return f"{protocol_name(cohort)}:{SEED}:"


def allocation_key(cohort: str, family_id: str) -> str:
    return sha256_bytes(f"{allocation_prefix(cohort)}{family_id}".encode())


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
                raise MediaPoolsError(f"invalid sample JSON at line {line_no}") from exc
    return samples


def load_dev_families(path: Path, expected_sha256: str) -> set[str]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise MediaPoolsError("unexpected dev JSONL sha256")
    return {sample.meta.family_id for sample in load_samples(path)}


def safe_media_uri(uri: str) -> Path:
    relative = Path(uri)
    if ":" in uri or relative.is_absolute() or ".." in relative.parts or "\x00" in uri:
        raise MediaPoolsError("unsafe relative media path")
    if any(part != PurePosixPath(part).name for part in relative.parts):
        raise MediaPoolsError("unsafe relative media path")
    return relative


def validate_source_samples(
    samples: list[Sample], *, spec: dict[str, Any], media_root: Path
) -> dict[str, Any]:
    ids: set[str] = set()
    families: set[str] = set()
    conflict_by_family: dict[str, dict[str, str]] = {}
    verified: dict[str, str] = {}
    for sample in samples:
        if sample.id in ids:
            raise MediaPoolsError("duplicate sample id")
        ids.add(sample.id)
        meta = sample.meta
        if meta.source_dataset != spec["source_dataset"]:
            raise MediaPoolsError("unexpected source_dataset")
        if meta.source_revision != spec["source_revision"]:
            raise MediaPoolsError("unexpected source_revision")
        if meta.license != spec["license"]:
            raise MediaPoolsError("unexpected license")
        if meta.data_version != spec["source_data_version"]:
            raise MediaPoolsError("unexpected source data_version")
        if spec["modality"] not in meta.modality:
            raise MediaPoolsError("source samples must declare the cohort modality")
        if len(sample.questions) != 1 or sample.questions[0].type != "choice":
            raise MediaPoolsError("source samples must be single Choice questions")
        target = sample.questions[0].target
        if not isinstance(target, HardTarget) or type(target.value) is not str:
            raise MediaPoolsError("source samples must use hard answer-token targets")
        media_entries = sample.prefix.media
        if len(media_entries) != 1:
            raise MediaPoolsError(f"sample {sample.id} must reference exactly one media file")
        media = media_entries[0]
        if media.type != spec["media_type"]:
            raise MediaPoolsError(f"sample {sample.id} media type mismatch")
        if media.uri not in verified:
            path = media_root / safe_media_uri(media.uri)
            if path.is_symlink() or not path.is_file():
                raise MediaPoolsError(f"media file missing for sample {sample.id}: {media.uri}")
            verified[media.uri] = sha256_file(path)
        if verified[media.uri] != media.sha256:
            raise MediaPoolsError(f"media sha256 mismatch for sample {sample.id}: {media.uri}")
        family = meta.family_id
        families.add(family)
        # 同图多问题合法（instruction/options 可不同）；仅完全相同题面配矛盾 target 才是真冲突。
        conflict_key = canonical_json(
            {"instruction": sample.questions[0].instruction, "options": sample.questions[0].options}
        )
        seen = conflict_by_family.setdefault(family, {})
        prior_target = seen.get(conflict_key)
        if prior_target is not None and prior_target != target.value:
            raise MediaPoolsError(
                f"family {family} has conflicting targets for an identical question"
            )
        seen[conflict_key] = target.value
    return {
        "families": len(families),
        "rows": len(samples),
        "media_verified": len(verified),
    }


def with_pool_and_version(sample: Sample, *, pool: str, data_version: str) -> Sample:
    payload = sample.model_dump(mode="json")
    payload["meta"]["pool"] = pool
    payload["meta"]["data_version"] = data_version
    return Sample.model_validate(payload)


def split_pools(
    samples: list[Sample],
    *,
    cohort: str,
    spec: dict[str, Any],
    holdout_count: int,
    calibration_count: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    ordered = sorted(
        samples,
        key=lambda sample: (
            allocation_key(cohort, sample.meta.family_id),
            sample.meta.family_id,
            sample.id,
        ),
    )
    family_order: list[str] = []
    seen: set[str] = set()
    for sample in ordered:
        family = sample.meta.family_id
        if family not in seen:
            seen.add(family)
            family_order.append(family)
    holdout_families = set(family_order[:holdout_count])
    calibration_families = set(family_order[holdout_count : holdout_count + calibration_count])
    version = str(spec["pools_data_version"])
    holdout = [
        with_pool_and_version(sample, pool="holdout", data_version=version)
        for sample in ordered
        if sample.meta.family_id in holdout_families
    ]
    calibration = [
        with_pool_and_version(sample, pool="calibration", data_version=version)
        for sample in ordered
        if sample.meta.family_id in calibration_families
    ]
    reserved = holdout_families | calibration_families
    train = [
        with_pool_and_version(sample, pool="train", data_version=version)
        for sample in ordered
        if sample.meta.family_id not in reserved
    ]
    if not train:
        raise MediaPoolsError("allocation leaves an empty train pool")
    validate_pool_isolation(holdout + calibration + train)
    return holdout, calibration, train


def allocation_rows(
    pools: dict[str, list[Sample]],
    *,
    cohort: str,
    source_data_version: str,
    original_pools: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for samples in pools.values():
        for sample in samples:
            original = with_pool_and_version(
                sample, pool=original_pools[sample.id], data_version=source_data_version
            )
            rows.append(
                {
                    "item_id": sample.id,
                    "pool": sample.meta.pool,
                    "family_id": sample.meta.family_id,
                    "family_hash": canonical_family_hash(sample.meta.family_id),
                    "allocation_key_sha256": allocation_key(cohort, sample.meta.family_id),
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
        raise MediaPoolsError("output paths must be unique")
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise MediaPoolsError("refusing to overwrite existing outputs")


def config_payload(
    *,
    cohort: str,
    spec: dict[str, Any],
    holdout_count: int,
    calibration_count: int,
    expected_dev_sha256: str,
    expected_source_data_version: str,
) -> dict[str, Any]:
    return {
        "protocol": protocol_name(cohort),
        "cohort": cohort,
        "seed": SEED,
        "allocation_prefix": allocation_prefix(cohort),
        "source_dataset": spec["source_dataset"],
        "source_revision": spec["source_revision"],
        "license": spec["license"],
        "expected_dev_sha256": expected_dev_sha256,
        "expected_source_data_version": expected_source_data_version,
        "pools_data_version": spec["pools_data_version"],
        "media_type": spec["media_type"],
        "holdout_reserve_count": holdout_count,
        "calibration_reserve_count": calibration_count,
    }


def build_report(
    *,
    status: str,
    exit_code: int,
    command: list[str],
    cohort: str,
    started_at: float,
    config: dict[str, Any],
    blocker: str | None = None,
    source: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finished_at = time.time()
    return {
        "status": status,
        "exit_code": exit_code,
        "blocker": blocker,
        "command": command,
        "seed": SEED,
        "protocol": protocol_name(cohort),
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
    cohort: str,
    source_path: Path,
    media_root: Path,
    dev_path: Path,
    train_out: Path,
    calibration_out: Path,
    holdout_out: Path,
    allocation_ledger_out: Path,
    report_out: Path,
    holdout_count: int,
    calibration_count: int,
    expected_dev_sha256: str,
    expected_source_data_version: str,
    command: list[str],
) -> int:
    started_at = time.time()
    report: dict[str, Any]
    spec = {**COHORT_SPECS[cohort], "source_data_version": expected_source_data_version}
    base_config = {
        **config_payload(
            cohort=cohort,
            spec=spec,
            holdout_count=holdout_count,
            calibration_count=calibration_count,
            expected_dev_sha256=expected_dev_sha256,
            expected_source_data_version=expected_source_data_version,
        ),
    }
    config = {
        **base_config,
        "config_sha256": sha256_bytes(canonical_json(base_config).encode()),
    }
    output_paths = [train_out, calibration_out, holdout_out, allocation_ledger_out]
    if report_out.exists() or report_out.is_symlink():
        return 1
    if report_out.resolve(strict=False) in {path.resolve(strict=False) for path in output_paths}:
        return 1
    try:
        validate_output_paths([*output_paths, report_out])
        actual_source_sha256 = sha256_file(source_path)
        samples = load_samples(source_path)
        original_pools = {sample.id: sample.meta.pool for sample in samples}
        validation = validate_source_samples(samples, spec=spec, media_root=media_root)
        dev_families = {
            canonical_family_hash(family)
            for family in load_dev_families(dev_path, expected_dev_sha256)
        }
        # 候选文件本来就含 dev500：切分前按 family 剔除（记录行数）。
        total_input_rows = len(samples)
        samples = [
            sample
            for sample in samples
            if canonical_family_hash(sample.meta.family_id) not in dev_families
        ]
        dev_excluded_rows = total_input_rows - len(samples)
        candidate_families = {canonical_family_hash(sample.meta.family_id) for sample in samples}
        dev_overlap = candidate_families & dev_families
        if dev_overlap:
            raise MediaPoolsError(f"pools overlap dev families after exclusion: {len(dev_overlap)}")
        if not samples:
            raise MediaPoolsError("no candidate samples remain after dev exclusion")
        held, calibration, train = split_pools(
            samples,
            cohort=cohort,
            spec=spec,
            holdout_count=holdout_count,
            calibration_count=calibration_count,
        )
        pools = {
            "holdout_reserve": held,
            "calibration_reserve": calibration,
            "train": train,
        }
        ledger = allocation_rows(
            pools,
            cohort=cohort,
            source_data_version=expected_source_data_version,
            original_pools=original_pools,
        )
        train_hash = write_jsonl(train_out, train)
        calibration_hash = write_jsonl(calibration_out, calibration)
        holdout_hash = write_jsonl(holdout_out, held)
        ledger_hash = write_jsonl(allocation_ledger_out, ledger)
        source = {
            "source_jsonl_sha256": actual_source_sha256,
            "source_dataset": spec["source_dataset"],
            "source_revision": spec["source_revision"],
            "license": spec["license"],
            "media_root": str(media_root),
            **validation,
            "dev_families": len(dev_families),
        }
        split = {
            "dev_excluded_rows": dev_excluded_rows,
            "train_items": len(train),
            "calibration_reserve_items": len(calibration),
            "holdout_reserve_items": len(held),
            "train_families": len({sample.meta.family_id for sample in train}),
            "calibration_reserve_families": len({sample.meta.family_id for sample in calibration}),
            "holdout_reserve_families": len({sample.meta.family_id for sample in held}),
            "dev_family_overlap": 0,
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
            cohort=cohort,
            started_at=started_at,
            config=config,
            source=source,
            split=split,
            outputs=outputs,
        )
        code = 0
    except (OSError, MediaPoolsError, ValueError) as exc:
        report = build_report(
            status="blocked",
            exit_code=1,
            command=command,
            cohort=cohort,
            started_at=started_at,
            config=config,
            blocker=str(exc) if isinstance(exc, MediaPoolsError) else type(exc).__name__,
        )
        code = 1
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=sorted(COHORT_SPECS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--calibration-out", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--allocation-ledger-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--holdout-count", type=int, default=None)
    parser.add_argument("--calibration-count", type=int, default=None)
    parser.add_argument("--expected-dev-sha256", default=None)
    parser.add_argument("--expected-source-data-version", default=None)
    args = parser.parse_args(argv)
    spec = COHORT_SPECS[args.cohort]
    if args.holdout_count is None:
        args.holdout_count = int(spec["holdout_reserve_count"])
    if args.calibration_count is None:
        args.calibration_count = int(spec["calibration_reserve_count"])
    for flag, dest, key in (
        ("--expected-dev-sha256", "expected_dev_sha256", "expected_dev_sha256"),
        ("--expected-source-data-version", "expected_source_data_version", "source_data_version"),
    ):
        if getattr(args, dest) is None:
            default = spec[key]
            if default is None:
                parser.error(f"{flag} is required for cohort {args.cohort}")
            setattr(args, dest, str(default))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        cohort=args.cohort,
        source_path=args.source,
        media_root=args.media_root,
        dev_path=args.dev,
        train_out=args.train_out,
        calibration_out=args.calibration_out,
        holdout_out=args.holdout_out,
        allocation_ledger_out=args.allocation_ledger_out,
        report_out=args.report,
        holdout_count=args.holdout_count,
        calibration_count=args.calibration_count,
        expected_dev_sha256=args.expected_dev_sha256,
        expected_source_data_version=args.expected_source_data_version,
        command=[sys.executable, "-m", "scripts.s2_media_pools", *(argv or sys.argv[1:])],
    )


if __name__ == "__main__":
    raise SystemExit(main())
