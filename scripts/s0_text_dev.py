"""Build the first public text development cohort from admitted CLINC train JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample, validate_pool_isolation
from eval.sealed_gate import DevelopmentRecord
from scripts.admit_clinc import (
    DATA_VERSION as CLINC_DATA_VERSION,
)
from scripts.admit_clinc import (
    EXPECTED_TRAIN_SHA256 as CLINC_EXPECTED_TRAIN_SHA256,
)
from scripts.admit_clinc import (
    LICENSE as CLINC_LICENSE,
)
from scripts.admit_clinc import (
    REPO_ID as CLINC_REPO_ID,
)
from scripts.admit_clinc import (
    REVISION as CLINC_REVISION,
)

REPO_ID: str = CLINC_REPO_ID
REVISION: str = CLINC_REVISION
LICENSE: str = CLINC_LICENSE
DATA_VERSION: str = CLINC_DATA_VERSION
EXPECTED_TRAIN_SHA256: str = CLINC_EXPECTED_TRAIN_SHA256

EXPECTED_ADMITTED_SHA256 = "5aace71f0c9af2b30630e2b2b379881e1ebf946221c59355e733fe496e5971d8"
EXPECTED_ADMITTED_ROWS = 15_250
TEXT_DEV_VERSION = "clinc-oos-plus-train-text-dev-v1"
TEXT_TRAIN_VERSION = "clinc-oos-plus-train-v1-text-dev-excluded"
SEED = 42
DEV_COUNT = 500
ALLOCATION_PREFIX = f"s0-text-dev-v1:{SEED}:"


class TextDevError(ValueError):
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


def target_intent_display(sample: Sample) -> str:
    question = sample.questions[0]
    target = question.target
    if not isinstance(target, HardTarget) or type(target.value) is not str:
        raise TextDevError("Choice sample lacks a hard token target")
    index = ord(target.value) - ord("A")
    options = question.options or []
    if not 0 <= index < len(options):
        raise TextDevError("Choice target token is outside options")
    return options[index]


def intent_display_counts(samples: list[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(target_intent_display(sample) for sample in samples).items()))


def config_payload(
    *, expected_source_sha256: str, expected_count: int, dev_count: int
) -> dict[str, Any]:
    return {
        "protocol": "s0-text-dev-v1",
        "seed": SEED,
        "allocation_prefix": ALLOCATION_PREFIX,
        "dev_count": dev_count,
        "expected_count": expected_count,
        "expected_admitted_jsonl_sha256": expected_source_sha256,
        "expected_raw_source_sha256": EXPECTED_TRAIN_SHA256,
        "source_data_version": DATA_VERSION,
        "dev_data_version": TEXT_DEV_VERSION,
        "train_data_version": TEXT_TRAIN_VERSION,
    }


def config_sha256(*, expected_source_sha256: str, expected_count: int, dev_count: int) -> str:
    return sha256_bytes(
        canonical_json(
            config_payload(
                expected_source_sha256=expected_source_sha256,
                expected_count=expected_count,
                dev_count=dev_count,
            )
        ).encode()
    )


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
                raise TextDevError(f"invalid sample JSON at line {line_no}") from exc
    return samples


def with_pool_and_version(sample: Sample, *, pool: str, data_version: str) -> Sample:
    payload = sample.model_dump(mode="json")
    payload["meta"]["pool"] = pool
    payload["meta"]["data_version"] = data_version
    return Sample.model_validate(payload)


def validate_source_samples(
    samples: list[Sample],
    *,
    expected_count: int,
    expected_raw_source_sha256: str,
) -> dict[str, Any]:
    if len(samples) != expected_count:
        raise TextDevError(f"expected {expected_count} admitted samples, found {len(samples)}")
    ids: set[str] = set()
    families: dict[str, Sample] = {}
    target_by_family: dict[str, str] = {}
    intent_counts: Counter[str] = Counter()
    option_sizes: Counter[int] = Counter()
    for sample in samples:
        if sample.id in ids:
            raise TextDevError("duplicate sample id")
        ids.add(sample.id)
        meta = sample.meta
        if meta.source_dataset != REPO_ID:
            raise TextDevError("unexpected source_dataset")
        if meta.source_revision != REVISION:
            raise TextDevError("unexpected source_revision")
        if meta.license != LICENSE:
            raise TextDevError("unexpected license")
        if meta.source_sha256 != expected_raw_source_sha256:
            raise TextDevError("unexpected raw source_sha256")
        if meta.pool != "train":
            raise TextDevError("source samples must all be train pool")
        if meta.data_version != DATA_VERSION:
            raise TextDevError("unexpected source data_version")
        if meta.modality != ["text"]:
            raise TextDevError("source samples must be text-only")
        if len(sample.questions) != 1 or sample.questions[0].type != "choice":
            raise TextDevError("source samples must be single Choice questions")
        target = sample.questions[0].target
        if not isinstance(target, HardTarget):
            raise TextDevError("source samples must use hard targets")
        if type(target.value) is not str:
            raise TextDevError("Choice target must be an answer token")
        if target.source != "clinc_human_label":
            raise TextDevError("unexpected target source")
        family = meta.family_id
        if family in families:
            previous = families[family]
            if canonical_sample_sha(previous) != canonical_sample_sha(sample):
                raise TextDevError("fixed CLINC source contains a multi-sample family")
        families[family] = sample
        previous_target = target_by_family.get(family)
        if previous_target is not None and previous_target != target.value:
            raise TextDevError("fixed CLINC source contains conflicting targets within a family")
        target_by_family[family] = target.value
        intent_counts[target_intent_display(sample)] += 1
        option_sizes[len(sample.questions[0].options or [])] += 1
    if len(families) != len(samples):
        raise TextDevError("fixed CLINC source contains duplicate families")
    return {
        "families": len(families),
        "choice_only": True,
        "intent_display_counts": dict(sorted(intent_counts.items())),
        "intent_display_coverage": len(intent_counts),
        "option_group_sizes": {str(key): value for key, value in sorted(option_sizes.items())},
    }


def split_samples(samples: list[Sample], *, dev_count: int) -> tuple[list[Sample], list[Sample]]:
    if len(samples) < dev_count:
        raise TextDevError(f"not enough single-sample families for dev{dev_count}")
    ordered = sorted(
        samples, key=lambda sample: (allocation_key(sample.meta.family_id), sample.meta.family_id)
    )
    dev_ids = {sample.id for sample in ordered[:dev_count]}
    dev = [
        with_pool_and_version(sample, pool="dev", data_version=TEXT_DEV_VERSION)
        for sample in samples
        if sample.id in dev_ids
    ]
    train = [
        with_pool_and_version(sample, pool="train", data_version=TEXT_TRAIN_VERSION)
        for sample in samples
        if sample.id not in dev_ids
    ]
    validate_pool_isolation(dev + train)
    return dev, train


def dev_ledger_rows(dev_samples: list[Sample]) -> list[DevelopmentRecord]:
    rows: list[DevelopmentRecord] = []
    for sample in dev_samples:
        original = with_pool_and_version(sample, pool="train", data_version=DATA_VERSION)
        rows.append(
            DevelopmentRecord(
                item_id=sample.id,
                modality="text",
                family_hash=canonical_family_hash(sample.meta.family_id),
                source_hash=canonical_sample_sha(original),
                source_dataset=sample.meta.source_dataset,
                source_revision=sample.meta.source_revision,
                license=sample.meta.license,
                label_origin="hard_gold",
            )
        )
    return rows


def allocation_rows(dev_samples: list[Sample], train_samples: list[Sample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in [*dev_samples, *train_samples]:
        original = with_pool_and_version(sample, pool="train", data_version=DATA_VERSION)
        rows.append(
            {
                "item_id": sample.id,
                "pool": sample.meta.pool,
                "family_id": sample.meta.family_id,
                "family_hash": canonical_family_hash(sample.meta.family_id),
                "allocation_key_sha256": allocation_key(sample.meta.family_id),
                "original_sample_sha256": canonical_sample_sha(original),
                "derived_sample_sha256": canonical_sample_sha(sample),
                "source_hash": canonical_sample_sha(original),
                "source_dataset": sample.meta.source_dataset,
                "source_revision": sample.meta.source_revision,
                "license": sample.meta.license,
            }
        )
    return sorted(rows, key=lambda row: (str(row["pool"]), str(row["item_id"])))


def write_jsonl(
    path: Path, rows: list[Sample] | list[DevelopmentRecord] | list[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if isinstance(row, Sample | DevelopmentRecord):
            lines.append(row.model_dump_json())
        else:
            lines.append(canonical_json(row))
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode())


def validate_output_paths(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise TextDevError("output paths must be unique")
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise TextDevError("refusing to overwrite existing outputs")


def validate_split_bounds(*, expected_count: int, dev_count: int) -> None:
    if expected_count <= 1:
        raise TextDevError("expected_count must leave both dev and train items")
    if not 1 <= dev_count <= expected_count - 1:
        raise TextDevError("dev_count must be between 1 and expected_count - 1")


def build_report(
    *,
    status: str,
    exit_code: int,
    command: list[str],
    started_at: float,
    blocker: str | None = None,
    source: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finished_at = time.time()
    return {
        "status": status,
        "exit_code": exit_code,
        "blocker": blocker,
        "command": command,
        "seed": SEED,
        "protocol": "s0-text-dev-v1",
        "config": config,
        "primitive_scope": {"choice_only": True, "score_covered": False, "noul_covered": False},
        "supersedes": {
            "source_data_version": DATA_VERSION,
            "superseded_for_training": status == "passed",
        },
        "training_admitted": False,
        "quality_gate": "not_run",
        "gpu_used": False,
        "holdout_accessed": False,
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
    dev_out: Path,
    train_out: Path,
    dev_ledger_out: Path,
    allocation_ledger_out: Path,
    report_out: Path,
    expected_source_sha256: str,
    expected_count: int,
    expected_raw_source_sha256: str,
    dev_count: int,
    command: list[str],
) -> int:
    started_at = time.time()
    report: dict[str, Any]
    output_paths = [dev_out, train_out, dev_ledger_out, allocation_ledger_out]
    if report_out.exists() or report_out.is_symlink():
        return 1
    if report_out.resolve(strict=False) in {path.resolve(strict=False) for path in output_paths}:
        return 1
    try:
        validate_split_bounds(expected_count=expected_count, dev_count=dev_count)
        validate_output_paths([*output_paths, report_out])
        actual_source_sha256 = sha256_file(source_path)
        if actual_source_sha256 != expected_source_sha256:
            raise TextDevError("unexpected admitted JSONL sha256")
        samples = load_samples(source_path)
        validation = validate_source_samples(
            samples,
            expected_count=expected_count,
            expected_raw_source_sha256=expected_raw_source_sha256,
        )
        dev_samples, train_samples = split_samples(samples, dev_count=dev_count)
        dev_families = {sample.meta.family_id for sample in dev_samples}
        train_families = {sample.meta.family_id for sample in train_samples}
        overlap = dev_families & train_families
        if overlap:
            raise TextDevError("dev/train family overlap")
        dev_ledger = dev_ledger_rows(dev_samples)
        allocation_ledger = allocation_rows(dev_samples, train_samples)
        dev_hash = write_jsonl(dev_out, dev_samples)
        train_hash = write_jsonl(train_out, train_samples)
        dev_ledger_hash = write_jsonl(dev_ledger_out, dev_ledger)
        allocation_hash = write_jsonl(allocation_ledger_out, allocation_ledger)
        source = {
            "admitted_jsonl_sha256": actual_source_sha256,
            "expected_admitted_jsonl_sha256": expected_source_sha256,
            "raw_train_sha256": expected_raw_source_sha256,
            "rows": len(samples),
            "source_dataset": REPO_ID,
            "source_revision": REVISION,
            "license": LICENSE,
            **validation,
        }
        split = {
            "dev_items": len(dev_samples),
            "train_items": len(train_samples),
            "dev_train_family_overlap": 0,
            "expected_train_items": expected_count - dev_count,
            "intent_display_counts_by_pool": {
                "dev": intent_display_counts(dev_samples),
                "train": intent_display_counts(train_samples),
            },
            "intent_display_coverage_by_pool": {
                "dev": len(intent_display_counts(dev_samples)),
                "train": len(intent_display_counts(train_samples)),
            },
        }
        fixed_config = config_payload(
            expected_source_sha256=expected_source_sha256,
            expected_count=expected_count,
            dev_count=dev_count,
        )
        outputs = {
            "dev_jsonl_sha256": dev_hash,
            "train_jsonl_sha256": train_hash,
            "development_ledger_sha256": dev_ledger_hash,
            "allocation_ledger_sha256": allocation_hash,
        }
        report = build_report(
            status="passed",
            exit_code=0,
            command=command,
            started_at=started_at,
            source=source,
            outputs=outputs,
            split=split,
            config={
                **fixed_config,
                "config_sha256": config_sha256(
                    expected_source_sha256=expected_source_sha256,
                    expected_count=expected_count,
                    dev_count=dev_count,
                ),
            },
        )
        code = 0
    except (OSError, TextDevError, ValueError) as exc:
        report = build_report(
            status="blocked",
            exit_code=1,
            command=command,
            started_at=started_at,
            blocker=str(exc) if isinstance(exc, TextDevError) else type(exc).__name__,
            config={
                **config_payload(
                    expected_source_sha256=expected_source_sha256,
                    expected_count=expected_count,
                    dev_count=dev_count,
                ),
                "config_sha256": config_sha256(
                    expected_source_sha256=expected_source_sha256,
                    expected_count=expected_count,
                    dev_count=dev_count,
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
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--dev-ledger-out", type=Path, required=True)
    parser.add_argument("--allocation-ledger-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=EXPECTED_ADMITTED_SHA256)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_ADMITTED_ROWS)
    parser.add_argument("--expected-raw-source-sha256", default=EXPECTED_TRAIN_SHA256)
    parser.add_argument("--dev-count", type=int, default=DEV_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        source_path=args.source,
        dev_out=args.dev_out,
        train_out=args.train_out,
        dev_ledger_out=args.dev_ledger_out,
        allocation_ledger_out=args.allocation_ledger_out,
        report_out=args.report,
        expected_source_sha256=args.expected_source_sha256,
        expected_count=args.expected_count,
        expected_raw_source_sha256=args.expected_raw_source_sha256,
        dev_count=args.dev_count,
        command=[sys.executable, "-m", "scripts.s0_text_dev", *(argv or sys.argv[1:])],
    )


if __name__ == "__main__":
    raise SystemExit(main())
