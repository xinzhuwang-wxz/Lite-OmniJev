"""Admit the public CLINC train split into the project data contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.schema import Pool, Sample, validate_pool_isolation
from scripts.assets import write_json
from scripts.check_tokens import probe as probe_answer_tokens

REPO_ID = "clinc/clinc_oos"
REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
LICENSE = "CC-BY-3.0"
EXPECTED_ROWS = 15_250
EXPECTED_LABELS = 151
EXPECTED_TRAIN_SHA256 = "30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f"
DATA_VERSION = "clinc-oos-plus-train-v1"
DATA_CARD = {
    "dataset": REPO_ID,
    "config": "plus",
    "split": "train",
    "revision": REVISION,
    "license": LICENSE,
    "license_url": f"https://huggingface.co/datasets/{REPO_ID}/blob/{REVISION}/LICENSE",
    "attribution": (
        "Stefan Larson et al., An Evaluation Dataset for Intent Classification "
        "and Out-of-Scope Prediction, EMNLP 2019."
    ),
    "source_url": f"https://huggingface.co/datasets/{REPO_ID}/tree/{REVISION}",
    "provenance_note": (
        "Public CLINC train parquet only; validation/test/holdout splits are not accessed."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(label: str) -> str:
    return label.replace("_", " ").strip()


def option_group(label_id: int, labels: list[str], group_size: int = 20) -> list[str]:
    if not 0 <= label_id < len(labels):
        raise ValueError(f"intent id out of range: {label_id}")
    start = (label_id // group_size) * group_size
    return [normalize_label(label) for label in labels[start : start + group_size]]


def label_names_from_hf_metadata(metadata: dict[bytes, bytes] | None) -> list[str]:
    if not metadata or b"huggingface" not in metadata:
        raise ValueError("Parquet schema lacks HuggingFace ClassLabel metadata")
    payload = json.loads(metadata[b"huggingface"])
    features = payload.get("info", {}).get("features", {})
    intent = features.get("intent", {})
    names = intent.get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("HuggingFace metadata lacks intent ClassLabel names")
    if len(names) != EXPECTED_LABELS:
        raise ValueError(f"expected {EXPECTED_LABELS} labels, found {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("intent labels must be distinct")
    return names


def read_clinc_parquet(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:  # pragma: no cover - covered by CLI blocker path.
        raise RuntimeError("pyarrow is required; run through infra/data on Linux") from exc

    table = parquet.read_table(path)
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} train rows, found {table.num_rows}")
    if table.column_names != ["text", "intent"]:
        raise ValueError(f"unexpected CLINC columns: {table.column_names}")
    labels = label_names_from_hf_metadata(table.schema.metadata)
    rows = table.to_pylist()
    return rows, labels


def build_samples(
    rows: Iterable[dict[str, Any]],
    labels: list[str],
    *,
    source_sha256: str,
    pool: Pool = "train",
) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    seen_texts: dict[str, int] = {}
    duplicate_texts = 0
    input_rows = 0
    label_counts: Counter[str] = Counter()
    group_sizes: Counter[int] = Counter()

    for index, row in enumerate(rows):
        input_rows += 1
        text = row.get("text")
        intent = row.get("intent")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"row {index} has empty text")
        if type(intent) is not int:
            raise ValueError(f"row {index} has non-integer intent")
        if not 0 <= intent < len(labels):
            raise ValueError(f"row {index} intent id out of range: {intent}")
        normalized_text = " ".join(text.split()).casefold()
        previous_intent = seen_texts.get(normalized_text)
        if previous_intent is not None:
            if previous_intent != intent:
                raise ValueError(
                    "conflicting intents for normalized utterance "
                    f"at row {index}: {previous_intent} != {intent}"
                )
            duplicate_texts += 1
            continue
        seen_texts[normalized_text] = intent

        raw_options = option_group(intent, labels)
        options = list(raw_options)
        if len(options) != len(set(options)):
            raise ValueError("normalized option labels must be distinct")
        target_option = normalize_label(labels[intent])
        target_index = options.index(target_option)
        target_token = chr(65 + target_index)
        family = hashlib.sha256(f"clinc-oos:{normalized_text}".encode()).hexdigest()
        label_counts[labels[intent]] += 1
        group_sizes[len(options)] += 1
        samples.append(
            Sample.model_validate(
                {
                    "id": f"clinc-oos-train-{index:05d}",
                    "prefix": {"state": text.strip()},
                    "questions": [
                        {
                            "qid": "intent",
                            "type": "choice",
                            "instruction": "Which intent best matches the user's utterance?",
                            "options": options,
                            "target": {
                                "kind": "hard",
                                "value": target_token,
                                "source": "clinc_human_label",
                            },
                        }
                    ],
                    "meta": {
                        "modality": ["text"],
                        "source_dataset": REPO_ID,
                        "source_revision": REVISION,
                        "source_sha256": source_sha256,
                        "license": LICENSE,
                        "usage": "commercial",
                        "pool": pool,
                        "family_id": family,
                        "data_version": DATA_VERSION,
                    },
                }
            )
        )

    validate_pool_isolation(samples)
    oos_index = labels.index("oos")
    report = {
        "input_rows": input_rows,
        "samples_written": len(samples),
        "duplicate_exact_texts_dropped": duplicate_texts,
        "labels": len(labels),
        "label_counts": dict(sorted(label_counts.items())),
        "raw_labels_original": labels,
        "option_group_sizes": dict(sorted(group_sizes.items())),
        "candidate_rule": {
            "source": "original HuggingFace ClassLabel order",
            "construction": "contiguous chunks of up to 20 labels",
            "display_normalization": "replace underscores with spaces and strip whitespace",
            "answer_tokens": (
                "A through T by option position; U remains separate insufficient token"
            ),
        },
        "oos_treatment": {
            "label_id": oos_index,
            "original_label": labels[oos_index],
            "mapped_to": "ordinary out-of-scope intent choice option",
            "not_mapped_to": "U",
        },
        "semantic_review": {
            "human_semantic_review_performed": False,
            "note": (
                "No claim of human semantic review; this script preserves CLINC "
                "human labels mechanically."
            ),
        },
    }
    return samples, report


def write_jsonl(path: Path, samples: list[Sample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    corpus = "".join(sample.model_dump_json() + "\n" for sample in samples)
    path.write_text(corpus)
    return hashlib.sha256(corpus.encode()).hexdigest()


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def require_token_probe(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        raise ValueError("Missing required --model-dir for official tokenizer answer-token probe")
    token_report = probe_answer_tokens(model_dir)
    if token_report.get("status") != "passed" or token_report.get("exit_code") != 0:
        failures = token_report.get("failures") or token_report.get("blocker") or "unknown failure"
        raise ValueError(f"official tokenizer answer-token probe failed: {failures}")
    return token_report


def run(
    input_path: Path,
    output: Path,
    report_path: Path,
    *,
    model_dir: Path | None,
    command: list[str] | None = None,
) -> int:
    started = datetime.now(UTC)
    actual_sha = None
    labels: list[str] = []
    dataset_sha = None
    details: dict[str, Any] = {}
    token_report: dict[str, Any] | None = None
    blocker = None
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing CLINC train parquet: {input_path}")
        actual_sha = sha256_file(input_path)
        if actual_sha != EXPECTED_TRAIN_SHA256:
            raise ValueError(f"CLINC train SHA mismatch: {actual_sha}")
        rows, labels = read_clinc_parquet(input_path)
        samples, details = build_samples(rows, labels, source_sha256=actual_sha)
        token_report = require_token_probe(model_dir)
        dataset_sha = write_jsonl(output, samples)
        status = "training_admitted"
        exit_code = 0
    except Exception as exc:
        status = "blocked"
        exit_code = 1
        blocker = str(exc)
    finished = datetime.now(UTC)

    report = {
        "status": status,
        "training_admitted": status == "training_admitted",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "data_card": DATA_CARD,
        "input": str(input_path),
        "input_sha256": actual_sha,
        "expected_input_sha256": EXPECTED_TRAIN_SHA256,
        "output": str(output) if status == "training_admitted" else None,
        "output_sha256": dataset_sha,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "token_probe": {
            "required": True,
            "status": token_report.get("status") if token_report else None,
            "model_dir": token_report.get("model_dir") if token_report else None,
            "template_sha256": token_report.get("template_sha256") if token_report else None,
            "tokenizer_sha256": token_report.get("tokenizer_sha256") if token_report else None,
            "answer_count": token_report.get("answer_count") if token_report else None,
            "scope": token_report.get("scope") if token_report else None,
        },
        "source_provenance": {
            "full_source": f"{REPO_ID}@{REVISION}:plus/train.parquet",
            "source_url": DATA_CARD["source_url"],
            "license_url": DATA_CARD["license_url"],
            "attribution": DATA_CARD["attribution"],
            "raw_label_source": "CLINC HuggingFace ClassLabel metadata",
            "target_source": "clinc_human_label",
        },
        "heldout_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "forbidden_teacher_labels_used": False,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "python": platform.python_version(),
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha256(),
        "command": command or [sys.executable, "-m", "scripts.admit_clinc"],
        "exit_code": exit_code,
        "dirty_worktree": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "details": details,
        "labels_sample": labels[:5],
        "blocker": blocker,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(f"artifacts/data/raw/clinc-clinc_oos/{REVISION}/train.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/admitted/clinc-oos-train-v1.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/clinc-admission.json"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to the fixed E4B official tokenizer files used for answer-token probing.",
    )
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "scripts.admit_clinc", *(argv or sys.argv[1:])]
    return run(args.input, args.output, args.report, model_dir=args.model_dir, command=command)


if __name__ == "__main__":
    raise SystemExit(main())
