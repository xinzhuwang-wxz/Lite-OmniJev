"""Build the first public Clotho-AQA audio development cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from data.schema import Sample
from eval.sealed_gate import DevelopmentRecord

TRAIN_SHA256 = "237c692e21e015a3e9b4391edd716b1387d7eb2d068d7ecdb5a6d6ead7e7a992"
METADATA_SHA256 = "7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5"
DATASET = "Clotho-AQA"
REVISION = "zenodo-6473207"
QA_LICENSE = "MIT"
DATA_VERSION = "clotho-aqa-train-noul-dev-v1"
SEED = 42
DEV_COUNT = 500
MAX_SECONDS = 30.0
ALLOCATION_PREFIX = f"s0-audio-dev-v1:{SEED}:"
FIXED_ARCHIVE_ALIASES = {
    "souffle_me_tallique.wav": "souffle_me.tallique.wav",
    "Geese1 - (Apollonia&#39;s sPA) 18_44 05.10.wav": (
        "Geese1 - (Apollonia&#39;s sPA) 18.44 05.10.wav"
    ),
}


class AudioDevError(ValueError):
    """User-correctable input or output contract failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def allocation_key(family_id: str) -> str:
    return sha256_bytes(f"{ALLOCATION_PREFIX}{family_id}".encode())


def row_sha(rows: list[dict[str, str]]) -> str:
    return sha256_bytes(canonical_json(rows).encode())


def sound_id_from_url(url: str) -> str:
    parsed = urlsplit(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc != "freesound.org" or "sounds" not in parts:
        raise AudioDevError("invalid freesound sound_link")
    index = parts.index("sounds")
    if index + 1 >= len(parts) or not parts[index + 1].isdigit():
        raise AudioDevError("missing freesound sound id")
    return parts[index + 1]


def validate_output_paths(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise AudioDevError("output paths must be unique")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise AudioDevError("refusing to overwrite existing outputs")


def validate_dev_count(*, expected_count: int, dev_count: int) -> None:
    if expected_count <= 1:
        raise AudioDevError("expected_count must leave non-dev families")
    if not 1 <= dev_count <= expected_count - 1:
        raise AudioDevError("dev_count must be between 1 and expected_count - 1")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def load_license_plan(path: Path, *, metadata_sha256: str) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metadata_sha256") != metadata_sha256:
        raise AudioDevError("license plan metadata hash mismatch")
    approved = payload.get("approved_audio")
    if not isinstance(approved, list):
        raise AudioDevError("license plan missing approved audio")
    result: dict[str, dict[str, str]] = {}
    for raw in approved:
        if not isinstance(raw, dict):
            raise AudioDevError("invalid license plan row")
        name = str(raw.get("file_name", ""))
        license_id = str(raw.get("license", ""))
        if license_id not in {"CC0-1.0", "CC-BY-3.0"}:
            raise AudioDevError("unapproved media license")
        if PurePosixPath(name).name != name or "\\" in name or "\x00" in name:
            raise AudioDevError("unsafe approved audio name")
        if name in result:
            raise AudioDevError("duplicate license plan file_name")
        license_url = str(raw.get("license_url", ""))
        original_url = str(raw.get("original_url", ""))
        attribution = str(raw.get("attribution", "")).strip()
        if not license_url or not original_url or not attribution:
            raise AudioDevError("license plan missing attribution or URLs")
        result[name] = {
            "file_name": name,
            "license": license_id,
            "license_url": license_url,
            "original_url": original_url,
            "attribution": attribution,
        }
    return result


def validate_audio_preparation(path: Path, *, metadata_sha256: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AudioDevError("audio preparation report must be an object")
    payload: dict[str, Any] = raw
    if payload.get("metadata_sha256") != metadata_sha256:
        raise AudioDevError("audio preparation metadata hash mismatch")
    if payload.get("status") != "approved_audio_extracted_source_audit_only":
        raise AudioDevError("audio preparation report is not approved")
    aliases = payload.get("archive_name_aliases", [])
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list):
        raise AudioDevError("audio preparation aliases invalid")
    alias_map: dict[str, str] = {}
    for raw in aliases:
        if not isinstance(raw, dict):
            raise AudioDevError("audio preparation aliases invalid")
        metadata_name = str(raw.get("metadata_file_name", ""))
        archive_name = str(raw.get("archive_member_name", ""))
        if FIXED_ARCHIVE_ALIASES.get(metadata_name) != archive_name:
            raise AudioDevError("unexpected audio filename alias")
        alias_map[metadata_name] = archive_name
    payload["fixed_alias_map"] = alias_map
    return payload


def load_metadata(
    path: Path, *, metadata_sha256: str, approved: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    if sha256_file(path) != metadata_sha256:
        raise AudioDevError("metadata csv sha256 mismatch")
    rows = read_csv_rows(path)
    result: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for row in rows:
        name = row.get("file_name", "")
        if name in seen:
            raise AudioDevError("duplicate metadata file_name")
        seen.add(name)
        if name not in approved:
            continue
        sound_id = sound_id_from_url(row.get("sound_link", ""))
        planned = approved[name]
        if row.get("license", "") != planned["license_url"]:
            raise AudioDevError("license plan disagrees with metadata")
        if row.get("sound_link", "") != planned["original_url"]:
            raise AudioDevError("license plan disagrees with metadata sound URL")
        if row.get("manufacturer", "").strip() != planned["attribution"]:
            raise AudioDevError("license plan disagrees with metadata attribution")
        result[name] = {**planned, "sound_id": sound_id}
    if set(result) != set(approved):
        raise AudioDevError("approved metadata rows missing")
    return result


def normalize_answer(value: str) -> str:
    lowered = value.strip().casefold()
    if lowered not in {"yes", "no"}:
        raise AudioDevError("answer is not yes/no")
    return lowered


def load_train_groups(
    path: Path, *, train_sha256: str
) -> dict[tuple[str, str], list[dict[str, str]]]:
    if sha256_file(path) != train_sha256:
        raise AudioDevError("train csv sha256 mismatch")
    rows = read_csv_rows(path)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen_columns = {"file_name", "QuestionText", "answer", "confidence"}
    for row in rows:
        if not seen_columns.issubset(row):
            raise AudioDevError("train csv missing required columns")
        name = row["file_name"]
        question = row["QuestionText"].strip()
        if PurePosixPath(name).name != name or not name.endswith(".wav") or not question:
            raise AudioDevError("invalid train row")
        groups[(name, question)].append(
            {
                "file_name": name,
                "QuestionText": question,
                "answer": row["answer"].strip(),
                "confidence": row["confidence"].strip(),
            }
        )
    return groups


def safe_media_path(root: Path, file_name: str, aliases: dict[str, str]) -> tuple[Path, str]:
    if PurePosixPath(file_name).name != file_name or "\\" in file_name or "\x00" in file_name:
        raise AudioDevError("unsafe media file name")
    if root.is_symlink() or root.absolute() != root.resolve() or not root.is_dir():
        raise AudioDevError("asset root must be a real directory")
    names = [file_name]
    alias = aliases.get(file_name)
    if alias is not None:
        names.append(alias)
    for media_name in names:
        if (
            PurePosixPath(media_name).name != media_name
            or "\\" in media_name
            or "\x00" in media_name
        ):
            raise AudioDevError("unsafe media file name")
        path = root
        for part in Path(media_name).parts:
            path = path / part
            if path.is_symlink():
                raise AudioDevError("media symlinks are forbidden")
        if path.exists() and path.is_file() and path.resolve().is_relative_to(root.resolve()):
            return path, media_name
    raise AudioDevError("approved media file missing")


def wav_header(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            compression = audio.getcomptype()
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioDevError("wav header invalid") from exc
    if rate <= 0 or frames <= 0 or channels <= 0 or sample_width <= 0 or compression != "NONE":
        raise AudioDevError("wav header invalid")
    return {
        "duration_seconds": frames / rate,
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
    }


def decode_wav_full(path: Path) -> dict[str, Any]:
    info = wav_header(path)
    remaining = int(info["frames"])
    frame_width = int(info["channels"]) * int(info["sample_width_bytes"])
    decoded = 0
    pcm = hashlib.sha256()
    try:
        with wave.open(str(path), "rb") as audio:
            while remaining:
                requested = min(remaining, 65536)
                data = audio.readframes(requested)
                if not data or len(data) % frame_width:
                    raise AudioDevError("wav decode ended early")
                actual = len(data) // frame_width
                if actual != requested:
                    raise AudioDevError("wav decode ended early")
                pcm.update(data)
                decoded += actual
                remaining -= actual
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioDevError("wav decode failed") from exc
    return {
        **info,
        "decoded_frames": decoded,
        "decoded_samples": decoded * int(info["channels"]),
        "decoded_pcm_sha256": pcm.hexdigest(),
    }


def candidate_rows(
    *,
    groups: dict[tuple[str, str], list[dict[str, str]]],
    metadata: dict[str, dict[str, str]],
    audio_root: Path,
    aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]]]:
    train_files = {file_name for file_name, _question in groups if file_name in metadata}
    header_by_file: dict[str, dict[str, Any]] = {}
    media_path_by_file: dict[str, Path] = {}
    media_name_by_file: dict[str, str] = {}
    for file_name in sorted(train_files):
        path, media_name = safe_media_path(audio_root, file_name, aliases)
        header = wav_header(path)
        header_by_file[file_name] = {
            **header,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        media_path_by_file[file_name] = path
        media_name_by_file[file_name] = media_name

    candidates: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for (file_name, question), rows in groups.items():
        if file_name not in metadata:
            exclusions["unapproved_media_license"] += 1
            continue
        if len(rows) == 6:
            exclusions["six_row_group_excluded"] += 1
            continue
        if len(rows) != 3:
            exclusions["wrong_vote_count"] += 1
            continue
        try:
            answers = [normalize_answer(row["answer"]) for row in rows]
        except AudioDevError:
            exclusions["non_yes_no_answer"] += 1
            continue
        if len(set(answers)) != 1:
            exclusions["non_unanimous_answer"] += 1
            continue
        header = header_by_file[file_name]
        if float(header["duration_seconds"]) > MAX_SECONDS:
            exclusions["duration_over_30s"] += 1
            continue
        meta = metadata[file_name]
        candidates.append(
            {
                "file_name": file_name,
                "question": question,
                "answer": answers[0],
                "raw_rows": rows,
                "sound_id": meta["sound_id"],
                "media": meta,
                "header": header,
                "media_path": media_path_by_file[file_name],
                "media_file_name": media_name_by_file[file_name],
            }
        )
    return candidates, dict(sorted(exclusions.items())), header_by_file


def choose_dev(candidates: list[dict[str, Any]], *, dev_count: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_family[str(candidate["sound_id"])].append(candidate)
    if len(by_family) <= dev_count:
        raise AudioDevError("not enough eligible sound families")
    selected_families = sorted(by_family, key=lambda family: (allocation_key(family), family))[
        :dev_count
    ]
    selected: list[dict[str, Any]] = []
    for family in selected_families:
        rows = sorted(
            by_family[family],
            key=lambda item: (sha256_bytes(str(item["question"]).encode()), str(item["question"])),
        )
        selected.append(rows[0])
    return selected


def source_hash_for(candidate: dict[str, Any]) -> str:
    payload = {
        "rows_sha256": row_sha(candidate["raw_rows"]),
        "media_sha256": candidate["header"]["sha256"],
        "original_file_name": candidate["file_name"],
        "media_file_name": candidate["media_file_name"],
        "question_sha256": sha256_bytes(str(candidate["question"]).encode()),
    }
    return sha256_bytes(canonical_json(payload).encode())


def sample_for(candidate: dict[str, Any]) -> Sample:
    source_hash = source_hash_for(candidate)
    license_value = f"QA:{QA_LICENSE};media:{candidate['media']['license']}"
    return Sample.model_validate(
        {
            "id": f"clotho-aqa-train-{candidate['sound_id']}-{source_hash[:12]}",
            "prefix": {
                "state": "Answer the yes/no question using the attached audio.",
                "media": [
                    {
                        "type": "audio",
                        "uri": str(candidate["media_file_name"]),
                        "sha256": candidate["header"]["sha256"],
                    }
                ],
            },
            "questions": [
                {
                    "qid": "noul",
                    "type": "noul",
                    "instruction": str(candidate["question"]),
                    "target": {
                        "kind": "hard",
                        "value": str(candidate["answer"]),
                        "source": "clotho_original_unanimous_three_answers",
                    },
                }
            ],
            "meta": {
                "modality": ["audio"],
                "source_dataset": DATASET,
                "source_revision": REVISION,
                "source_sha256": source_hash,
                "license": license_value,
                "usage": "commercial",
                "pool": "dev",
                "family_id": str(candidate["sound_id"]),
                "data_version": DATA_VERSION,
            },
        }
    )


def development_record_for(sample: Sample) -> DevelopmentRecord:
    return DevelopmentRecord(
        item_id=sample.id,
        modality="audio",
        family_hash=sha256_bytes(sample.meta.family_id.encode()),
        source_hash=sample.meta.source_sha256,
        source_dataset=sample.meta.source_dataset,
        source_revision=sample.meta.source_revision,
        license=sample.meta.license,
        label_origin="hard_gold",
    )


def media_manifest_row(candidate: dict[str, Any], sample: Sample) -> dict[str, Any]:
    header = candidate["header"]
    media = candidate["media"]
    return {
        "item_id": sample.id,
        "original_file_name": candidate["file_name"],
        "media_file_name": candidate["media_file_name"],
        "sound_id": candidate["sound_id"],
        "sha256": header["sha256"],
        "duration_seconds": header["duration_seconds"],
        "sample_rate": header["sample_rate"],
        "channels": header["channels"],
        "bytes": header["bytes"],
        "decoded_frames": header.get("decoded_frames"),
        "decoded_samples": header.get("decoded_samples"),
        "decoded_pcm_sha256": header.get("decoded_pcm_sha256"),
        "license": media["license"],
        "license_url": media["license_url"],
        "original_url": media["original_url"],
        "attribution": media["attribution"],
    }


def exclusion_ledger_rows(
    *,
    groups: dict[tuple[str, str], list[dict[str, str]]],
    metadata: dict[str, dict[str, str]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_families = {str(item["sound_id"]) for item in selected}
    by_family: dict[str, dict[str, Any]] = {}
    for (file_name, question), rows in groups.items():
        if file_name not in metadata:
            continue
        family = metadata[file_name]["sound_id"]
        entry = by_family.setdefault(
            family,
            {
                "family_id": family,
                "pool": "dev_excluded" if family in selected_families else "train_available",
                "file_names": set(),
                "qa_group_count": 0,
                "row_count": 0,
                "question_hashes": [],
            },
        )
        entry["file_names"].add(file_name)
        entry["qa_group_count"] += 1
        entry["row_count"] += len(rows)
        entry["question_hashes"].append(sha256_bytes(question.encode()))
    result: list[dict[str, Any]] = []
    for entry in by_family.values():
        result.append(
            {
                "family_id": entry["family_id"],
                "pool": entry["pool"],
                "file_names": sorted(entry["file_names"]),
                "qa_group_count": entry["qa_group_count"],
                "row_count": entry["row_count"],
                "question_hashes_sha256": sha256_bytes(
                    canonical_json(sorted(entry["question_hashes"])).encode()
                ),
            }
        )
    return sorted(result, key=lambda row: str(row["family_id"]))


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


def config_payload(*, dev_count: int, train_sha256: str, metadata_sha256: str) -> dict[str, Any]:
    return {
        "protocol": "s0-audio-dev-v1",
        "seed": SEED,
        "dev_count": dev_count,
        "max_seconds": MAX_SECONDS,
        "expected_train_sha256": train_sha256,
        "expected_metadata_sha256": metadata_sha256,
        "dataset": DATASET,
        "revision": REVISION,
        "data_version": DATA_VERSION,
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
    outputs: dict[str, str] | None = None,
    config: dict[str, Any],
) -> dict[str, Any]:
    finished_at = time.time()
    return {
        "status": status,
        "exit_code": exit_code,
        "blocker": blocker,
        "command": command,
        "protocol": "s0-audio-dev-v1",
        "config": {**config, "config_sha256": sha256_bytes(canonical_json(config).encode())},
        "primitive_scope": {"noul_only": True, "choice_covered": False, "score_covered": False},
        "training_admitted": False,
        "quality_gate": "not_run",
        "gpu_used": False,
        "holdout_accessed": False,
        "cost": None,
        "real_human_kappa_verified": False,
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
    train_csv: Path,
    metadata_csv: Path,
    audio_root: Path,
    license_plan: Path,
    audio_preparation_report: Path,
    dev_out: Path,
    dev_ledger_out: Path,
    exclusion_ledger_out: Path,
    media_manifest_out: Path,
    report_out: Path,
    train_sha256: str,
    metadata_sha256: str,
    dev_count: int,
    command: list[str],
) -> int:
    started_at = time.time()
    config = config_payload(
        dev_count=dev_count, train_sha256=train_sha256, metadata_sha256=metadata_sha256
    )
    output_paths = [dev_out, dev_ledger_out, exclusion_ledger_out, media_manifest_out]
    if report_out.exists() or report_out.is_symlink():
        return 1
    if report_out.resolve(strict=False) in {path.resolve(strict=False) for path in output_paths}:
        return 1
    try:
        validate_output_paths([*output_paths, report_out])
        preparation = validate_audio_preparation(
            audio_preparation_report, metadata_sha256=metadata_sha256
        )
        aliases = preparation["fixed_alias_map"]
        approved = load_license_plan(license_plan, metadata_sha256=metadata_sha256)
        metadata = load_metadata(metadata_csv, metadata_sha256=metadata_sha256, approved=approved)
        groups = load_train_groups(train_csv, train_sha256=train_sha256)
        candidates, exclusions, header_by_file = candidate_rows(
            groups=groups, metadata=metadata, audio_root=audio_root, aliases=aliases
        )
        family_count = len({candidate["sound_id"] for candidate in candidates})
        validate_dev_count(expected_count=family_count, dev_count=dev_count)
        selected = choose_dev(candidates, dev_count=dev_count)
        samples: list[Sample] = []
        media_rows: list[dict[str, Any]] = []
        for candidate in selected:
            candidate["header"] = {
                **candidate["header"],
                **decode_wav_full(candidate["media_path"]),
            }
            sample = sample_for(candidate)
            samples.append(sample)
            media_rows.append(media_manifest_row(candidate, sample))
        development = [development_record_for(sample) for sample in samples]
        exclusion = exclusion_ledger_rows(groups=groups, metadata=metadata, selected=selected)
        dev_hash = write_jsonl(dev_out, samples)
        dev_ledger_hash = write_jsonl(dev_ledger_out, development)
        exclusion_hash = write_jsonl(exclusion_ledger_out, exclusion)
        media_hash = write_jsonl(media_manifest_out, media_rows)
        outputs = {
            "dev_jsonl_sha256": dev_hash,
            "development_ledger_sha256": dev_ledger_hash,
            "exclusion_ledger_sha256": exclusion_hash,
            "media_manifest_sha256": media_hash,
        }
        split = {
            "dev_items": len(samples),
            "candidate_qa": len(candidates),
            "candidate_sound_families": family_count,
            "excluded_group_counts": exclusions,
            "selected_sound_families": len({sample.meta.family_id for sample in samples}),
            "all_train_sound_families": len({row["family_id"] for row in exclusion}),
            "duration_filter_seconds": MAX_SECONDS,
        }
        source = {
            "train_csv_sha256": train_sha256,
            "metadata_csv_sha256": metadata_sha256,
            "license_plan": "clotho-aqa-license-plan.json",
            "audio_preparation_report": "clotho-aqa-audio-preparation-002.json",
            "approved_audio_count": len(approved),
            "header_checked_train_audio_count": len(header_by_file),
            "fixed_alias_count": len(aliases),
            "dataset": DATASET,
            "revision": REVISION,
            "qa_license": QA_LICENSE,
        }
        report = build_report(
            status="passed",
            exit_code=0,
            command=command,
            started_at=started_at,
            source=source,
            split=split,
            outputs=outputs,
            config=config,
        )
        code = 0
    except (OSError, AudioDevError, ValueError) as exc:
        report = build_report(
            status="blocked",
            exit_code=1,
            command=command,
            started_at=started_at,
            blocker=str(exc) if isinstance(exc, AudioDevError) else type(exc).__name__,
            config=config,
        )
        code = 1
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--license-plan", type=Path, required=True)
    parser.add_argument("--audio-preparation-report", type=Path, required=True)
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--dev-ledger-out", type=Path, required=True)
    parser.add_argument("--exclusion-ledger-out", type=Path, required=True)
    parser.add_argument("--media-manifest-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", default=TRAIN_SHA256)
    parser.add_argument("--expected-metadata-sha256", default=METADATA_SHA256)
    parser.add_argument("--dev-count", type=int, default=DEV_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        train_csv=args.train_csv,
        metadata_csv=args.metadata_csv,
        audio_root=args.audio_root,
        license_plan=args.license_plan,
        audio_preparation_report=args.audio_preparation_report,
        dev_out=args.dev_out,
        dev_ledger_out=args.dev_ledger_out,
        exclusion_ledger_out=args.exclusion_ledger_out,
        media_manifest_out=args.media_manifest_out,
        report_out=args.report,
        train_sha256=args.expected_train_sha256,
        metadata_sha256=args.expected_metadata_sha256,
        dev_count=args.dev_count,
        command=[sys.executable, "-m", "scripts.s0_audio_dev", *(argv or sys.argv[1:])],
    )


if __name__ == "__main__":
    raise SystemExit(main())
