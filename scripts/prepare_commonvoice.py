"""CPU-only Common Voice archive readiness check.

This command never downloads data, never reads credentials, and never admits this
source for training. It verifies one already-authorized archive by hash, audits
every tar member before any extraction, records metadata schema/counts without
printing transcript or speaker fields, and fully decodes a bounded deterministic
audio sample on CPU.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.assets import write_json

EXPECTED_ARCHIVE_BYTES = 544_267_165
EXPECTED_ARCHIVE_SHA256 = "390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6"
CPU_LOCK = Path("/tmp/omnijev-cpu-heavy.lock")
MAX_MEMBERS = 500_000
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_EXPANDED_BYTES = 2 * 1024**3
MAX_SAMPLE_BYTES = 128 * 1024**2
MAX_DECODE_SAMPLES = 32
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}
METADATA_SUFFIXES = {".tsv", ".csv"}
REDACTED_METADATA_FIELDS = {
    "accent",
    "age",
    "client_id",
    "down_votes",
    "gender",
    "locale",
    "path",
    "segment",
    "sentence",
    "speaker",
    "speaker_id",
    "transcript",
    "transcription",
    "up_votes",
    "variant",
}

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def cpu_lock(path: Path = CPU_LOCK) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical_member_name(member: tarfile.TarInfo) -> str:
    name = member.name.rstrip("/")
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in name
        or ":" in name
        or "\x00" in name
        or str(path) != name
    ):
        raise ValueError("unsafe_tar_path")
    return name


def audit_archive(
    archive: Path,
    *,
    max_members: int = MAX_MEMBERS,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_expanded_bytes: int = MAX_EXPANDED_BYTES,
) -> dict[str, Any]:
    names: set[str] = set()
    member_count = 0
    expanded = 0
    audio_names: list[str] = []
    metadata_names: list[str] = []
    suffixes: Counter[str] = Counter()
    top_dirs: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    largest_member = 0
    largest_member_name_hash = None
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = canonical_member_name(member)
            if name in names:
                raise ValueError("duplicate_tar_path")
            names.add(name)
            member_count += 1
            if member_count > max_members:
                raise ValueError("tar_member_count_limit")
            if member.isdir():
                kinds["directory"] += 1
                continue
            if not member.isfile():
                raise ValueError("tar_special_file_denied")
            if member.islnk() or member.issym():
                raise ValueError("tar_link_denied")
            if member.size < 0 or member.size > max_member_bytes:
                raise ValueError("tar_member_size_limit")
            expanded += member.size
            if expanded > max_expanded_bytes:
                raise ValueError("tar_total_expansion_limit")
            suffix = PurePosixPath(name).suffix.lower()
            suffixes[suffix or "<none>"] += 1
            top_dirs[PurePosixPath(name).parts[0]] += 1
            kinds["file"] += 1
            if member.size > largest_member:
                largest_member = member.size
                largest_member_name_hash = hashlib.sha256(name.encode()).hexdigest()
            if suffix in AUDIO_SUFFIXES:
                audio_names.append(name)
            elif suffix in METADATA_SUFFIXES:
                metadata_names.append(name)
    if not audio_names:
        raise ValueError("commonvoice_audio_missing")
    if not metadata_names:
        raise ValueError("commonvoice_metadata_missing")
    return {
        "members": member_count,
        "files": kinds["file"],
        "directories": kinds["directory"],
        "expanded_bytes": expanded,
        "audio_members": len(audio_names),
        "metadata_members": len(metadata_names),
        "audio_names": audio_names,
        "metadata_names": metadata_names,
        "suffix_counts": dict(sorted(suffixes.items())),
        "top_level_counts": dict(sorted(top_dirs.items())),
        "largest_member_bytes": largest_member,
        "largest_member_name_sha256": largest_member_name_hash,
        "limits": {
            "max_members": max_members,
            "max_member_bytes": max_member_bytes,
            "max_expanded_bytes": max_expanded_bytes,
        },
    }


def selected_samples(audio_names: list[str], limit: int = MAX_DECODE_SAMPLES) -> list[str]:
    names = sorted(audio_names)
    if len(names) <= limit:
        return names
    stride = max(1, len(names) // limit)
    return names[::stride][:limit]


def tar_members_by_name(archive: Path, wanted: set[str]) -> dict[str, tarfile.TarInfo]:
    found: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = canonical_member_name(member)
            if name in wanted:
                if not member.isfile():
                    raise ValueError("selected_member_not_file")
                found[name] = member
            if len(found) == len(wanted):
                break
    if set(found) != wanted:
        raise ValueError("selected_member_missing")
    return found


def metadata_schema(archive: Path, metadata_names: list[str], output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=False)
    wanted = set(metadata_names)
    schemas: list[dict[str, Any]] = []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = canonical_member_name(member)
            if name not in wanted:
                continue
            if member.size > 64 * 1024**2:
                raise ValueError("metadata_member_size_limit")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("metadata_member_missing")
            header_line = stream.readline(1024 * 1024 + 1)
            if len(header_line) > 1024 * 1024:
                raise ValueError("metadata_header_size_limit")
            header = header_line.decode("utf-8-sig").rstrip("\r\n").split("\t")
            if len(header) == 1 and "," in header[0]:
                header = header[0].split(",")
            if not header or any(not field or "\x00" in field for field in header):
                raise ValueError("metadata_header_invalid")
            rows = sum(1 for _line in stream)
            redacted = sorted(set(header) & REDACTED_METADATA_FIELDS)
            public_columns = [
                field
                if field not in REDACTED_METADATA_FIELDS
                else "redacted:" + hashlib.sha256(field.encode()).hexdigest()[:16]
                for field in header
            ]
            schema = {
                "member_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "columns": public_columns,
                "redacted_column_count": len(redacted),
                "redacted_column_name_sha256": [
                    hashlib.sha256(field.encode()).hexdigest() for field in redacted
                ],
                "row_count": rows,
                "bytes": member.size,
            }
            schema_name = hashlib.sha256(name.encode()).hexdigest()[:16] + ".schema.json"
            (output / schema_name).write_text(json.dumps(schema, indent=2) + "\n")
            schemas.append(schema)
    if len(schemas) != len(metadata_names):
        raise ValueError("metadata_schema_incomplete")
    return schemas


def extract_selected_audio(archive: Path, selected: list[str], output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    wanted = set(selected)
    extracted: list[Path] = []
    total = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = canonical_member_name(member)
            if name not in wanted:
                continue
            if not member.isfile() or member.size > MAX_SAMPLE_BYTES:
                raise ValueError("selected_audio_size_limit")
            total += member.size
            if total > MAX_SAMPLE_BYTES:
                raise ValueError("sample_temp_space_limit")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("selected_audio_missing")
            target = output / f"{len(extracted):06d}{PurePosixPath(name).suffix.lower()}"
            with target.open("xb") as destination:
                shutil.copyfileobj(stream, destination, length=1024**2)
            if target.stat().st_size != member.size:
                raise ValueError("selected_audio_size_mismatch")
            extracted.append(target)
            if len(extracted) == len(selected):
                break
    if len(extracted) != len(selected):
        raise ValueError("selected_audio_incomplete")
    return extracted


def probe_audio(path: Path) -> dict[str, Any]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    stream = next(
        (item for item in metadata["streams"] if item.get("codec_type") == "audio"),
        None,
    )
    if stream is None:
        raise ValueError("audio_stream_missing")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-threads",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=180,
        check=True,
    )
    duration = float(metadata["format"].get("duration", 0))
    sample_rate = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)
    if duration <= 0 or sample_rate <= 0 or channels <= 0:
        raise ValueError("audio_metadata_invalid")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration,
        "bytes": path.stat().st_size,
        "decoder": "ffmpeg",
    }


def aggregate_audio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["duration_seconds"]) for row in rows]
    return {
        "decoded": len(rows),
        "codecs": dict(Counter(str(row["codec"]) for row in rows)),
        "sample_rates": dict(Counter(str(row["sample_rate"]) for row in rows)),
        "channels": dict(Counter(str(row["channels"]) for row in rows)),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "total": sum(durations),
        },
        "bytes": {
            "min": min(row["bytes"] for row in rows) if rows else None,
            "max": max(row["bytes"] for row in rows) if rows else None,
            "total": sum(row["bytes"] for row in rows),
        },
        "decode_scope": "complete selected audio streams through EOF using ffmpeg -f null",
    }


def validate_commonvoice(archive: Path, output_dir: Path) -> dict[str, Any]:
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("archive_size_mismatch")
    archive_sha = sha256_path(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("archive_sha256_mismatch")
    audit = audit_archive(archive)
    selected = selected_samples(audit["audio_names"])
    sample_name_hashes = [hashlib.sha256(name.encode()).hexdigest() for name in selected]
    if output_dir.exists():
        raise ValueError("output_dir_exists")
    output_dir.mkdir(parents=True)
    metadata_dir = output_dir / "metadata-schema"
    schemas = metadata_schema(archive, audit["metadata_names"], metadata_dir)
    with tempfile.TemporaryDirectory(prefix="commonvoice-audio-", dir=output_dir) as temporary:
        audio_paths = extract_selected_audio(archive, selected, Path(temporary))
        decoded = [probe_audio(path) for path in audio_paths]
    public_audit = {key: value for key, value in audit.items() if not key.endswith("_names")}
    return {
        "status": "commonvoice_archive_cpu_ready",
        "archive": {
            "bytes": archive.stat().st_size,
            "sha256": archive_sha,
            "source": "Mozilla Data Collective Common Voice Spontaneous Speech 5.0 English",
        },
        "safety": {
            **public_audit,
            "scope": "all tar members streamed and validated before selected extraction",
            "rejected_types": (
                "absolute/up-level paths, links, special files, duplicate canonical names, "
                "oversize members, oversize total expansion"
            ),
        },
        "metadata": {
            "schema_files_dir": str(metadata_dir),
            "files": len(schemas),
            "total_rows": sum(item["row_count"] for item in schemas),
            "schemas": schemas,
            "raw_transcripts_or_speaker_fields_in_report": False,
        },
        "audio_decode": {
            "sample_limit": MAX_DECODE_SAMPLES,
            "selected": len(selected),
            "selection": "sorted audio member names, fixed stride, numeric temp filenames",
            "selected_member_name_sha256": sample_name_hashes,
            **aggregate_audio(decoded),
        },
        "sample_accounting_note": (
            "audio_members is a physical archive media-file count. The current MDC datasheet "
            "row count is 5898 clips; this report is source inventory only and must not treat "
            "physical file count as independent training samples. Training requires later "
            "audio_id and source-family deduplication plus split isolation."
        ),
        "training_admitted": False,
        "source_only": True,
        "gpu_seconds": 0,
        "network_used": False,
        "credentials_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "script_sha256": sha256_path(Path(__file__)),
        "cpu_lock": str(CPU_LOCK),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "training_admitted": False,
        "source_only": True,
        "gpu_seconds": 0,
        "network_used": False,
    }
    try:
        with cpu_lock():
            result.update(validate_commonvoice(args.archive, args.output_dir))
            result["exit_code"] = 0
    except Exception as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            reason=str(error)[:300],
        )
    result["elapsed_seconds"] = time.monotonic() - start
    write_json(args.report, result)
    print(json.dumps({"status": result["status"], "exit_code": result["exit_code"]}))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
