"""Prepare the pinned Clotho-AQA CC0/CC-BY audio subset; never admit QA labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from scripts.assets import write_json

METADATA_SHA256 = "7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5"
ARCHIVE_MD5 = "48c2c46ccd71dc06121228e2ed68d6b9"
ARCHIVE_SHA256 = "4793d97781689f8bdb1a6669056386825bb0ec8022f2217acc8083abc4774d66"
ARCHIVE_BYTES = 3140677494
MAX_AUDIO_BYTES = 64 * 1024**2
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_EXPANDED_BYTES = 8 * 1024**3
DECODE_SAMPLE_COUNT = 32

ARCHIVE_NAME_ALIASES = {
    "souffle_me_tallique.wav": "souffle_me.tallique.wav",
    "Geese1 - (Apollonia&#39;s sPA) 18_44 05.10.wav": (
        "Geese1 - (Apollonia&#39;s sPA) 18.44 05.10.wav"
    ),
}


def hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def plan(metadata: Path) -> dict[str, Any]:
    if hashes(metadata)[1] != METADATA_SHA256:
        raise ValueError("metadata_hash_mismatch")
    permitted = {"/publicdomain/zero/1.0/": "CC0-1.0", "/licenses/by/3.0/": "CC-BY-3.0"}
    approved = []
    rejected = 0
    seen = set()
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = row["file_name"]
            if (
                PurePosixPath(name).name != name
                or "\\" in name
                or "\x00" in name
                or not name.endswith(".wav")
                or name in seen
            ):
                raise ValueError("invalid_or_duplicate_audio_name")
            seen.add(name)
            url = urlsplit(row["license"])
            license_id = permitted.get(url.path)
            if (
                url.scheme not in {"http", "https"}
                or url.hostname != "creativecommons.org"
                or url.username
                or url.query
                or url.fragment
                or not license_id
            ):
                rejected += 1
                continue
            approved.append(
                {
                    "file_name": name,
                    "license": license_id,
                    "license_url": row["license"],
                    "original_url": row["sound_link"],
                    "attribution": row["manufacturer"],
                }
            )
    if not approved:
        raise ValueError("no_approved_audio")
    return {
        "source": "https://zenodo.org/records/6473207",
        "metadata_sha256": METADATA_SHA256,
        "approved_audio": approved,
        "excluded_audio_count": rejected,
        "training_admitted": False,
        "private_holdout_assigned": False,
    }


def extract(archive: Path, metadata: Path, output: Path) -> dict[str, Any]:
    manifest = plan(metadata)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("archive_size_mismatch")
    md5, sha = hashes(archive)
    if md5 != ARCHIVE_MD5 or sha != ARCHIVE_SHA256:
        raise ValueError("archive_checksum_mismatch")
    wanted = {entry["file_name"]: entry for entry in manifest["approved_audio"]}
    alias_targets = {
        canonical: alias for canonical, alias in ARCHIVE_NAME_ALIASES.items() if canonical in wanted
    }
    if len(set(alias_targets.values())) != len(alias_targets):
        raise ValueError("archive_alias_conflict")
    wanted_by_archive = {alias_targets.get(name, name): name for name in wanted}
    if len(wanted_by_archive) != len(wanted):
        raise ValueError("archive_alias_conflict")
    selected = {}
    archive_member_count = 0
    archive_expanded_bytes = 0
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            archive_member_count += 1
            if archive_member_count > MAX_ARCHIVE_MEMBERS:
                raise ValueError("zip_member_count_limit")
            path = PurePosixPath(entry.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.filename
                or "\x00" in entry.filename
            ):
                raise ValueError("unsafe_zip_path")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir():
                continue
            if mode not in {0, stat.S_IFREG}:
                raise ValueError("zip_special_file_denied")
            if entry.file_size > MAX_AUDIO_BYTES:
                raise ValueError("zip_member_size_limit")
            archive_expanded_bytes += entry.file_size
            if archive_expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                raise ValueError("zip_total_expansion_limit")
            if path.name not in wanted_by_archive:
                continue
            canonical_name = wanted_by_archive[path.name]
            if canonical_name in selected:
                raise ValueError("invalid_or_duplicate_zip_audio")
            selected[canonical_name] = entry
        if set(selected) != set(wanted):
            raise ValueError("approved_audio_missing")
        required = sum(entry.file_size for entry in selected.values())
        if required > 8 * 1024**3:
            raise ValueError("unexpected_audio_expansion")
        output.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output).free < required + 128 * 1024**2:
            raise ValueError("insufficient_audio_disk")
        for name, entry in selected.items():
            target = output / name
            if target.is_symlink():
                raise ValueError("existing_audio_preserved")
            if target.exists():
                expected = hashlib.sha256()
                with zipped.open(entry) as source:
                    for block in iter(lambda: source.read(1024**2), b""):
                        expected.update(block)
                actual = hashes(target)[1]
                if target.stat().st_size != entry.file_size or actual != expected.hexdigest():
                    raise ValueError("existing_audio_preserved")
                wanted[name]["sha256"] = actual
                wanted[name]["archive_member"] = entry.filename
                continue
            temporary = target.with_suffix(".wav.part")
            with zipped.open(entry) as source, temporary.open("xb") as dest:
                shutil.copyfileobj(source, dest, length=1024**2)
            if temporary.stat().st_size != entry.file_size:
                raise ValueError("audio_size_mismatch")
            os.link(temporary, target)
            temporary.unlink()
            wanted[name]["sha256"] = hashes(target)[1]
            wanted[name]["archive_member"] = entry.filename
    decoded = decode_representative_audio(output, sorted(selected))
    manifest_rows = [
        {
            "file_name": entry["file_name"],
            "archive_member": entry.get("archive_member"),
            "license": entry["license"],
            "sha256": entry["sha256"],
        }
        for entry in manifest["approved_audio"]
    ]
    return {
        "status": "approved_audio_extracted_source_audit_only",
        "exit_code": 0,
        "source": manifest["source"],
        "metadata_sha256": manifest["metadata_sha256"],
        "archive_md5": md5,
        "archive_sha256": sha,
        "archive_member_count": archive_member_count,
        "archive_expanded_bytes": archive_expanded_bytes,
        "approved_audio_count": len(manifest["approved_audio"]),
        "excluded_audio_count": manifest["excluded_audio_count"],
        "extracted_audio_count": len(selected),
        "alias_count": len(alias_targets),
        "archive_name_aliases": [
            {
                "metadata_file_name": canonical,
                "archive_member_name": alias,
                "mode": "fixed_revision_alias",
            }
            for canonical, alias in sorted(alias_targets.items())
        ],
        "approved_audio_manifest_sha256": sha256_bytes(
            json.dumps(manifest_rows, sort_keys=True, separators=(",", ":")).encode()
        ),
        "decode": decoded,
        "training_admitted": False,
        "private_holdout_assigned": False,
    }


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
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration_seconds": float(metadata["format"].get("duration", 0)),
        "bytes": path.stat().st_size,
    }


def decode_representative_audio(output: Path, names: list[str]) -> dict[str, Any]:
    selected = (
        names
        if len(names) <= DECODE_SAMPLE_COUNT
        else names[:: max(1, len(names) // DECODE_SAMPLE_COUNT)][:DECODE_SAMPLE_COUNT]
    )
    decoded = [probe_audio(output / name) for name in selected]
    durations = [row["duration_seconds"] for row in decoded]
    return {
        "sample_limit": DECODE_SAMPLE_COUNT,
        "selected": len(selected),
        "selection": "sorted approved canonical filenames, fixed stride",
        "selected_name_sha256": [sha256_bytes(name.encode()) for name in selected],
        "codecs": dict(Counter(str(row["codec"]) for row in decoded)),
        "sample_rates": dict(Counter(str(row["sample_rate"]) for row in decoded)),
        "channels": dict(Counter(str(row["channels"]) for row in decoded)),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "total": sum(durations),
        },
        "bytes": {
            "min": min(row["bytes"] for row in decoded) if decoded else None,
            "max": max(row["bytes"] for row in decoded) if decoded else None,
            "total": sum(row["bytes"] for row in decoded),
        },
        "decode_scope": "complete selected audio streams through EOF using ffmpeg -f null",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "extract"])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = {**plan(args.metadata), "status": "audio_license_plan_ready", "exit_code": 0}
        else:
            if args.archive is None or args.output is None:
                raise ValueError("archive_and_output_required")
            result = extract(args.archive, args.metadata, args.output)
    except Exception as error:
        result = {
            "status": "blocked",
            "exit_code": 1,
            "error_type": type(error).__name__,
            "reason": str(error) if type(error) is ValueError else "audio_preparation_failed",
        }
    write_json(args.report, result)
    print(json.dumps({k: result[k] for k in ["status", "exit_code"]}))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
