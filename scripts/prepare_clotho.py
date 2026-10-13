"""Prepare the pinned Clotho-AQA CC0/CC-BY audio subset; never admit QA labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from scripts.assets import write_json

METADATA_SHA256 = "7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5"
ARCHIVE_MD5 = "48c2c46ccd71dc06121228e2ed68d6b9"
ARCHIVE_BYTES = 3140677494
MAX_AUDIO_BYTES = 64 * 1024**2


def hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


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
    if md5 != ARCHIVE_MD5:
        raise ValueError("archive_checksum_mismatch")
    wanted = {entry["file_name"]: entry for entry in manifest["approved_audio"]}
    selected = {}
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError("unsafe_zip_path")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if mode == stat.S_IFLNK:
                raise ValueError("zip_symlink_denied")
            if entry.is_dir() or path.name not in wanted:
                continue
            if (
                path.name in selected
                or mode not in {0, stat.S_IFREG}
                or entry.file_size > MAX_AUDIO_BYTES
            ):
                raise ValueError("invalid_or_duplicate_zip_audio")
            selected[path.name] = entry
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
                continue
            temporary = target.with_suffix(".wav.part")
            with zipped.open(entry) as source, temporary.open("xb") as dest:
                shutil.copyfileobj(source, dest, length=1024**2)
            if temporary.stat().st_size != entry.file_size:
                raise ValueError("audio_size_mismatch")
            os.link(temporary, target)
            temporary.unlink()
            wanted[name]["sha256"] = hashes(target)[1]
    return {
        **manifest,
        "status": "approved_audio_extracted_source_audit_only",
        "exit_code": 0,
        "archive_md5": md5,
        "archive_sha256": sha,
        "extracted_audio_count": len(selected),
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
