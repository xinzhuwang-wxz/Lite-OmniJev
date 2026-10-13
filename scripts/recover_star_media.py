"""Recover only the twelve missing public MVBench clips from pinned AI2 Charades."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.public_eval_media import checked_zip_members, probe_decode, sha256

SOURCE_URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
SOURCE_BYTES = 16_339_546_533
SOURCE_ETAG = '"d37c91565b08ce1f432a46e11351751e-1948"'
SOURCE_VERSION = "2I0cX8bvsoqLbtsk6mi6Wrb0d0uV1H8E"
STAR_SHA256 = "bdb5f192a98f78e79e51da6ae801f8ba79d1d09c0d8c19b6e63105251cb17585"
PUBLIC_TASK_SHA256 = "d5f8c063a5ae5a433c3fb8ff78a03acbf0513a01928b384ed90a1e044adf230e"


class HTTPRangeFile(io.RawIOBase):
    """Seekable bounded HTTP pages; ZIP parsing stays entirely in stdlib ZipFile."""

    def __init__(
        self,
        url: str,
        length: int,
        etag: str,
        version: str,
        *,
        budget: int = 500_000_000,
        page_size: int = 1024**2,
    ) -> None:
        super().__init__()
        if length <= 0 or not 0 < page_size <= 1024**2 or not 0 <= budget <= 500_000_000:
            raise ValueError("invalid_http_range_bounds")
        self.url, self.length, self.etag, self.version = url, length, etag, version
        self.budget, self.page_size = budget, page_size
        self.position = 0
        self.downloaded = 0
        self.requests = 0
        self.pages: OrderedDict[int, bytes] = OrderedDict()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        target = (
            offset + {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.length}[whence]
        )
        if not 0 <= target <= self.length:
            raise ValueError("range_seek_outside_source")
        self.position = target
        return target

    def _page(self, start: int) -> bytes:
        if start in self.pages:
            self.pages.move_to_end(start)
            return self.pages[start]
        end = min(start + self.page_size, self.length) - 1
        wanted = end - start + 1
        if self.downloaded + wanted + 1 > self.budget:
            raise ValueError("http_range_download_budget_exceeded")
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag}
        )
        self.requests += 1
        with urllib.request.urlopen(request, timeout=60) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.length}"
                or response.headers.get("ETag") != self.etag
                or response.headers.get("x-amz-version-id") != self.version
            ):
                raise ValueError("http_range_response_not_exact_pinned_object")
            data: bytes = response.read(wanted + 1)
        self.downloaded += len(data)
        if len(data) != wanted:
            raise ValueError("http_range_body_length_mismatch")
        self.pages[start] = data
        if len(self.pages) > 8:
            self.pages.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size > 8 * 1024**2:
            raise ValueError("http_range_single_read_limit")
        result = bytearray()
        while len(result) < size:
            page_start = self.position // self.page_size * self.page_size
            page = self._page(page_start)
            offset = self.position - page_start
            block = page[offset : offset + size - len(result)]
            result.extend(block)
            self.position += len(block)
        return bytes(result)


def recover(
    star: Path, metadata: Path, config: Path, output: Path, source: HTTPRangeFile
) -> dict[str, Any]:
    if sha256(star) != STAR_SHA256:
        raise ValueError("original_star_archive_hash_mismatch")
    task = next(
        x
        for x in json.loads(config.read_text())["mvbench"]["tasks"]
        if x["file"] == "json/action_sequence.json"
    )
    if sha256(metadata) != PUBLIC_TASK_SHA256 or task["sha256"] != PUBLIC_TASK_SHA256:
        raise ValueError("public_task_metadata_hash_mismatch")
    rows = json.loads(metadata.read_text())
    with zipfile.ZipFile(star) as archive:
        present = checked_zip_members(archive)
    missing = [
        {
            "row_index": i,
            "video": row["video"],
            "start": row["start"],
            "end": row["end"],
            "expected_member": "star/Charades_v1_480/" + row["video"],
        }
        for i, row in enumerate(rows)
        if "star/Charades_v1_480/" + row["video"] not in present
    ]
    if len(missing) != 12 or len({x["video"] for x in missing}) != 12:
        raise ValueError("unexpected_missing_public_clip_scope")
    if any(not re.fullmatch(r"[A-Z0-9]{5}\.mp4", x["video"]) for x in missing):
        raise ValueError("unexpected_charades_video_id")
    with urllib.request.urlopen(
        urllib.request.Request(SOURCE_URL, method="HEAD"), timeout=30
    ) as head:
        if (
            int(head.headers["Content-Length"]) != SOURCE_BYTES
            or head.headers.get("ETag") != SOURCE_ETAG
            or head.headers.get("x-amz-version-id") != SOURCE_VERSION
        ):
            raise ValueError("primary_source_head_not_pinned_object")
        modified = head.headers.get("Last-Modified")
    records = []
    with source:
        with zipfile.ZipFile(source) as official:
            members = checked_zip_members(official, max_expanded_bytes=32 * 1024**3)
            selected = {}
            for row in missing:
                matches = [name for name in members if PurePosixPath(name).name == row["video"]]
                if len(matches) != 1:
                    raise ValueError("official_exact_video_id_not_unique_or_missing")
                selected[row["video"]] = members[matches[0]]
            if (
                any(x.file_size > 64 * 1024**2 for x in selected.values())
                or sum(x.file_size for x in selected.values()) > 500_000_000
            ):
                raise ValueError("selected_clip_expansion_budget_exceeded")
            output.mkdir(parents=True, exist_ok=False)
            for row in missing:
                entry = selected[row["video"]]
                temporary = output / (row["video"] + ".part")
                target = output / row["video"]
                checksum = hashlib.sha256()
                crc = 0
                written = 0
                with official.open(entry) as stream, temporary.open("xb") as dest:
                    for block in iter(lambda: stream.read(1024**2), b""):
                        written += len(block)
                        if written > entry.file_size:
                            raise ValueError("selected_clip_expansion_size_mismatch")
                        checksum.update(block)
                        crc = zlib.crc32(block, crc)
                        dest.write(block)
                if written != entry.file_size or crc != entry.CRC:
                    raise ValueError("selected_clip_crc_or_size_mismatch")
                temporary.rename(target)
                decoded = probe_decode(target, "video")
                records.append(
                    {
                        **row,
                        "official_archive_member": entry.filename,
                        "bytes": written,
                        "sha256": checksum.hexdigest(),
                        "crc32": f"{crc:08x}",
                        "source_compressed_bytes": entry.compress_size,
                        "full_video_decode": decoded,
                        "official_end_within_full_media_duration": row["end"]
                        <= decoded["duration_seconds"],
                    }
                )
                print(json.dumps({"recovered_public_clips": len(records)}), flush=True)
        return {
            "status": "official_missing_clips_recovered",
            "exit_code": 0,
            "primary_source": {
                "url": SOURCE_URL,
                "bytes": SOURCE_BYTES,
                "etag": SOURCE_ETAG,
                "s3_version_id": SOURCE_VERSION,
                "last_modified": modified,
                "binding": "Every HEAD/Range response must match exact version and ETag",
                "whole_archive_downloaded": False,
                "whole_archive_sha256": None,
            },
            "range_requests": source.requests,
            "range_download_bytes": source.downloaded,
            "range_download_budget_bytes": source.budget,
            "original_star_archive_sha256": STAR_SHA256,
            "public_task_metadata_sha256": task["sha256"],
            "task": "Action Sequence",
            "original_archive_present_rows": 188,
            "supplemental_rows": len(records),
            "combined_present_rows": 188 + len(records),
            "records": records,
            "training_admitted": False,
            "commercial_release_admitted": False,
            "license_url": "https://prior.allenai.org/projects/data/charades/license.txt",
            "media_scope": (
                "Internal evaluation only; no media redistribution or trained derivatives"
            ),
            "teacher_14_task_2786_scope": "Still missing; no inferred exclusions or label changes",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["star", "metadata", "config", "output", "report"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    source = HTTPRangeFile(SOURCE_URL, SOURCE_BYTES, SOURCE_ETAG, SOURCE_VERSION)
    try:
        result = recover(args.star, args.metadata, args.config, args.output, source)
    except Exception as error:
        result = {
            "status": "blocked",
            "exit_code": 1,
            "error_type": type(error).__name__,
            "reason": str(error)[:400],
            "training_admitted": False,
        }
    result.update(
        range_download_bytes=source.downloaded,
        range_requests=source.requests,
        range_download_budget_bytes=source.budget,
        elapsed_seconds=time.monotonic() - start,
        script_sha256=sha256(Path(__file__)),
        gpu_seconds=0,
        cost_usd=0,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "exit_code": result["exit_code"]}))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
