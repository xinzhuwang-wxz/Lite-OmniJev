"""Range-plan and selectively extract GUIOdyssey screenshot media.

This helper is intentionally stdlib-only. It does not accept gated terms or use
credentials; it reads the public pinned HF revision with HTTP Range requests.
"""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, cast

REPO_ID = "hflqf88888/GUIOdyssey"
REVISION = "61632d0f3f4d51d7e9561ce4f84347dd06b2019d"
PART_NAMES = [f"screenshots/screenshots.z{i:02d}" for i in range(1, 9)] + [
    "screenshots/screenshots.zip"
]
PART_SIZES = [10_737_418_240] * 8 + [6_740_164_517]
SCREENSHOT_RE = re.compile(r"^screenshots/(?P<episode>\d+)_(?P<step>\d+)\.png$")


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    disk_start: int
    local_header_offset: int


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: str
    episode_file: str
    split: str
    partition: str
    screenshot_count: int
    compressed_bytes: int
    uncompressed_bytes: int
    archive_member_count: int


def resolve_url(repo_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


class HTTPRangeReader:
    def __init__(self, url: str, timeout: int = 180, resolve_once: bool = False) -> None:
        self.url = url
        self.timeout = timeout
        if resolve_once:
            self.url = self._resolve(url)

    def _resolve(self, url: str) -> str:
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "lite-omnijev-gui-odyssey-range"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return str(response.geturl())
        except Exception:
            return url

    def read_range(self, start: int, end_inclusive: int) -> bytes:
        if start < 0 or end_inclusive < start:
            raise ValueError(f"invalid range {start}-{end_inclusive}")
        expected = end_inclusive - start + 1
        last_error: BaseException | None = None
        for attempt in range(1, 6):
            request = urllib.request.Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end_inclusive}",
                    "User-Agent": "lite-omnijev-gui-odyssey-range",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = cast(bytes, response.read())
                    if response.status != 206:
                        raise RuntimeError(
                            f"range request returned HTTP {response.status} for {self.url}"
                        )
                    if len(data) != expected:
                        raise RuntimeError(
                            "range request returned "
                            f"{len(data)} bytes, expected {expected} for {self.url}"
                        )
                    return data
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                if attempt == 5:
                    break
                time.sleep(min(30, 2**attempt))
        raise RuntimeError(
            f"failed range request {start}-{end_inclusive} for {self.url}: {last_error}"
        )


@contextlib.contextmanager
def optional_flock(lock_path: str | None) -> Iterator[None]:
    if not lock_path:
        yield
        return
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_u16(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<H", data, offset)[0])


def _read_u32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<L", data, offset)[0])


def _read_u64(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<Q", data, offset)[0])


def _zip64_values(extra: bytes) -> list[int]:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + size]
        pos += size
        if header_id == 0x0001:
            return [int(struct.unpack_from("<Q", payload, i)[0]) for i in range(0, len(payload), 8)]
    return []


def parse_central_directory(cd: bytes) -> list[ZipEntry]:
    entries: list[ZipEntry] = []
    pos = 0
    while pos < len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central directory signature at byte {pos}")
        fields = struct.unpack_from("<4s6H3L5H2L", cd, pos)
        method = fields[4]
        crc32 = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        disk_start = fields[13]
        local_header_offset = fields[16]
        name_start = pos + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        name = cd[name_start:extra_start].decode("utf-8")
        zip64 = _zip64_values(cd[extra_start:comment_start])
        z = iter(zip64)
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = next(z)
        if compressed_size == 0xFFFFFFFF:
            compressed_size = next(z)
        if local_header_offset == 0xFFFFFFFF:
            local_header_offset = next(z)
        if disk_start == 0xFFFF:
            disk_start = next(z)
        entries.append(
            ZipEntry(
                name=name,
                method=method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                disk_start=disk_start,
                local_header_offset=local_header_offset,
            )
        )
        pos = comment_start + comment_len
    return entries


def fetch_central_directory(
    repo_id: str = REPO_ID,
    revision: str = REVISION,
    part_sizes: list[int] | None = None,
) -> tuple[bytes, dict[str, int | str]]:
    sizes = part_sizes or PART_SIZES
    final_size = sizes[-1]
    final_url = resolve_url(repo_id, revision, PART_NAMES[-1])
    final = HTTPRangeReader(final_url)
    probe_size = min(8 * 1024 * 1024, final_size)
    probe_start = final_size - probe_size
    tail = final.read_range(probe_start, final_size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("EOCD signature not found in final split tail")
    eocd_abs = probe_start + eocd_pos
    locator_pos = eocd_pos - 20
    if locator_pos < 0 or tail[locator_pos : locator_pos + 4] != b"PK\x06\x07":
        raise RuntimeError("ZIP64 locator not found before EOCD")
    _, eocd64_disk, eocd64_offset, total_disks = struct.unpack_from("<4sLQL", tail, locator_pos)
    if eocd64_disk != len(sizes) - 1:
        raise RuntimeError(f"ZIP64 EOCD is on unexpected disk {eocd64_disk}")
    eocd64 = final.read_range(eocd64_offset, eocd64_offset + 75)
    if eocd64[:4] != b"PK\x06\x06":
        raise RuntimeError("bad ZIP64 EOCD signature")
    record_size = _read_u64(eocd64, 4)
    disk_no = _read_u32(eocd64, 16)
    cd_disk = _read_u32(eocd64, 20)
    disk_entries = _read_u64(eocd64, 24)
    total_entries = _read_u64(eocd64, 32)
    cd_size = _read_u64(eocd64, 40)
    cd_offset = _read_u64(eocd64, 48)
    if cd_disk != len(sizes) - 1:
        raise RuntimeError(f"central directory is on unexpected disk {cd_disk}")
    cd = final.read_range(cd_offset, cd_offset + cd_size - 1)
    meta = {
        "total_disks": total_disks,
        "zip64_record_size": record_size,
        "disk_no": disk_no,
        "central_directory_disk": cd_disk,
        "disk_entries": disk_entries,
        "total_entries": total_entries,
        "central_directory_bytes": cd_size,
        "central_directory_offset_in_final_part": cd_offset,
        "eocd_offset_in_final_part": eocd_abs,
        "central_directory_sha256": hashlib.sha256(cd).hexdigest(),
    }
    return cd, meta


def entries_by_episode(entries: Iterable[ZipEntry]) -> dict[str, list[ZipEntry]]:
    by_episode: dict[str, list[ZipEntry]] = {}
    for entry in entries:
        match = SCREENSHOT_RE.match(entry.name)
        if not match:
            continue
        by_episode.setdefault(match.group("episode"), []).append(entry)
    for values in by_episode.values():
        values.sort(key=lambda item: int(SCREENSHOT_RE.match(item.name).group("step")))  # type: ignore[union-attr]
    return by_episode


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return cast(Any, json.load(handle))


def build_episode_plans(
    all_annot_path: pathlib.Path,
    split_path: pathlib.Path,
    split_name: str,
    partition: str,
    entries: list[ZipEntry],
) -> list[EpisodePlan]:
    all_annot = load_json(all_annot_path)
    expected_steps = {str(row["episode_id"]): int(row["step_length"]) for row in all_annot}
    split = load_json(split_path)
    episode_files = split[partition]
    by_episode = entries_by_episode(entries)
    plans: list[EpisodePlan] = []
    for episode_file in episode_files:
        episode_id = episode_file[:-5] if episode_file.endswith(".json") else episode_file
        media = by_episode.get(episode_id, [])
        if len(media) != expected_steps[episode_id]:
            raise RuntimeError(
                f"episode {episode_id} expected {expected_steps[episode_id]} "
                f"screenshots, found {len(media)}"
            )
        plans.append(
            EpisodePlan(
                episode_id=episode_id,
                episode_file=episode_file,
                split=split_name,
                partition=partition,
                screenshot_count=len(media),
                compressed_bytes=sum(item.compressed_size for item in media),
                uncompressed_bytes=sum(item.uncompressed_size for item in media),
                archive_member_count=len(media),
            )
        )
    return plans


def select_under_budget(
    plans: list[EpisodePlan], budget_bytes: int, budget_field: str = "compressed_bytes"
) -> list[EpisodePlan]:
    if budget_field not in {"compressed_bytes", "uncompressed_bytes"}:
        raise ValueError(f"unsupported budget field {budget_field}")
    selected: list[EpisodePlan] = []
    total = 0
    for plan in sorted(plans, key=lambda item: item.episode_id):
        size = (
            plan.compressed_bytes if budget_field == "compressed_bytes" else plan.uncompressed_bytes
        )
        if total + size > budget_bytes:
            continue
        selected.append(plan)
        total += size
    return selected


def part_ranges(
    disk_start: int,
    offset: int,
    length: int,
    part_sizes: list[int] | None = None,
) -> Iterator[tuple[int, int, int]]:
    sizes = part_sizes or PART_SIZES
    disk = disk_start
    pos = offset
    remaining = length
    while remaining > 0:
        if disk >= len(sizes):
            raise RuntimeError("range crosses past final split part")
        available = sizes[disk] - pos
        take = min(available, remaining)
        yield disk, pos, pos + take - 1
        remaining -= take
        disk += 1
        pos = 0


def read_split_bytes(readers: list[HTTPRangeReader], disk: int, offset: int, length: int) -> bytes:
    chunks = []
    for part, start, end in part_ranges(disk, offset, length):
        chunks.append(readers[part].read_range(start, end))
    return b"".join(chunks)


def local_data_offset(readers: list[HTTPRangeReader], entry: ZipEntry) -> int:
    header = read_split_bytes(readers, entry.disk_start, entry.local_header_offset, 30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header signature for {entry.name}")
    name_len = _read_u16(header, 26)
    extra_len = _read_u16(header, 28)
    return entry.local_header_offset + 30 + name_len + extra_len


def read_entry_compressed(readers: list[HTTPRangeReader], entry: ZipEntry) -> bytes:
    name_len = len(entry.name.encode("utf-8"))
    probe_len = 30 + name_len + 256 + entry.compressed_size
    blob = read_split_bytes(readers, entry.disk_start, entry.local_header_offset, probe_len)
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header signature for {entry.name}")
    local_name_len = _read_u16(blob, 26)
    local_extra_len = _read_u16(blob, 28)
    data_start = 30 + local_name_len + local_extra_len
    data_end = data_start + entry.compressed_size
    if data_end <= len(blob):
        return blob[data_start:data_end]
    data_offset = entry.local_header_offset + data_start
    return read_split_bytes(readers, entry.disk_start, data_offset, entry.compressed_size)


def extract_entry(
    readers: list[HTTPRangeReader],
    entry: ZipEntry,
    out_path: pathlib.Path,
    decompress_lock_path: str | None = None,
) -> dict[str, Any]:
    compressed = read_entry_compressed(readers, entry)
    with optional_flock(decompress_lock_path):
        if entry.method == 8:
            payload = zlib.decompress(compressed, -15)
        elif entry.method == 0:
            payload = compressed
        else:
            raise RuntimeError(f"unsupported compression method {entry.method} for {entry.name}")
        crc = binascii.crc32(payload) & 0xFFFFFFFF
        if crc != entry.crc32:
            raise RuntimeError(f"crc mismatch for {entry.name}: {crc:08x} != {entry.crc32:08x}")
        if len(payload) != entry.uncompressed_size:
            raise RuntimeError(
                f"size mismatch for {entry.name}: {len(payload)} != {entry.uncompressed_size}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".partial")
        tmp.write_bytes(payload)
        os.replace(tmp, out_path)
    return {
        "name": entry.name,
        "path": str(out_path),
        "compressed_bytes": entry.compressed_size,
        "bytes": len(payload),
        "crc32": f"{entry.crc32:08x}",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def command_probe(args: argparse.Namespace) -> None:
    cd, meta = fetch_central_directory(args.repo_id, args.revision)
    entries = parse_central_directory(cd)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
    report = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "central_directory": meta,
        "entry_count": len(entries),
        "screenshot_entry_count": sum(1 for item in entries if SCREENSHOT_RE.match(item.name)),
        "manifest_jsonl": str(out),
        "manifest_jsonl_bytes": out.stat().st_size,
        "manifest_jsonl_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "probe_downloaded_bytes": 8 * 1024 * 1024 + 76 + int(meta["central_directory_bytes"]),
    }
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def load_entries_jsonl(path: pathlib.Path) -> list[ZipEntry]:
    return [ZipEntry(**json.loads(line)) for line in path.read_text().splitlines() if line]


def command_plan(args: argparse.Namespace) -> None:
    entries = load_entries_jsonl(pathlib.Path(args.manifest))
    plans = build_episode_plans(
        pathlib.Path(args.all_annot),
        pathlib.Path(args.split_file),
        args.split_name,
        args.partition,
        entries,
    )
    selected = select_under_budget(plans, args.budget_bytes, args.budget_field)
    report = {
        "split_name": args.split_name,
        "partition": args.partition,
        "budget_bytes": args.budget_bytes,
        "candidate_episode_count": len(plans),
        "candidate_compressed_bytes": sum(p.compressed_bytes for p in plans),
        "candidate_uncompressed_bytes": sum(p.uncompressed_bytes for p in plans),
        "selected_episode_count": len(selected),
        "selected_compressed_bytes": sum(p.compressed_bytes for p in selected),
        "selected_uncompressed_bytes": sum(p.uncompressed_bytes for p in selected),
        "budget_field": args.budget_field,
        "selection_rule": (
            "episode_id lexical order, whole episodes only, "
            "skip any episode that would exceed budget"
        ),
        "selected_episodes": [asdict(p) for p in selected],
    }
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "selected_episodes"},
            indent=2,
        )
    )


def command_fetch(args: argparse.Namespace) -> None:
    entries = load_entries_jsonl(pathlib.Path(args.manifest))
    by_episode = entries_by_episode(entries)
    plan = load_json(pathlib.Path(args.plan))
    selected_ids = [item["episode_id"] for item in plan["selected_episodes"]]
    readers = [
        HTTPRangeReader(resolve_url(args.repo_id, args.revision, name), resolve_once=True)
        for name in PART_NAMES
    ]
    out_root = pathlib.Path(args.output_root)
    started = time.time()
    tasks: list[tuple[ZipEntry, pathlib.Path]] = []
    for episode_id in selected_ids:
        for entry in by_episode[episode_id]:
            match = SCREENSHOT_RE.match(entry.name)
            if match is None:
                raise RuntimeError(f"non-screenshot entry selected: {entry.name}")
            step = match.group("step")
            out = out_root / "screenshots" / f"{episode_id}_{step}.png"
            tasks.append((entry, out))

    def one(task: tuple[ZipEntry, pathlib.Path]) -> dict[str, Any]:
        entry, out = task
        if out.exists() and not args.force:
            payload = out.read_bytes()
            return {
                "name": entry.name,
                "path": str(out),
                "compressed_bytes": entry.compressed_size,
                "bytes": len(payload),
                "crc32": f"{entry.crc32:08x}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "reused": True,
            }
        row = extract_entry(readers, entry, out, args.decompress_lock_path)
        row["reused"] = False
        return row

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for row in pool.map(one, tasks):
            results.append(row)
            if len(results) % args.progress_every == 0:
                progress = {
                    "phase": "fetch",
                    "done": len(results),
                    "elapsed_sec": round(time.time() - started, 1),
                }
                print(json.dumps(progress), flush=True)
    report = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "plan": args.plan,
        "output_root": str(out_root),
        "selected_episode_count": len(selected_ids),
        "screenshot_count": len(results),
        "compressed_bytes": sum(item["compressed_bytes"] for item in results),
        "bytes": sum(item["bytes"] for item in results),
        "reused_count": sum(1 for item in results if item.get("reused")),
        "elapsed_sec": round(time.time() - started, 3),
        "files": results,
    }
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("probe", help="range-fetch and parse the central directory")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(func=command_probe)
    p = sub.add_parser("plan", help="build a whole-episode selection under a byte budget")
    p.add_argument("--manifest", required=True)
    p.add_argument("--all-annot", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--split-name", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--budget-bytes", type=int, required=True)
    p.add_argument(
        "--budget-field",
        choices=["compressed_bytes", "uncompressed_bytes"],
        default="compressed_bytes",
    )
    p.add_argument("--output", required=True)
    p.set_defaults(func=command_plan)
    p = sub.add_parser("fetch", help="extract selected screenshots with HTTP ranges")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--manifest", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--decompress-lock-path")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=command_fetch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
