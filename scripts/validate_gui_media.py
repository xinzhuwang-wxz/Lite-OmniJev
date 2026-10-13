"""Read-only full PNG decode for the selected GUIOdyssey train episodes; no network."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib
import io
import json
import platform
import time
from pathlib import Path
from typing import Any

from scripts.gui_odyssey_media import (
    REVISION,
    SCREENSHOT_RE,
    entries_by_episode,
    load_entries_jsonl,
    optional_flock,
)


def validate_media(
    manifest: Path, plan_path: Path, media_root: Path, lock_path: str | None = None
) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "gpu_used": False,
        "training_admitted": False,
        "test_media_read": False,
        "revision": REVISION,
        "decoded_images": 0,
        "verified_bytes": 0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Selected train PNG bytes/CRC/SHA and full decode; not model quality admission",
    }
    try:
        report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        report["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        plan = json.loads(plan_path.read_text())
        selected = plan["selected_episodes"]
        if plan["partition"] != "train" or not selected:
            raise ValueError("nonempty train-only plan required")
        ids = [item["episode_id"] for item in selected]
        if len(ids) != len(set(ids)) or any(item["partition"] != "train" for item in selected):
            raise ValueError("unique train episodes required")
        by_episode = entries_by_episode(load_entries_jsonl(manifest))
        receipts = hashlib.sha256()
        for episode in selected:
            entries = by_episode[episode["episode_id"]]
            if len(entries) != episode["screenshot_count"] or not entries:
                raise ValueError("episode screenshot count mismatch")
            for entry in entries:
                match = SCREENSHOT_RE.fullmatch(entry.name)
                if match is None:
                    raise ValueError("unexpected screenshot name")
                path = media_root / "screenshots" / Path(entry.name).name
                if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
                    raise ValueError("regular screenshot required")
                with optional_flock(lock_path):
                    payload = path.read_bytes()
                    if len(payload) != entry.uncompressed_size:
                        raise ValueError("screenshot size mismatch")
                    if binascii.crc32(payload) & 0xFFFFFFFF != entry.crc32:
                        raise ValueError("screenshot CRC mismatch")
                    image_module = importlib.import_module("PIL.Image")
                    report["pillow"] = importlib.import_module("PIL").__version__
                    with image_module.open(io.BytesIO(payload)) as image:
                        if image.format != "PNG":
                            raise ValueError("PNG screenshot required")
                        image.load()
                        receipt = {
                            "name": entry.name,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload),
                            "width": image.width,
                            "height": image.height,
                        }
                receipts.update(json.dumps(receipt, sort_keys=True).encode() + b"\n")
                report["decoded_images"] += 1
                report["verified_bytes"] += len(payload)
            if sum(entry.uncompressed_size for entry in entries) != episode["uncompressed_bytes"]:
                raise ValueError("episode byte count mismatch")
        if report["verified_bytes"] != plan["selected_uncompressed_bytes"]:
            raise ValueError("plan byte count mismatch")
        report.update(
            status="passed",
            exit_code=0,
            selected_episode_count=len(ids),
            receipt_sha256=receipts.hexdigest(),
        )
    except Exception as error:
        report["error_type"] = type(error).__name__
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lock-path")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("report already exists; use a fresh report path")
    report = validate_media(args.manifest, args.plan, args.media_root, args.lock_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
