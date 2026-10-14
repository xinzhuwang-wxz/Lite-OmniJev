"""Download a hashed requirements wheelhouse one requirement at a time."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time


def _requirement_blocks(requirements: pathlib.Path) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in requirements.read_text().splitlines():
        if line and not line.startswith((" ", "#")):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _package_name(block: list[str]) -> str:
    return block[0].split("==", 1)[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=pathlib.Path, required=True)
    parser.add_argument("--dest", type=pathlib.Path, required=True)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--progress", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)

    blocks = _requirement_blocks(args.requirements)
    started = time.time()
    with args.log.open("a", encoding="utf-8") as log:
        log.write(f"START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} blocks={len(blocks)}\n")
        for index, block in enumerate(blocks, start=1):
            name = _package_name(block)
            with tempfile.TemporaryDirectory(prefix="req-one-") as temp_dir:
                temp_req = pathlib.Path(temp_dir) / "requirements.txt"
                temp_req.write_text("\n".join(block) + "\n", encoding="utf-8")
                command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--dest",
                    str(args.dest),
                    "--require-hashes",
                    "--no-deps",
                    "-r",
                    str(temp_req),
                ]
                log.write(f"BLOCK {index}/{len(blocks)} {name} start\n")
                log.flush()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)

            file_count = sum(1 for path in args.dest.iterdir() if path.is_file())
            progress = {
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "current": name,
                "current_index": index,
                "elapsed_sec": round(time.time() - started, 1),
                "files": file_count,
                "returncode": result.returncode,
                "schema": "lite-omnijev.lowrss-wheelhouse-download-progress.v1",
                "total": len(blocks),
            }
            args.progress.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
            log.write(
                f"BLOCK {index}/{len(blocks)} {name} rc={result.returncode} files={file_count}\n"
            )
            log.flush()
            if result.returncode != 0:
                return result.returncode

        log.write(
            f"DONE {time.strftime('%Y-%m-%dT%H:%M:%S%z')} elapsed={time.time() - started:.1f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
