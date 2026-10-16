"""Check release inventory files without treating their presence as stage acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.assets import verify_file


def check(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    if manifest.get("schema") != "lite-omnijev.release-inventory.v1":
        raise ValueError("unsupported inventory schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty files required")
    results = []
    seen = set()
    for entry in entries:
        identifier = entry["id"]
        if not identifier or identifier in seen:
            raise ValueError("unique nonempty ids required")
        seen.add(identifier)
        if entry["state"] == "pending":
            if not entry.get("reason"):
                raise ValueError("pending entry requires reason")
            results.append({"id": identifier, "status": "pending", "reason": entry["reason"]})
            continue
        if entry["state"] != "available":
            raise ValueError("state must be available or pending")
        path = Path(entry["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("inventory paths must stay under root")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("inventory symlink escapes root")
        try:
            actual = verify_file(resolved, entry)
            results.append({"id": identifier, "status": "verified", "sha256": actual})
        except (ValueError, OSError) as exc:
            results.append({"id": identifier, "status": "invalid", "error": str(exc)})
    complete = all(row["status"] == "verified" for row in results)
    return {
        "status": "inventory_verified" if complete else "inventory_incomplete",
        "exit_code": 0 if complete else 1,
        "claim": "file integrity only; not S4/S5/S6/FINAL or production acceptance",
        "files": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = check(json.loads(args.manifest.read_text()), args.root)
    print(json.dumps(report, indent=2))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
