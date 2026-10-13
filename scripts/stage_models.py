"""Stage reviewed model files without importing upstream code or loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.assets import METADATA, verify_file, write_json

CHECKPOINTS = {
    "akhilaaa3/Jev-Omni": {"head.pt"},
    "Valen-Team/Valen-Preview-0923": {"checkpoint.pt"},
}
EXTRA_METADATA = {
    ".gitattributes",
    "decision_config.json",
    "sha256.json",
    "verification.json",
    "training_metadata.json",
    "video_preprocessor_config.json",
}


def inventory(
    manifest: dict[str, Any], root: Path
) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    entries = []
    seen = set()
    if manifest.get("schema_version") != 1 or not manifest.get("models"):
        raise ValueError("Missing model inventory")
    for model in manifest["models"]:
        repo, revision = model["repo_id"], model["revision"]
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or repo.split("/")[-1].upper() in {
            "JEV-27B",
            "JEV-27B-VL",
        }:
            raise ValueError("Unresolved or prohibited model")
        if not re.fullmatch(r"[a-f0-9]{40}", revision) or model.get("license") != "apache-2.0":
            raise ValueError("Unreviewed revision or model license")
        unsafe = any(f["required"] and f["path"].endswith(".pt") for f in model["files"])
        directory = (
            root / ("quarantine" if unsafe else "models") / repo.replace("/", "__") / revision
        )
        if repo == "akhilaaa3/Jev-Omni":
            directory = root / "quarantine" / "teacher" / revision
        for file in model["files"]:
            if not file["required"]:
                continue
            name = file["path"]
            allowed = (
                name in METADATA | EXTRA_METADATA
                or bool(re.fullmatch(r"model(?:[.\w-]*)\.safetensors", name))
                or name in CHECKPOINTS.get(repo, set())
            )
            if (
                not allowed
                or Path(name).name != name
                or type(file["bytes"]) is not int
                or file["bytes"] < 0
            ):
                raise ValueError("Unsafe asset path or size")
            if not (file.get("lfs_sha256") or file.get("git_blob_sha1")):
                raise ValueError("Missing upstream hash")
            path = directory / name
            key = (repo, revision, name)
            if key in seen or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Duplicate or escaping asset")
            seen.add(key)
            entries.append((model, file, path))
    return entries


def check(path: Path, file: dict[str, Any]) -> str:
    return verify_file(
        path,
        {
            "size": file["bytes"],
            "upstream_sha256": file.get("lfs_sha256"),
            "git_blob_sha1": file.get("git_blob_sha1"),
        },
    )


def stage(
    manifest: dict[str, Any], root: Path, report_path: Path, download: Any, workers: int = 2
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    entries = inventory(manifest, root)
    root.mkdir(parents=True, exist_ok=True)
    missing = sum(f["bytes"] for _, f, p in entries if not p.exists())
    if shutil.disk_usage(root).free < missing + max(1024**3, missing // 10):
        raise ValueError("Insufficient disk for missing files and temporary reserve")
    report: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "command": sys.argv,
        "cost_scope": "CPU/network only; provider price unavailable; no GPU compute requested",
        "weights_loaded": False,
        "training_admitted": False,
        "files": [],
        "failures": [],
    }
    lock = threading.Lock()
    write_json(report_path, report)

    def fetch(entry: tuple[dict[str, Any], dict[str, Any], Path]) -> None:
        model, file, path = entry
        result = {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "path": file["path"],
            "bytes": file["bytes"],
            "destination": str(path),
        }
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                download(
                    repo_id=model["repo_id"],
                    filename=file["path"],
                    revision=model["revision"],
                    local_dir=path.parent,
                )
            result.update(status="hash_verified", sha256=check(path, file))
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__)
        with lock:
            report["files"].append(result)
            if result["status"] == "failed":
                report["failures"].append(result)
            write_json(report_path, report)

    # Small metadata first makes CPU preparation possible while weights transfer.
    entries.sort(key=lambda entry: entry[1]["bytes"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, entries))
    report.update(
        status="failed" if report["failures"] else "all_required_files_hash_verified_not_admitted",
        exit_code=1 if report["failures"] else 0,
        finished_at=datetime.now(UTC).isoformat(),
    )
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/project-model-assets.json"))
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--report", type=Path, default=Path("reports/project-model-download.json"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download

        result = stage(
            json.loads(args.manifest.read_text()),
            args.root,
            args.report,
            hf_hub_download,
            args.workers,
        )
    except Exception as error:
        result = {"status": "failed", "exit_code": 1, "error_type": type(error).__name__}
        write_json(args.report, result)
    print(json.dumps({k: result[k] for k in ("status", "exit_code")}))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
