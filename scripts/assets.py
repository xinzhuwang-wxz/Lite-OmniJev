"""Pinned, bounded model asset preparation; never imports model code or weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METADATA = {
    "README.md",
    ".gitattributes",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "feature_extractor_config.json",
    "model.safetensors.index.json",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def allowed(name: str) -> bool:
    return name in METADATA or bool(re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name))


def plan_assets(config: dict[str, Any], names: list[str], api: Any) -> dict[str, Any]:
    models = []
    for name in names:
        entry = config["models"][name]
        repo = entry["repo_id"]
        if not repo or repo.rsplit("/", 1)[-1].upper() in {"JEV-27B", "JEV-27B-VL"}:
            raise ValueError(f"Unresolved or prohibited model: {name}")
        info = api.model_info(
            repo, revision=entry.get("revision", "main"), files_metadata=True, timeout=30
        )
        if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
            raise ValueError("Hub did not return an immutable commit SHA")
        license_id = (info.card_data or {}).get("license")
        if not license_id or license_id not in entry.get("accepted_licenses", []):
            raise ValueError(f"License not reviewed for {repo}: {license_id}")
        files = []
        for sibling in info.siblings or []:
            if not allowed(sibling.rfilename):
                continue
            if sibling.size is None or sibling.size < 0:
                raise ValueError(f"Unknown file size: {sibling.rfilename}")
            lfs = sibling.lfs
            files.append(
                {
                    "path": sibling.rfilename,
                    "size": sibling.size,
                    "kind": "weights" if sibling.rfilename.endswith(".safetensors") else "metadata",
                    "upstream_sha256": lfs.sha256 if lfs else None,
                    "git_blob_sha1": None if lfs else sibling.blob_id,
                }
            )
        paths = {f["path"] for f in files}
        if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= paths:
            raise ValueError(f"Required model/tokenizer files missing: {repo}")
        if not any(f["kind"] == "weights" for f in files):
            raise ValueError(f"No approved safetensors weights: {repo}")
        models.append(
            {
                "name": name,
                "repo_id": repo,
                "revision": info.sha,
                "source": f"https://huggingface.co/{repo}/tree/{info.sha}",
                "license": license_id,
                "gated": bool(info.gated),
                "files": files,
                "total_bytes": sum(f["size"] for f in files),
                "metadata_bytes": sum(f["size"] for f in files if f["kind"] == "metadata"),
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "planned",
        "models": models,
    }


def model_directory(root: Path, model: dict[str, Any]) -> Path:
    repo, revision = str(model["repo_id"]), str(model["revision"])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Unsafe repository or unpinned revision in manifest")
    return root / repo.replace("/", "__") / revision


def selected_files(
    manifest: dict[str, Any], metadata_only: bool
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if manifest.get("schema_version") != 1 or not manifest.get("models"):
        raise ValueError("Missing or unsupported asset manifest")
    selected = []
    for model in manifest["models"]:
        if not model.get("files") or not model.get("license"):
            raise ValueError("Manifest lacks files or license provenance")
        for file in model["files"]:
            if not allowed(file["path"]) or file["size"] < 0:
                raise ValueError("Unsafe file or invalid size in manifest")
            is_weight = file["path"].endswith(".safetensors")
            if file["kind"] != ("weights" if is_weight else "metadata"):
                raise ValueError("Incorrect file kind in manifest")
            if not metadata_only or not is_weight:
                selected.append((model, file))
        paths = {file["path"] for file in model["files"]}
        required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
        weights = {path for path in paths if path.endswith(".safetensors")}
        if len(paths) != len(model["files"]) or not required <= paths or not weights:
            raise ValueError(
                "Model inventory incomplete: required metadata and weights missing/duplicated"
            )
        if weights != {"model.safetensors"}:
            shards = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", path) for path in weights]
            if any(match is None for match in shards):
                raise ValueError("Model inventory incomplete: ambiguous weights")
            parts = [(int(match[1]), int(match[2])) for match in shards if match is not None]
            totals = {total for _, total in parts}
            total = max(totals)
            if (
                len(totals) != 1
                or len(parts) != total
                or {index for index, _ in parts} != set(range(1, total + 1))
                or "model.safetensors.index.json" not in paths
            ):
                raise ValueError("Model inventory incomplete: missing safetensors shard/index")
    if not selected:
        raise ValueError("No files selected")
    return selected


def verify_file(path: Path, file: dict[str, Any]) -> str:
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    if path.stat().st_size != file["size"]:
        raise ValueError(f"Size mismatch: {path.name}")
    digest = hashlib.sha256()
    git = hashlib.sha1(f"blob {file['size']}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            git.update(block)
    actual = digest.hexdigest()
    for field in ("upstream_sha256", "actual_sha256"):
        if file.get(field) and file[field] != actual:
            raise ValueError(f"SHA256 mismatch ({field}): {path.name}")
    if file.get("git_blob_sha1") and file["git_blob_sha1"] != git.hexdigest():
        raise ValueError(f"Git blob mismatch: {path.name}")
    if not any(file.get(key) for key in ("upstream_sha256", "git_blob_sha1", "actual_sha256")):
        raise ValueError(f"No trusted checksum: {path.name}")
    return actual


def prepare_assets(
    manifest: dict[str, Any],
    root: Path,
    *,
    metadata_only: bool,
    download: Any = None,
    workers: int = 2,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    selected = selected_files(manifest, metadata_only)
    root.mkdir(parents=True, exist_ok=True)
    remaining = []
    for model, file in selected:
        directory = model_directory(root, model)
        path = directory / file["path"]
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Asset path escapes root")
        if path.exists():
            # Never silently overwrite corrupt assets or re-baseline their checksums.
            file["actual_sha256"] = verify_file(path, file)
        else:
            remaining.append((model, file))
    if download is None and remaining:
        raise ValueError(f"Missing file: {remaining[0][1]['path']}")
    if remaining:
        pending_bytes = sum(file["size"] for _, file in remaining)
        reserve = max(64 * 1024**2, pending_bytes // 10)
        free = shutil.disk_usage(root).free
        if free < pending_bytes + reserve:
            raise ValueError(
                f"Insufficient disk: need {pending_bytes + reserve} bytes, free {free}"
            )

        def fetch(pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
            model, file = pair
            directory = model_directory(root, model)
            download(
                repo_id=model["repo_id"],
                filename=file["path"],
                revision=model["revision"],
                local_dir=directory,
            )
            file["actual_sha256"] = verify_file(directory / file["path"], file)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, remaining))
    if not metadata_only:
        for model in manifest["models"]:
            directory = model_directory(root, model)
            index_path = directory / "model.safetensors.index.json"
            if index_path.exists():
                index = json.loads(index_path.read_text())
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                weights = {file["path"] for file in model["files"] if file["kind"] == "weights"}
                if (
                    not isinstance(weight_map, dict)
                    or not weight_map
                    or any(not isinstance(value, str) for value in weight_map.values())
                    or set(weight_map.values()) != weights
                ):
                    raise ValueError(
                        "Model inventory incomplete: weight index does not match shards"
                    )
    manifest["status"] = "metadata_verified" if metadata_only else "all_files_verified"
    manifest["verified_at"] = datetime.now(UTC).isoformat()
    return {
        "status": manifest["status"],
        "files_verified": len(selected),
        "bytes_verified": sum(file["size"] for _, file in selected),
        "weights_ready": not metadata_only,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, default=Path("configs/assets.json"))
    plan.add_argument("--model", action="append", dest="models")
    plan.add_argument("--output", type=Path, required=True)
    for command in ("download", "verify"):
        child = sub.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--root", type=Path, default=Path("artifacts/models"))
        child.add_argument("--metadata-only", action="store_true")
        child.add_argument("--workers", type=int, default=2)
    parser.add_argument("--report", type=Path, default=Path("reports/assets-last-run.json"))
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"command": args.command, "status": "blocked", "exit_code": 1}
    manifest = None
    try:
        if args.command == "plan":
            from huggingface_hub import HfApi

            config = json.loads(args.config.read_text())
            manifest = plan_assets(config, args.models or ["e4b"], HfApi())
            write_json(args.output, manifest)
            report.update(
                status="planned", models=len(manifest["models"]), manifest=str(args.output)
            )
        else:
            manifest = json.loads(args.manifest.read_text())
            download = None
            if args.command == "download":
                from huggingface_hub import hf_hub_download

                download = hf_hub_download
            report.update(
                prepare_assets(
                    manifest,
                    args.root,
                    metadata_only=args.metadata_only,
                    download=download,
                    workers=args.workers,
                )
            )
        report["exit_code"] = 0
    except Exception as exc:
        # HTTP exception strings may contain signed URLs or credentials; keep only the class.
        report["blocker"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        if manifest is not None:
            manifest["status"] = "blocked"
    finally:
        if manifest is not None and args.command != "plan":
            write_json(args.manifest, manifest)
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
