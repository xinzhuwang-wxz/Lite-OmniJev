"""Cache bounded official shadcn/ui and Animate UI registry items."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path("/root/autodl-tmp/lite-omnijev")
DEFAULT_CACHE = ROOT / "artifacts/runtime-builds/offline-tools/frontend/registry-sources"
DEFAULT_REPORT = ROOT / "reports/frontend-registry-sources.json"
SHADCN_ITEMS = ("button", "card", "input", "textarea", "select", "tabs", "alert", "dialog")
ANIMATE_ITEMS = (
    "primitives-animate-tabs",
    "primitives-effects-highlight",
    "primitives-animate-slot",
    "lib-get-strict-context",
)
LICENSES = {
    "shadcn-ui": "MIT",
    "animate-ui": "MIT",
}
JsonObject = dict[str, Any]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str) -> tuple[JsonObject, bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "Lite-OmniJev-offline-prep"})
    content_type: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            content_type = response.headers.get("content-type")
    except urllib.error.URLError:
        node = subprocess.run(
            [
                "node",
                "-e",
                (
                    "fetch(process.argv[1], {headers: {'user-agent': 'Lite-OmniJev-offline-prep'}})"
                    ".then(async response => {"
                    "if (!response.ok) throw new Error(`HTTP ${response.status}`);"
                    "process.stdout.write(Buffer.from(await response.arrayBuffer()));"
                    "}).catch(error => {console.error(error); process.exit(1);})"
                ),
                url,
            ],
            text=False,
            capture_output=True,
            check=False,
        )
        if node.returncode == 0:
            body = node.stdout
        else:
            completed = subprocess.run(
                ["curl", "-fsSL", "-A", "Lite-OmniJev-offline-prep", url],
                text=False,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                node_error = node.stderr[-1000:].decode(errors="replace")
                curl_error = completed.stderr[-1000:].decode(errors="replace")
                raise RuntimeError(
                    f"node fetch failed for {url}: {node_error}; curl fallback failed: {curl_error}"
                ) from None
            body = completed.stdout
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError(f"registry item is not an object: {url}")
    return parsed, body, content_type


def item_dependencies(item: Mapping[str, Any]) -> list[str]:
    deps: list[str] = []
    for key in ("dependencies", "devDependencies"):
        value = item.get(key, [])
        if isinstance(value, list):
            deps.extend(str(dep) for dep in value)
    return sorted(set(deps))


def registry_dependencies(item: Mapping[str, Any]) -> list[str]:
    value = item.get("registryDependencies", [])
    return sorted(str(dep) for dep in value) if isinstance(value, list) else []


def write_item(cache_root: Path, provider: str, name: str, url: str) -> JsonObject:
    item, raw, content_type = fetch_json(url)
    if item.get("name") != name:
        raise ValueError(f"registry item name mismatch for {url}: {item.get('name')!r}")
    provider_dir = cache_root / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    path = provider_dir / f"{name}.json"
    path.write_bytes(raw)
    files = item.get("files", [])
    file_count = len(files) if isinstance(files, list) else 0
    inline_file_sha256: dict[str, str] = {}
    if isinstance(files, list):
        for file_item in files:
            if isinstance(file_item, dict) and isinstance(file_item.get("content"), str):
                file_path = str(file_item.get("path", "unknown"))
                inline_file_sha256[file_path] = sha256_bytes(file_item["content"].encode())
    return {
        "name": name,
        "provider": provider,
        "url": url,
        "content_type": content_type,
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_file(path),
        "license": LICENSES[provider],
        "schema": item.get("$schema"),
        "type": item.get("type"),
        "dependencies": item_dependencies(item),
        "registryDependencies": registry_dependencies(item),
        "file_count": file_count,
        "inline_file_sha256": inline_file_sha256,
    }


def readback_check(cache_root: Path, items: Iterable[Mapping[str, Any]]) -> JsonObject:
    checked: dict[str, JsonObject] = {}
    for item in items:
        path = Path(str(item["path"]))
        parsed = json.loads(path.read_text())
        checked[str(item["provider"]) + ":" + str(item["name"])] = {
            "name": parsed.get("name"),
            "sha256_matches": sha256_file(path) == item["sha256"],
            "file_count": len(parsed.get("files", [])),
        }
    return {"cache_root": str(cache_root), "items": checked}


def package_name(dep: str) -> str:
    if dep.startswith("@"):
        return "@".join(dep.split("@")[:2])
    return dep.split("@", 1)[0]


def is_virtual_dependency(dep: str) -> bool:
    return dep == "cn"


def lock_has_package(lockfile: Path, package: str) -> bool:
    text = lockfile.read_text()
    if package.startswith("@"):
        return f"{package}@" in text or f"{package}:" in text
    return f"  {package}:" in text or f"/{package}@" in text or f"  {package}@" in text


def frontend_dependency_versions(package_json: Path) -> dict[str, str]:
    parsed = json.loads(package_json.read_text())
    dependencies = parsed.get("dependencies", {})
    if not isinstance(dependencies, dict):
        return {}
    return {str(name): str(version) for name, version in dependencies.items()}


def npm_metadata(package: str, version: str) -> JsonObject:
    completed = subprocess.run(
        [
            "npm",
            "view",
            f"{package}@{version}",
            "version",
            "license",
            "dist.integrity",
            "dist.tarball",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "version": version,
            "source": f"npm registry metadata via npm view {package}",
            "error": completed.stderr[-1000:],
            "returncode": completed.returncode,
        }
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        raise ValueError(f"npm view returned non-object metadata for {package}@{version}")
    return {
        "version": version,
        "registry_version_observed": parsed.get("version"),
        "license": parsed.get("license"),
        "integrity": parsed.get("dist.integrity"),
        "tarball": parsed.get("dist.tarball"),
        "source": f"npm registry metadata via npm view {package}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frontend-dir", type=Path, default=Path("infra/offline-tools/frontend"))
    parser.add_argument("--remote-sync-status")
    parser.add_argument("--remote-sync-error")
    parser.add_argument("--local-canary-report", type=Path)
    args = parser.parse_args(argv)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    items: list[JsonObject] = []
    for name in SHADCN_ITEMS:
        items.append(
            write_item(
                args.cache_root,
                "shadcn-ui",
                name,
                f"https://ui.shadcn.com/r/styles/new-york-v4/{name}.json",
            )
        )
    for name in ANIMATE_ITEMS:
        items.append(
            write_item(args.cache_root, "animate-ui", name, f"https://animate-ui.com/r/{name}.json")
        )

    npm_deps = sorted(
        {
            package_name(dep)
            for item in items
            for dep in item["dependencies"]
            if not is_virtual_dependency(dep)
        }
    )
    lockfile = args.frontend_dir / "pnpm-lock.yaml"
    missing = [dep for dep in npm_deps if not lock_has_package(lockfile, dep)]
    frontend_versions = frontend_dependency_versions(args.frontend_dir / "package.json")
    metadata = {
        dep: npm_metadata(dep, frontend_versions[dep])
        for dep in npm_deps
        if dep in frontend_versions
    }
    total_bytes = sum(int(item["bytes"]) for item in items)
    remote_sync_status: JsonObject | None = None
    if args.remote_sync_status:
        remote_sync_status = {"status": args.remote_sync_status}
        if args.remote_sync_error:
            remote_sync_status["last_error"] = args.remote_sync_error
        if args.local_canary_report:
            remote_sync_status["local_canary_used"] = str(args.local_canary_report)

    report: JsonObject = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": "lite-omnijev.frontend-registry-sources.v1",
        "scope": "bounded official registry source cache only; no product UI generated",
        "providers": {
            "shadcn-ui": {
                "source": "https://ui.shadcn.com/r/styles/new-york-v4/{name}.json",
                "docs": "https://ui.shadcn.com/docs/registry/api-reference",
                "license": LICENSES["shadcn-ui"],
            },
            "animate-ui": {
                "source": "https://animate-ui.com/r/{name}.json",
                "docs": "https://animate-ui.com/docs/components",
                "license": LICENSES["animate-ui"],
            },
        },
        "cache_root": str(args.cache_root),
        "total_raw_bytes": total_bytes,
        "budget": {
            "max_raw_bytes": 100 * 1024 * 1024,
            "within_budget": total_bytes < 100 * 1024 * 1024,
        },
        "items": items,
        "npm_dependencies_declared_by_registry": npm_deps,
        "npm_dependencies_missing_from_frontend_lock": missing,
        "npm_dependency_metadata": metadata,
        "offline_reader": readback_check(args.cache_root, items),
        "not_claimed": [
            "No product UI generated",
            "No full registry crawl",
            "No account or paid API used",
        ],
    }
    if remote_sync_status is not None:
        report["remote_sync_status"] = remote_sync_status
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"items": len(items), "bytes": total_bytes, "missing": missing}, sort_keys=True)
    )
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
