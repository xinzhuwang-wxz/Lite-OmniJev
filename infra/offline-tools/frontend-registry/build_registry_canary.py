"""Build a synthetic Vite app from cached registry source without network."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import textwrap
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
DEFAULT_CACHE = Path("artifacts/runtime-builds/offline-tools/frontend/registry-sources")
DEFAULT_FRONTEND = Path("infra/offline-tools/frontend")
DEFAULT_STORE = Path("artifacts/runtime-builds/offline-tools/frontend/pnpm-store")
DEFAULT_WORK = Path("artifacts/runtime-builds/offline-tools/frontend/registry-build-canary")
DEFAULT_REPORT = Path("reports/frontend-registry-offline-build.json")
DEFAULT_PNPM_COMMAND = ("npm", "exec", "--offline", "--yes", "pnpm@12.11.2", "--")


class ExistingOutputError(RuntimeError):
    """Raised when a canary output path would overwrite existing evidence."""


class RegistryFileError(RuntimeError):
    """Raised when cached registry file metadata is unsafe or ambiguous."""


def ensure_new_path(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ExistingOutputError(f"{label} already exists, choose a new path: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def load_items(cache_root: Path) -> list[JsonObject]:
    items: list[JsonObject] = []
    for path in sorted(cache_root.rglob("*.json")):
        item = json.loads(path.read_text())
        if not isinstance(item, dict):
            raise ValueError(f"registry item is not an object: {path}")
        items.append(item)
    return items


def safe_source_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RegistryFileError(f"invalid registry source path: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RegistryFileError(f"registry source path escapes cache root: {value}")
    return path


def mapped_target(provider_name: str, source_path: Path) -> Path:
    source = source_path.as_posix()
    if provider_name.startswith("primitives-"):
        prefix = "registry/primitives/"
        if not source.startswith(prefix):
            raise RegistryFileError(f"Animate UI primitive has unexpected path: {source}")
        return Path("src/components/animate-ui/primitives") / source_path.relative_to(prefix)
    if provider_name.startswith("lib-"):
        prefix = "registry/lib/"
        if not source.startswith(prefix):
            raise RegistryFileError(f"Animate UI lib has unexpected path: {source}")
        return Path("src/lib") / source_path.relative_to(prefix)
    prefix = "registry/new-york-v4/ui/"
    if not source.startswith(prefix):
        raise RegistryFileError(f"shadcn registry item has unexpected path: {source}")
    return Path("src") / source_path


def validate_output_path(work: Path, relative_target: Path) -> None:
    if relative_target.is_absolute() or ".." in relative_target.parts:
        raise RegistryFileError(f"mapped target escapes work dir: {relative_target}")
    if relative_target.parts[0] != "src":
        raise RegistryFileError(f"mapped target is outside src/: {relative_target}")
    target = work / relative_target
    current = work
    if current.is_symlink():
        raise RegistryFileError(f"refusing symlink work dir: {current}")
    for part in relative_target.parent.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise RegistryFileError(f"refusing symlink in output path: {current}")
    if target.exists() or target.is_symlink():
        raise RegistryFileError(f"registry target already exists: {relative_target}")
    try:
        target.parent.resolve().relative_to(work.resolve())
    except ValueError as exc:
        raise RegistryFileError(f"mapped target escapes work dir: {relative_target}") from exc


def planned_registry_files(
    work: Path, items: Sequence[Mapping[str, Any]]
) -> list[tuple[Path, str]]:
    planned: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for item in items:
        provider_name = str(item.get("name"))
        files = item.get("files", [])
        if not isinstance(files, list):
            continue
        for file_item in files:
            if not isinstance(file_item, dict) or not isinstance(file_item.get("content"), str):
                continue
            source_path = safe_source_path(file_item.get("path"))
            target = mapped_target(provider_name, source_path)
            validate_output_path(work, target)
            if target in seen:
                raise RegistryFileError(f"duplicate registry target: {target}")
            seen.add(target)
            planned.append((target, str(file_item["content"])))
    return planned


def write_registry_files(work: Path, items: Sequence[Mapping[str, Any]]) -> list[str]:
    planned = planned_registry_files(work, items)
    written: list[str] = []
    for target, content in planned:
        output = work / target
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        written.append(target.as_posix())
    return sorted(written)


def write_canary_sources(work: Path) -> None:
    (work / "src/lib").mkdir(parents=True, exist_ok=True)
    (work / "src/lib/utils.ts").write_text(
        "import { clsx, type ClassValue } from 'clsx'\n"
        "import { twMerge } from 'tailwind-merge'\n\n"
        "export function cn(...inputs: ClassValue[]) {\n"
        "  return twMerge(clsx(inputs))\n"
        "}\n"
    )
    (work / "index.html").write_text(
        '<div id="root"></div><script type="module" src="/src/main.tsx"></script>\n'
    )
    (work / "src/main.tsx").write_text(
        textwrap.dedent(
            """\
            import * as React from 'react'
            import { createRoot } from 'react-dom/client'
            import { Alert, AlertDescription, AlertTitle } from './registry/new-york-v4/ui/alert'
            import { Button } from './registry/new-york-v4/ui/button'
            import {
  Card, CardContent, CardHeader, CardTitle,
} from './registry/new-york-v4/ui/card'
            import {
  Dialog, DialogContent, DialogTitle, DialogTrigger,
} from './registry/new-york-v4/ui/dialog'
            import { Input } from './registry/new-york-v4/ui/input'
            import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from './registry/new-york-v4/ui/select'
            import {
  Tabs, TabsContent, TabsList, TabsTrigger,
} from './registry/new-york-v4/ui/tabs'
            import { Textarea } from './registry/new-york-v4/ui/textarea'
            import {
              Tabs as AnimateTabs,
              TabsContent as AnimateTabsContent,
              TabsList as AnimateTabsList,
              TabsTrigger as AnimateTabsTrigger,
            } from './components/animate-ui/primitives/animate/tabs'

            function App() {
              return <main>
                <Card>
                  <CardHeader><CardTitle>Registry canary</CardTitle></CardHeader>
                  <CardContent>
                    <Alert>
                      <AlertTitle>Ready</AlertTitle>
                      <AlertDescription>offline</AlertDescription>
                    </Alert>
                    <Input defaultValue="input" />
                    <Textarea defaultValue="textarea" />
                    <Button type="button">Button</Button>
                    <Select defaultValue="a">
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent><SelectItem value="a">A</SelectItem></SelectContent>
                    </Select>
                    <Tabs defaultValue="one">
                      <TabsList><TabsTrigger value="one">One</TabsTrigger></TabsList>
                      <TabsContent value="one">Panel</TabsContent>
                    </Tabs>
                    <Dialog>
                      <DialogTrigger asChild><Button>Open</Button></DialogTrigger>
                      <DialogContent><DialogTitle>Dialog</DialogTitle></DialogContent>
                    </Dialog>
                    <AnimateTabs defaultValue="x">
                      <AnimateTabsList>
                        <AnimateTabsTrigger value="x">X</AnimateTabsTrigger>
                      </AnimateTabsList>
                      <AnimateTabsContent value="x">Animated</AnimateTabsContent>
                    </AnimateTabs>
                  </CardContent>
                </Card>
              </main>
            }

            createRoot(document.getElementById('root')!).render(<App />)
            """
        )
    )
    (work / "vite.config.ts").write_text(
        "import { defineConfig } from 'vite'\n"
        "import react from '@vitejs/plugin-react'\n"
        "import { fileURLToPath, URL } from 'node:url'\n\n"
        "export default defineConfig({\n"
        "  plugins: [react()],\n"
        "  resolve: {\n"
        "    alias: {\n"
        "      '@': fileURLToPath(new URL('./src', import.meta.url)),\n"
        "      'cn': fileURLToPath(new URL('./src/lib/utils.ts', import.meta.url)),\n"
        "    },\n"
        "  },\n"
        "})\n"
    )
    (work / "tsconfig.json").write_text(
        '{"compilerOptions":{"target":"ES2022","useDefineForClassFields":true,'
        '"lib":["DOM","DOM.Iterable","ES2022"],"allowJs":false,"skipLibCheck":true,'
        '"esModuleInterop":true,"allowSyntheticDefaultImports":true,"strict":false,'
        '"forceConsistentCasingInFileNames":true,"module":"ESNext","moduleResolution":"Bundler",'
        '"resolveJsonModule":true,"isolatedModules":true,"noEmit":true,"jsx":"react-jsx"},'
        '"include":["src","vite.config.ts"]}\n'
    )


def copy_frontend_seed(frontend: Path, work: Path) -> None:
    for name in ("package.json", "pnpm-lock.yaml", "pnpm-workspace.yaml"):
        source = frontend / name
        if source.exists():
            shutil.copy2(source, work / name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--frontend-dir", type=Path, default=DEFAULT_FRONTEND)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pnpm-command", nargs="+", default=list(DEFAULT_PNPM_COMMAND))
    args = parser.parse_args(argv)

    started = time.time()
    ensure_new_path(args.work_dir, "work dir")
    ensure_new_path(args.report, "report")
    args.work_dir.mkdir(parents=True)
    copy_frontend_seed(args.frontend_dir, args.work_dir)
    items = load_items(args.cache_root)
    written = write_registry_files(args.work_dir, items)
    write_canary_sources(args.work_dir)
    pnpm_command = [str(part) for part in args.pnpm_command]
    install_command = [
        *pnpm_command,
        "install",
        "--offline",
        "--frozen-lockfile",
        "--trust-lockfile",
        "--ignore-scripts",
        "--store-dir",
        str(args.store_dir.resolve()),
    ]
    build_command = [
        *pnpm_command,
        "--offline",
        "exec",
        "vite",
        "build",
    ]
    version = run([*pnpm_command, "--version"], cwd=args.work_dir)
    install = run(install_command, cwd=args.work_dir)
    if install.returncode == 0:
        build = run(build_command, cwd=args.work_dir)
    else:
        build = subprocess.CompletedProcess(
            build_command,
            1,
            "",
            "install failed; build skipped to preserve offline evidence\n",
        )
    dist_files = (
        sorted(
            str(path.relative_to(args.work_dir))
            for path in (args.work_dir / "dist").rglob("*")
            if path.is_file()
        )
        if (args.work_dir / "dist").exists()
        else []
    )
    report: JsonObject = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": "lite-omnijev.frontend-registry-offline-build.v1",
        "work_dir": str(args.work_dir),
        "store_dir": str(args.store_dir),
        "registry_files_written": written,
        "package_manager_version": f"pnpm {version.stdout.strip()}",
        "pnpm_invocation": {
            "install": " ".join(install_command),
            "build": " ".join(build_command),
        },
        "install": {
            "returncode": install.returncode,
            "stdout_tail": install.stdout[-4000:],
            "stderr_tail": install.stderr[-4000:],
        },
        "build": {
            "returncode": build.returncode,
            "stdout_tail": build.stdout[-4000:],
            "stderr_tail": build.stderr[-4000:],
        },
        "dist_files": dist_files,
        "dist_hashes": {path: sha256_file(args.work_dir / path) for path in dist_files},
        "elapsed_sec": round(time.time() - started, 1),
        "not_claimed": [
            "No product UI generated",
            "No browser visual QA",
            "No full registry crawl",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "install": install.returncode,
                "build": build.returncode,
                "dist_files": len(dist_files),
            },
            sort_keys=True,
        )
    )
    return install.returncode or build.returncode


if __name__ == "__main__":
    raise SystemExit(main())
