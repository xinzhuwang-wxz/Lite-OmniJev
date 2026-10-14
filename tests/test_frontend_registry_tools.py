from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_tool(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_canary = load_tool(
    "frontend_registry_build_canary",
    "infra/offline-tools/frontend-registry/build_registry_canary.py",
)
cache_sources = load_tool(
    "frontend_registry_cache_sources",
    "infra/offline-tools/frontend-registry/cache_registry_sources.py",
)


def registry_item(
    path: str, *, name: str = "button", content: str = "export {}"
) -> dict[str, object]:
    return {"name": name, "files": [{"path": path, "content": content}]}


def test_build_canary_refuses_existing_work_or_report_before_commands(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    report = tmp_path / "report.json"
    frontend = tmp_path / "frontend"
    cache = tmp_path / "cache"
    store = tmp_path / "store"
    work.mkdir()
    frontend.mkdir()
    cache.mkdir()

    def fail_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("subprocess should not run when outputs already exist")

    monkeypatch.setattr(build_canary, "run", fail_run)
    with pytest.raises(build_canary.ExistingOutputError):
        build_canary.main(
            [
                "--work-dir",
                str(work),
                "--report",
                str(report),
                "--frontend-dir",
                str(frontend),
                "--cache-root",
                str(cache),
                "--store-dir",
                str(store),
            ]
        )

    work = tmp_path / "fresh-work"
    report.write_text("{}\n")
    with pytest.raises(build_canary.ExistingOutputError):
        build_canary.main(
            [
                "--work-dir",
                str(work),
                "--report",
                str(report),
                "--frontend-dir",
                str(frontend),
                "--cache-root",
                str(cache),
                "--store-dir",
                str(store),
            ]
        )


def test_cache_sources_refuses_existing_cache_or_report_before_fetch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    report = tmp_path / "report.json"
    frontend = tmp_path / "frontend"
    cache.mkdir()
    frontend.mkdir()

    def fail_write_item(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network/cache writes should not start when outputs already exist")

    monkeypatch.setattr(cache_sources, "write_item", fail_write_item)
    with pytest.raises(cache_sources.ExistingOutputError):
        cache_sources.main(
            [
                "--cache-root",
                str(cache),
                "--report",
                str(report),
                "--frontend-dir",
                str(frontend),
            ]
        )

    cache = tmp_path / "fresh-cache"
    report.write_text("{}\n")
    with pytest.raises(cache_sources.ExistingOutputError):
        cache_sources.main(
            [
                "--cache-root",
                str(cache),
                "--report",
                str(report),
                "--frontend-dir",
                str(frontend),
            ]
        )


def test_registry_write_rejects_escape_without_partial_write(tmp_path: pathlib.Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    items = [
        registry_item("registry/new-york-v4/ui/button.tsx", content="valid"),
        registry_item("../escape.tsx", content="bad", name="input"),
    ]

    with pytest.raises(build_canary.RegistryFileError):
        build_canary.write_registry_files(work, items)

    assert not (work / "src/registry/new-york-v4/ui/button.tsx").exists()
    assert not (tmp_path / "escape.tsx").exists()


def test_registry_write_rejects_duplicate_targets(tmp_path: pathlib.Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    items = [
        registry_item("registry/new-york-v4/ui/button.tsx", content="one"),
        registry_item("registry/new-york-v4/ui/button.tsx", content="two"),
    ]

    with pytest.raises(build_canary.RegistryFileError):
        build_canary.write_registry_files(work, items)

    assert not (work / "src/registry/new-york-v4/ui/button.tsx").exists()


def test_registry_write_rejects_symlink_parent(tmp_path: pathlib.Path) -> None:
    work = tmp_path / "work"
    escape = tmp_path / "escape"
    work.mkdir()
    escape.mkdir()
    (work / "src").symlink_to(escape, target_is_directory=True)

    with pytest.raises(build_canary.RegistryFileError):
        build_canary.write_registry_files(
            work, [registry_item("registry/new-york-v4/ui/button.tsx")]
        )

    assert not (escape / "registry/new-york-v4/ui/button.tsx").exists()


def test_cache_source_url_and_size_guards() -> None:
    with pytest.raises(ValueError):
        cache_sources.validate_https_url("http://ui.shadcn.com/r/button.json")
    with pytest.raises(ValueError):
        cache_sources.validate_https_url("https://example.com/r/button.json")
    with pytest.raises(cache_sources.FetchLimitError):
        cache_sources.check_size(b"abc", "https://ui.shadcn.com/r/button.json", 2)
