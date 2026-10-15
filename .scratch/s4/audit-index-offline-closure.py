"""Read-only Decision Index cache, dynamic-path and Git closure audit.

No upstream import/execution, no pickle, no benchmark rows retained. ForecastBench
scanner captures only the top-level question_set string and skips all other values.
Run on the input host; stdout is a metadata/hash receipt, never a readiness verdict.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TOKEN_SHA = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
ROOT = Path("/root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-work-001")
RAW = Path("/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3")
GIT_REPOS = (
    "searchless_chess",
    "habermas_machine",
    "pop909cl",
    "isarcasm",
    "sata",
    "contractnli",
    "nli4ct",
    "esci",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metadata_string(path: Path, field: str) -> str:
    """Stream JSON structure, capturing no nested/benchmark value strings."""
    depth = 0
    in_string = escape = False
    want_key = True
    key = None
    role = "skip"
    token = bytearray()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            for char in chunk:
                if in_string:
                    if role != "skip":
                        token.append(char)
                    if escape:
                        escape = False
                    elif char == 92:
                        escape = True
                    elif char == 34:
                        in_string = False
                        if role == "key":
                            key = json.loads(token)
                            want_key = False
                        elif role == "value":
                            return json.loads(token)
                    continue
                if char == 34:
                    in_string = True
                    role = (
                        "key"
                        if depth == 1 and want_key
                        else "value"
                        if depth == 1 and key == field
                        else "skip"
                    )
                    token = bytearray(b'"') if role != "skip" else bytearray()
                elif char in (123, 91):
                    depth += 1
                elif char in (125, 93):
                    depth -= 1
                elif char == 44 and depth == 1:
                    key = None
                    want_key = True
    raise ValueError(f"missing top-level string metadata field: {field}")


def git_read(path: Path, *argv: str) -> dict:
    if not path.is_dir():
        return {"exists": False}
    proc = subprocess.run(
        ["git", "-C", str(path), *argv],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
    )
    return {
        "exists": True,
        "exit_code": proc.returncode,
        "value": proc.stdout.strip() if proc.returncode == 0 else None,
    }


def audit(root: Path, raw: Path, site_packages: Path, extra_caches: list[Path]) -> dict:
    source = root / "source/decision_index/suite/build"
    repos = root / "work/artifacts/benchmark-suite/raw/repos"
    assignments = {}
    for node in ast.parse((source / "acquire.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GIT" for t in node.targets
        ):
            assignments = ast.literal_eval(node.value)
    calls = []
    for path in sorted(source.glob("*.py")):
        if path.name.startswith("._"):
            continue
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if "git_revision(" in text or "get_encoding(" in text:
                calls.append(
                    {
                        "path": str(path.relative_to(root / "source")),
                        "line": line,
                        "code": text.strip(),
                    }
                )
    provenance = []
    for name in GIT_REPOS:
        path = repos / name
        observed = git_read(path, "rev-parse", "HEAD")
        top = git_read(path, "rev-parse", "--show-toplevel")
        original = raw / "repos" / name
        original_head = git_read(original, "rev-parse", "HEAD")
        original_top = git_read(original, "rev-parse", "--show-toplevel")
        expected = assignments[name][1]
        provenance.append(
            {
                "name": name,
                "required_path": str(path),
                "expected_revision": expected,
                "local_dot_git_exists": (path / ".git").exists(),
                "observed_head": observed,
                "observed_toplevel": top,
                "original_head": original_head,
                "original_toplevel": original_top,
                "original_matches_pin": (
                    original_head.get("value") == expected
                    and original_top.get("value") == str(original)
                ),
                "safe_for_unpatched_git_revision": top.get("value") == str(path)
                and observed.get("value") == expected,
            }
        )
    forecast = repos / "forecastbench-datasets/datasets"
    original = raw / "repos/forecastbench-datasets/datasets"
    references = []
    for resolution in sorted((forecast / "resolution_sets").glob("*_resolution_set.json")):
        filename = metadata_string(resolution, "question_set")
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("unsafe ForecastBench question_set metadata path")
        required = forecast / "question_sets" / filename
        origin = original / "question_sets" / filename
        resolved = origin.resolve()
        safe_origin = resolved.is_relative_to(original.resolve()) and resolved.is_file()
        references.append(
            {
                "resolution_file": resolution.name,
                "question_set": filename,
                "required_exists": required.is_file(),
                "required_is_symlink": required.is_symlink(),
                "source_is_symlink": origin.is_symlink(),
                "source_link_target": os.readlink(origin) if origin.is_symlink() else None,
                "source_resolved_basename": resolved.name if safe_origin else None,
                "required_sha256": sha(required)
                if required.is_file() and not required.is_symlink()
                else None,
                "source_resolved_sha256": sha(resolved) if safe_origin else None,
            }
        )
    installed_encoding = site_packages / "tiktoken_ext/openai_public.py"
    definition = next(
        node
        for node in ast.parse(installed_encoding.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "cl100k_base"
    )
    loader = next(
        node
        for node in ast.walk(definition)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_tiktoken_bpe"
    )
    extracted_url = ast.literal_eval(loader.args[0])
    extracted_hash = ast.literal_eval(
        next(k.value for k in loader.keywords if k.arg == "expected_hash")
    )
    if extracted_url != URL or extracted_hash != TOKEN_SHA:
        raise ValueError("installed cl100k_base definition differs from audited cache identity")
    reference_set = sorted((r["resolution_file"], r["question_set"]) for r in references)
    reference_hash = hashlib.sha256(
        json.dumps(reference_set, separators=(",", ":")).encode()
    ).hexdigest()
    cache_key = hashlib.sha1(URL.encode()).hexdigest()
    candidates = [
        root / "work/cache/tiktoken",
        Path(tempfile.gettempdir()) / "data-gym-cache",
        *extra_caches,
    ]
    for variable in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]))
    caches = []
    for directory in dict.fromkeys(candidates):
        path = directory / cache_key
        digest = sha(path) if path.is_file() and not path.is_symlink() else None
        caches.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": digest,
                "matches": digest == TOKEN_SHA,
            }
        )
    return {
        "ready": False,
        "tool_versions": {
            "python": platform.python_version(),
            "git": subprocess.run(
                ["git", "--version"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "tiktoken": importlib.metadata.version("tiktoken"),
        },
        "upstream_executed": False,
        "benchmark_rows_retained": False,
        "source_calls": calls,
        "git_revision": provenance,
        "forecastbench": {
            "resolution_count": len(references),
            "reference_set_sha256": reference_hash,
            "references": references,
            "missing_question_sets": sorted(
                {r["question_set"] for r in references if not r["required_exists"]}
            ),
            "latest_llm_referenced": any(
                r["question_set"] == "latest-llm.json" for r in references
            ),
        },
        "tiktoken": {
            "url": extracted_url,
            "extracted_definition": {
                "path": str(installed_encoding),
                "function": "cl100k_base",
                "line": definition.lineno,
            },
            "cache_key": cache_key,
            "expected_sha256": TOKEN_SHA,
            "candidates": caches,
            "installed_source_sha256": {
                str(path): sha(path)
                for path in (
                    site_packages / "tiktoken/load.py",
                    site_packages / "tiktoken_ext/openai_public.py",
                )
            },
        },
        "auditor_sha256": sha(Path(__file__)) if Path(__file__).is_file() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.raw, args.site_packages, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
