"""Rebuild and import-check the serve-vllm026 runtime from a local wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import textwrap
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any

ROOT = pathlib.Path("/root/autodl-tmp/lite-omnijev")
DEFAULT_WHEELHOUSE = ROOT / "artifacts/runtime-builds/python-wheels/serve-vllm026"
DEFAULT_ENV = pathlib.Path("/root/autodl-tmp/envs/serve-vllm026-offline-check")
DEFAULT_LOG = ROOT / "logs/serve-vllm026-offline-install.log"
DEFAULT_REPORT = ROOT / "reports/serve-vllm026-offline-import.json"
MANDATORY_IMPORTS = ("torch", "transformers", "vllm")
MANDATORY_PROBES = ("torch_probe",)
JsonObject = dict[str, Any]


class ExistingOutputError(RuntimeError):
    """Raised before any destructive overwrite of canary evidence or envs."""


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def ensure_fresh_outputs(env: pathlib.Path, report: pathlib.Path, log: pathlib.Path) -> None:
    existing = [str(path) for path in (env, report, log) if path.exists()]
    if existing:
        joined = ", ".join(existing)
        raise ExistingOutputError(f"refusing to overwrite existing canary outputs: {joined}")


def probe_errors(
    parsed: Mapping[str, Any],
    *,
    mandatory_imports: Collection[str],
    mandatory_probes: Collection[str],
) -> list[str]:
    errors: list[str] = []
    imports = parsed.get("imports")
    if not isinstance(imports, Mapping):
        return ["imports_missing"]
    for name in mandatory_imports:
        value = imports.get(name)
        if not isinstance(value, Mapping) or value.get("status") != "ok":
            errors.append(f"import:{name}")
    for name in mandatory_probes:
        value = parsed.get(name)
        if not isinstance(value, Mapping) or value.get("status") == "error":
            errors.append(f"probe:{name}")
    return errors


def module_probe(python: pathlib.Path) -> JsonObject:
    code = textwrap.dedent(
        r"""
        import importlib
        import json
        import sys

        mandatory = ["torch", "transformers", "vllm"]
        out = {"python": sys.version, "imports": {}}
        failed = False
        for name in mandatory:
            try:
                mod = importlib.import_module(name)
                out["imports"][name] = {
                    "status": "ok",
                    "version": getattr(mod, "__version__", None),
                    "file": getattr(mod, "__file__", None),
                }
            except Exception as exc:
                failed = True
                out["imports"][name] = {
                    "status": "error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        try:
            import torch

            out["torch_probe"] = {
                "version": torch.__version__,
                "file": torch.__file__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cpu_sum": float(torch.ones(2, dtype=torch.float32).sum().item()),
            }
        except Exception as exc:
            failed = True
            out["torch_probe"] = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        print(json.dumps(out, sort_keys=True))
        raise SystemExit(1 if failed else 0)
        """
    )
    result = run([str(python), "-c", code])
    payload: JsonObject = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    try:
        payload["parsed"] = json.loads(result.stdout)
    except json.JSONDecodeError:
        pass
    return payload


def build_report(
    *,
    wheelhouse: pathlib.Path,
    requirements: pathlib.Path,
    env: pathlib.Path,
    log: pathlib.Path,
) -> JsonObject:
    files = sorted(path for path in wheelhouse.iterdir() if path.is_file())
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": "lite-omnijev.serve-vllm026-offline-import.v1",
        "wheelhouse": str(wheelhouse),
        "requirements": str(requirements),
        "requirements_sha256": sha(requirements),
        "file_count": len(files),
        "contains": {
            "vllm-0.26.0": any(path.name.startswith("vllm-0.26.0") for path in files),
            "torch-2.11.0": any(path.name.startswith("torch-2.11.0") for path in files),
            "triton-3.6.0": any(path.name.startswith("triton-3.6.0") for path in files),
        },
        "env": str(env),
        "log": str(log),
        "not_claimed": ["No GPU execution", "No vLLM server startup", "No model forward"],
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=pathlib.Path, default=DEFAULT_WHEELHOUSE)
    parser.add_argument("--requirements", type=pathlib.Path)
    parser.add_argument("--env", type=pathlib.Path, default=DEFAULT_ENV)
    parser.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    wheelhouse = args.wheelhouse
    requirements = args.requirements or wheelhouse / "requirements.txt"
    started = time.time()
    try:
        ensure_fresh_outputs(args.env, args.report, args.log)
    except ExistingOutputError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report = build_report(
        wheelhouse=wheelhouse,
        requirements=requirements,
        env=args.env,
        log=args.log,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    create = run([sys.executable, "-m", "venv", str(args.env)])
    report["venv_create"] = {"returncode": create.returncode, "stderr": create.stderr[-4000:]}
    if create.returncode != 0:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return create.returncode

    python = args.env / "bin/python"
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--require-hashes",
        "-r",
        str(requirements),
    ]
    with args.log.open("w", encoding="utf-8") as log_handle:
        install = subprocess.run(command, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
    report["install_returncode"] = install.returncode
    report["log_sha256"] = sha(args.log)
    return_code = install.returncode

    if install.returncode == 0:
        probe = module_probe(python)
        parsed = probe.get("parsed")
        if isinstance(parsed, Mapping):
            report.update(parsed)
            errors = probe_errors(
                parsed,
                mandatory_imports=MANDATORY_IMPORTS,
                mandatory_probes=MANDATORY_PROBES,
            )
            report["mandatory_probe_errors"] = errors
            if errors and probe["returncode"] == 0:
                probe["returncode"] = 1
        report["probe_returncode"] = probe["returncode"]
        if probe["returncode"] != 0:
            report["probe_stdout"] = str(probe["stdout"])[-4000:]
            report["probe_stderr"] = str(probe["stderr"])[-4000:]
        return_code = int(probe["returncode"])

    report["elapsed_sec"] = round(time.time() - started, 1)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
