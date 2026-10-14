"""Build a fresh before-GPU teacher runtime from audited wheelhouses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time
from collections.abc import Collection, Mapping, Sequence
from typing import Any

ROOT = pathlib.Path("/root/autodl-tmp/lite-omnijev")
DEFAULT_BASE = ROOT / "artifacts/runtime-builds/python-wheels/serve-next-vllm031"
DEFAULT_PYARROW = ROOT / "artifacts/runtime-builds/python-wheels/data-pyarrow"
DEFAULT_ENV = pathlib.Path("/root/autodl-tmp/envs/before-gpu-teacher-runtime-offline-check")
DEFAULT_LOG = ROOT / "logs/before-gpu-teacher-runtime-offline-install.log"
DEFAULT_REPORT = ROOT / "reports/before-gpu-teacher-runtime-canary.json"
DEFAULT_SOURCE = ROOT / "artifacts/before-gpu-teacher-runtime-source/main-08709d9"
MANDATORY_IMPORTS = (
    "torch",
    "transformers",
    "safetensors",
    "PIL.Image",
    "numpy",
    "pyarrow",
    "pyarrow.parquet",
    "cv2",
    "torchvision",
    "scripts.s0_teacher_eval",
    "eval.teacher_benchmarks",
    "train.teacher_inference",
    "train.teacher_media",
)
MANDATORY_PROBES = (
    "torch_probe",
    "pil_numpy_probe",
    "torchvision_probe",
    "cv2_probe",
    "pyarrow_probe",
)
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


def canary(python: pathlib.Path, source: pathlib.Path) -> JsonObject:
    code = textwrap.dedent(
        r"""
        import importlib
        import importlib.metadata as md
        import json
        import os
        import pathlib
        import sys
        import tempfile

        source = pathlib.Path(os.environ["TEACHER_RUNTIME_SOURCE"])
        sys.path.insert(0, str(source))
        modules = [
            "torch",
            "transformers",
            "safetensors",
            "PIL.Image",
            "numpy",
            "pyarrow",
            "pyarrow.parquet",
            "cv2",
            "torchvision",
            "scripts.s0_teacher_eval",
            "eval.teacher_benchmarks",
            "train.teacher_inference",
            "train.teacher_media",
        ]
        out = {"imports": {}, "versions": {}}
        failed = False
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                out["imports"][module_name] = {
                    "status": "ok",
                    "file": getattr(module, "__file__", None),
                }
            except Exception as exc:
                failed = True
                out["imports"][module_name] = {
                    "status": "error",
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
        for dist in [
            "torch",
            "transformers",
            "safetensors",
            "Pillow",
            "numpy",
            "pyarrow",
            "opencv-python-headless",
            "torchvision",
            "vllm",
        ]:
            try:
                out["versions"][dist] = md.version(dist)
            except Exception as exc:
                out["versions"][dist] = {"error": type(exc).__name__}
        try:
            import torch

            out["torch_probe"] = {
                "cpu_sum": float(torch.ones(2).sum().item()),
                "cuda_available": bool(torch.cuda.is_available()),
                "version": torch.__version__,
            }
        except Exception as exc:
            failed = True
            out["torch_probe"] = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        try:
            import numpy as np
            from PIL import Image

            image = Image.new("RGB", (2, 2), color=(10, 20, 30))
            out["pil_numpy_probe"] = {
                "shape": list(np.asarray(image).shape),
                "sum": int(np.asarray(image).sum()),
            }
        except Exception as exc:
            failed = True
            out["pil_numpy_probe"] = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        try:
            from PIL import Image
            import torchvision.transforms.functional as functional

            image = Image.new("RGB", (2, 2), color=(10, 20, 30))
            out["torchvision_probe"] = {
                "tensor_shape": list(functional.pil_to_tensor(image).shape)
            }
        except Exception as exc:
            failed = True
            out["torchvision_probe"] = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        try:
            import cv2

            out["cv2_probe"] = {"version": cv2.__version__}
        except Exception as exc:
            failed = True
            out["cv2_probe"] = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "tiny.parquet"
                table = pa.table({"x": [1, 2, 3]})
                pq.write_table(table, path)
                read_back = pq.read_table(path)
            out["pyarrow_probe"] = {
                "rows": read_back.num_rows,
                "columns": read_back.num_columns,
            }
        except Exception as exc:
            failed = True
            out["pyarrow_probe"] = {
                "status": "error",
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        print(json.dumps(out, sort_keys=True))
        raise SystemExit(1 if failed else 0)
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TEACHER_RUNTIME_SOURCE": str(source),
        }
    )
    proc = run([str(python), "-c", code], env=env)
    result: JsonObject = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        result["parsed"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        pass
    return result


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wheelhouse", type=pathlib.Path, default=DEFAULT_BASE)
    parser.add_argument("--pyarrow-wheelhouse", type=pathlib.Path, default=DEFAULT_PYARROW)
    parser.add_argument("--base-requirements", type=pathlib.Path)
    parser.add_argument("--pyarrow-requirements", type=pathlib.Path)
    parser.add_argument("--env", type=pathlib.Path, default=DEFAULT_ENV)
    parser.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_requirements = args.base_requirements or args.base_wheelhouse / "requirements.txt"
    pyarrow_requirements = args.pyarrow_requirements or args.pyarrow_wheelhouse / "requirements.txt"
    started = time.time()
    try:
        ensure_fresh_outputs(args.env, args.report, args.log)
    except ExistingOutputError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    report: JsonObject = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": "lite-omnijev.before-gpu-teacher-runtime-canary.v1",
        "selected_runtime": "serve-next-vllm031 plus data-pyarrow overlay",
        "source_snapshot": str(args.source),
        "env": str(args.env),
        "base_wheelhouse": str(args.base_wheelhouse),
        "pyarrow_wheelhouse": str(args.pyarrow_wheelhouse),
        "base_requirements_sha256": sha(base_requirements),
        "pyarrow_requirements_sha256": sha(pyarrow_requirements),
        "install_log": str(args.log),
        "not_claimed": [
            "No GPU execution",
            "No model forward",
            "No processor AutoProcessor load",
            "No vLLM server startup",
        ],
    }
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
        str(args.base_wheelhouse),
        "--find-links",
        str(args.pyarrow_wheelhouse),
        "--require-hashes",
        "-r",
        str(base_requirements),
        "-r",
        str(pyarrow_requirements),
    ]
    report["install_command"] = command
    with args.log.open("w", encoding="utf-8") as log_handle:
        install = subprocess.run(command, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
    report["install_returncode"] = install.returncode
    report["install_log_sha256"] = sha(args.log)
    return_code = install.returncode

    if install.returncode == 0:
        probe = canary(python, args.source)
        parsed = probe.get("parsed")
        if isinstance(parsed, Mapping):
            errors = probe_errors(
                parsed,
                mandatory_imports=MANDATORY_IMPORTS,
                mandatory_probes=MANDATORY_PROBES,
            )
            report["probe"] = parsed
            report["mandatory_probe_errors"] = errors
            if errors and probe["returncode"] == 0:
                probe["returncode"] = 1
        else:
            report["probe_stdout_tail"] = str(probe["stdout"])[-4000:]
            report["probe_stderr_tail"] = str(probe["stderr"])[-4000:]
        report["probe_returncode"] = probe["returncode"]
        return_code = int(probe["returncode"])

    report["elapsed_sec"] = round(time.time() - started, 1)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
