from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from scripts import run_before_gpu_teacher_runtime_canary as teacher_canary
from scripts import run_serve_vllm026_offline_canary as serve_canary


def test_serve_canary_refuses_existing_env_report_or_log(tmp_path: pathlib.Path) -> None:
    env = tmp_path / "env"
    report = tmp_path / "report.json"
    log = tmp_path / "install.log"
    env.mkdir()
    report.write_text("{}\n")
    log.write_text("old log\n")

    with pytest.raises(serve_canary.ExistingOutputError):
        serve_canary.ensure_fresh_outputs(env, report, log)


@pytest.mark.parametrize(
    ("parsed", "expected"),
    [
        ({"imports": {"torch": {"status": "error"}}, "torch_probe": {}}, ["import:torch"]),
        (
            {"imports": {"torch": {"status": "ok"}}, "torch_probe": {"status": "error"}},
            ["probe:torch_probe"],
        ),
        ({"torch_probe": {}}, ["imports_missing"]),
    ],
)
def test_serve_canary_detects_mandatory_probe_failures(
    parsed: dict[str, object], expected: list[str]
) -> None:
    assert (
        serve_canary.probe_errors(
            parsed,
            mandatory_imports=("torch",),
            mandatory_probes=("torch_probe",),
        )
        == expected
    )


def test_teacher_canary_refuses_existing_outputs(tmp_path: pathlib.Path) -> None:
    env = tmp_path / "env"
    report = tmp_path / "report.json"
    log = tmp_path / "install.log"
    env.mkdir()

    with pytest.raises(teacher_canary.ExistingOutputError):
        teacher_canary.ensure_fresh_outputs(env, report, log)


def test_teacher_canary_detects_import_and_probe_failures() -> None:
    parsed = {
        "imports": {
            "scripts.s0_teacher_eval": {"status": "ok"},
            "pyarrow.parquet": {"status": "error"},
        },
        "torch_probe": {"cpu_sum": 2.0},
        "pyarrow_probe": {"status": "error"},
    }

    assert teacher_canary.probe_errors(
        parsed,
        mandatory_imports=("scripts.s0_teacher_eval", "pyarrow.parquet"),
        mandatory_probes=("torch_probe", "pyarrow_probe"),
    ) == ["import:pyarrow.parquet", "probe:pyarrow_probe"]


def test_teacher_canary_job_overlay_records_require_manifest_hashes(tmp_path: pathlib.Path) -> None:
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "peft-0.21.2-py3-none-any.whl"
    payload = b"fake peft wheel"
    wheel.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": wheel.name,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        )
    )

    records = teacher_canary.job_overlay_records(wheelhouse, manifest, (wheel.name,))

    assert records == [
        {
            "name": wheel.name,
            "path": str(wheel),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]


def test_teacher_canary_job_overlay_rejects_hash_mismatch(tmp_path: pathlib.Path) -> None:
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    wheel = wheelhouse / "accelerate-1.15.0-py3-none-any.whl"
    wheel.write_bytes(b"changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"name": wheel.name, "sha256": "0" * 64}]}))

    with pytest.raises(teacher_canary.WheelManifestError, match="hash mismatch"):
        teacher_canary.job_overlay_records(wheelhouse, manifest, (wheel.name,))


def test_teacher_canary_job_overlay_install_uses_no_deps_exact_paths() -> None:
    records = [
        {"path": "/wheelhouse/pytest-9.1.1-py3-none-any.whl"},
        {"path": "/wheelhouse/peft-0.21.2-py3-none-any.whl"},
    ]

    command = teacher_canary.job_overlay_install_command(pathlib.Path("/env/bin/python"), records)

    assert command[:6] == [
        "/env/bin/python",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
    ]
    assert command[6:] == [record["path"] for record in records]


def test_teacher_canary_job_runtime_imports_are_mandatory() -> None:
    assert "pytest" in teacher_canary.MANDATORY_IMPORTS
    assert "peft" in teacher_canary.MANDATORY_IMPORTS
    assert "accelerate" in teacher_canary.MANDATORY_IMPORTS
