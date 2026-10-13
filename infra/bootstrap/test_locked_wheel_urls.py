"""Offline public CLI behavior checks for the bounded bootstrap converter."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("locked_wheel_urls.py")
HASH = "sha256:" + "a" * 64


def run(tmp_path: Path, wheels: list[str], requirement: str, *args: str):
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "sample"\nversion = "1.0"\nwheels = [\n'
        + ",\n".join(
            f'{{url = "https://files.pythonhosted.org/packages/{w}", hash = "{HASH}"}}'
            for w in wheels
        )
        + "\n]\n"
    )
    requirements = tmp_path / "export.txt"
    requirements.write_text(requirement)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lock",
            str(lock),
            "--requirements",
            str(requirements),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_selects_target_wheel_and_keeps_hash_marker(tmp_path):
    result = run(
        tmp_path,
        [
            "sample-1.0-cp311-cp311-macosx_11_0_arm64.whl",
            "sample-1.0-cp311-cp311-manylinux_2_35_x86_64.whl",
            "sample-1.0-py3-none-any.whl",
        ],
        f"sample==1.0 ; sys_platform == 'linux' \\\n --hash={HASH}\n"
        "missing==2.0 ; sys_platform == 'darwin'\n",
    )
    assert result.returncode == 0, result.stderr
    assert "manylinux_2_35_x86_64.whl" in result.stdout
    assert f"--hash={HASH}" in result.stdout
    assert 'sys_platform == "linux"' in result.stdout
    assert "missing" not in result.stdout


@pytest.mark.parametrize(
    "wheels,requirement",
    [
        ([], f"sample==1.0 --hash={HASH}"),
        (["sample-1.0-cp312-cp312-manylinux_2_35_x86_64.whl"], f"sample==1.0 --hash={HASH}"),
        (["sample-1.0-py3-none-any.whl"], "sample==1.0 --hash=sha256:" + "b" * 64),
        (["sample-1.0-py3-none-any.whl"], f"sample==2.0 --hash={HASH}"),
        (["sample-1.0-py3-none-any.whl"], "sample==1.0"),
    ],
)
def test_refuses_unlocked_or_unavailable_artifacts(tmp_path, wheels, requirement):
    result = run(tmp_path, wheels, requirement)
    assert result.returncode == 1
    assert not result.stdout
    assert "error:" in result.stderr


def test_explicit_mirror_changes_url_not_hash(tmp_path):
    result = run(
        tmp_path,
        ["sample-1.0-py3-none-any.whl"],
        f"sample==1.0 --hash={HASH}",
        "--pypi-mirror-base",
        "https://mirror.example/pypi",
    )
    assert result.returncode == 0, result.stderr
    assert "https://mirror.example/pypi/packages/sample-1.0-py3-none-any.whl" in result.stdout
    assert f"--hash={HASH}" in result.stdout
