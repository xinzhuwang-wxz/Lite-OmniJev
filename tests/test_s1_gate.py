import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.s1_gate import E4B_TEXT_DEV500, ONEJEV4_TEXT_DEV500, main

METRIC_KEYS = [
    "smoke_steps",
    "smoke_accuracy",
    "loss_decreasing",
    "text_vs_onejev_delta",
    "text_vs_base_delta",
    "text_ece",
    "30k_vs_10k_delta",
]


def write_report(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def smoke_run_report(losses: list[float], steps: int = 40) -> dict:
    return {"status": "passed", "exit_code": 0, "config": {"steps": steps}, "losses": losses}


def smoke_selfacc_report(accuracy: float = 0.99) -> dict:
    # selfacc 的 repeat_identical 不在 s1_gate 契约内，故意给 False 证明不拦截。
    return {
        "status": "passed",
        "exit_code": 0,
        "metrics": {"product_accuracy": accuracy, "repeat_identical": False},
    }


def dev_eval_report(accuracy: float, ece15: float = 0.02, **overrides: object) -> dict:
    report: dict = {
        "status": "passed",
        "exit_code": 0,
        "repeat_identical": True,
        "samples": 500,
        "passes": 2,
        "metrics": {"product_accuracy": accuracy, "ece15": ece15},
    }
    report.update(overrides)
    return report


def write_inputs(
    tmp_path: Path, *, losses: list[float] | None = None
) -> tuple[Path, Path, Path, Path]:
    return (
        write_report(
            tmp_path / "smoke-run.json",
            smoke_run_report(losses if losses is not None else [0.9, 0.7, 0.5]),
        ),
        write_report(tmp_path / "smoke-selfacc.json", smoke_selfacc_report()),
        write_report(tmp_path / "dev-30k.json", dev_eval_report(0.97)),
        write_report(tmp_path / "dev-10k.json", dev_eval_report(0.94)),
    )


def argv_for(
    smoke_run: Path, smoke_selfacc: Path, dev_30k: Path, dev_10k: Path, out: Path
) -> list[str]:
    return [
        "--smoke-run",
        str(smoke_run),
        "--smoke-selfacc",
        str(smoke_selfacc),
        "--dev-30k",
        str(dev_30k),
        "--dev-10k",
        str(dev_10k),
        "--output",
        str(out),
    ]


def test_assembles_seven_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 0
    text = out.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    metrics = json.loads(text)
    assert list(metrics) == METRIC_KEYS
    assert metrics["smoke_steps"] == pytest.approx(40)
    assert metrics["smoke_accuracy"] == pytest.approx(0.99)
    assert metrics["loss_decreasing"] == 1.0
    assert metrics["text_vs_onejev_delta"] == pytest.approx(0.97 - ONEJEV4_TEXT_DEV500)
    assert metrics["text_vs_base_delta"] == pytest.approx(0.97 - E4B_TEXT_DEV500)
    assert metrics["text_ece"] == pytest.approx(0.02)
    assert metrics["30k_vs_10k_delta"] == pytest.approx(0.97 - 0.94)
    assert json.loads(capsys.readouterr().out) == metrics


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ([0.9, 0.7, 0.5], 1.0),
        ([0.9, 0.7, 0.7, 0.5], 1.0),
        ([0.1, 0.1, 0.9], 0.0),
        ([0.9, float("nan"), 0.5], 0.0),
    ],
)
def test_loss_decreasing_uses_window_means(
    tmp_path: Path, losses: list[float], expected: float
) -> None:
    paths = write_inputs(tmp_path, losses=losses)
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["loss_decreasing"] == expected


@pytest.mark.parametrize("overrides", [{"status": "blocked"}, {"exit_code": 1}])
def test_rejects_unhealthy_report_without_output(tmp_path: Path, overrides: dict) -> None:
    paths = write_inputs(tmp_path)
    broken = write_report(tmp_path / "dev-30k-bad.json", dev_eval_report(0.97, **overrides))
    out = tmp_path / "metrics.json"
    assert main(argv_for(paths[0], paths[1], broken, paths[3], out)) == 1
    assert not out.exists()


def test_rejects_missing_report_without_output(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    assert main(argv_for(tmp_path / "missing.json", *paths[1:], out)) == 1
    assert not out.exists()


def test_cli_exit_code_one_names_the_bad_report(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    argv = argv_for(paths[0], paths[1], tmp_path / "gone.json", paths[3], out)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.s1_gate", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "gone.json" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    ("which", "overrides"),
    [
        (2, {"repeat_identical": False}),
        (2, {"samples": 499}),
        (2, {"passes": 1}),
        (3, {"repeat_identical": False}),
        (3, {"samples": 100}),
    ],
)
def test_rejects_dev_reports_off_protocol(tmp_path: Path, which: int, overrides: dict) -> None:
    paths = list(write_inputs(tmp_path))
    paths[which] = write_report(tmp_path / "dev-bad.json", dev_eval_report(0.97, **overrides))
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 1
    assert not out.exists()


def test_baseline_overrides_change_deltas(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    argv = [
        *argv_for(*paths, out),
        "--baseline-onejev4",
        "0.90",
        "--baseline-e4b",
        "0.80",
    ]
    assert main(argv) == 0
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["text_vs_onejev_delta"] == pytest.approx(0.97 - 0.90)
    assert metrics["text_vs_base_delta"] == pytest.approx(0.97 - 0.80)
