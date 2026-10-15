import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.s2_gate import (
    FROZEN_DEV_SHA256,
    ONEJEV4_IMAGE_DEV500,
    ONEJEV4_SCREENSHOT_DEV500,
    S1_TEXT_DEV500_ADAPTER,
    main,
)

METRIC_KEYS = [
    "image_vs_onejev_delta",
    "screenshot_vs_onejev_delta",
    "text_regression",
    "counterfactual_flip_rate",
    "image_accuracy",
    "screenshot_accuracy",
    "text_accuracy",
    "image_ece15",
    "screenshot_ece15",
    "image_cf_match",
    "screenshot_cf_match",
]


def write_report(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def smoke_run_report(**overrides: object) -> dict:
    report: dict = {"status": "completed", "exit_code": 0, "config": {"steps": 300}}
    report.update(overrides)
    return report


def dev_eval_report(
    accuracy: float,
    ece15: float | None = 0.02,
    cf_match: float | None = 0.9,
    cohort: str = "image",
    **overrides: object,
) -> dict:
    # cf_match 评测未覆盖时报告里是 null，脚本必须按无效报告拒绝。
    report: dict = {
        "status": "passed",
        "exit_code": 0,
        "model_revision": "a" * 40,
        "adapter_sha256": "b" * 64,
        "data_sha256": FROZEN_DEV_SHA256[cohort],
        "repeat_identical": True,
        "samples": 500,
        "passes": 2,
        "metrics": {
            "product_accuracy": accuracy,
            "ece15": ece15,
            "counterfactual_content_match_rate": cf_match,
        },
    }
    report.update(overrides)
    return report


def write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return (
        write_report(tmp_path / "image.json", dev_eval_report(0.82, ece15=0.03, cf_match=0.85)),
        write_report(
            tmp_path / "screenshot.json",
            dev_eval_report(0.70, ece15=0.04, cf_match=0.80, cohort="screenshot"),
        ),
        write_report(tmp_path / "text.json", dev_eval_report(0.99, cohort="text")),
        write_report(tmp_path / "smoke-run.json", smoke_run_report()),
    )


def argv_for(image: Path, screenshot: Path, text: Path, smoke_run: Path, out: Path) -> list[str]:
    return [
        "--image",
        str(image),
        "--screenshot",
        str(screenshot),
        "--text",
        str(text),
        "--smoke-run",
        str(smoke_run),
        "--output",
        str(out),
    ]


def test_assembles_all_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 0
    text = out.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    metrics = json.loads(text)
    assert list(metrics) == METRIC_KEYS
    assert metrics["image_vs_onejev_delta"] == pytest.approx(0.82 - ONEJEV4_IMAGE_DEV500)
    assert metrics["screenshot_vs_onejev_delta"] == pytest.approx(0.70 - ONEJEV4_SCREENSHOT_DEV500)
    assert metrics["text_regression"] == pytest.approx(S1_TEXT_DEV500_ADAPTER - 0.99)
    assert metrics["counterfactual_flip_rate"] == pytest.approx(0.80)
    assert metrics["image_accuracy"] == pytest.approx(0.82)
    assert metrics["screenshot_accuracy"] == pytest.approx(0.70)
    assert metrics["text_accuracy"] == pytest.approx(0.99)
    assert metrics["image_ece15"] == pytest.approx(0.03)
    assert metrics["screenshot_ece15"] == pytest.approx(0.04)
    assert metrics["image_cf_match"] == pytest.approx(0.85)
    assert metrics["screenshot_cf_match"] == pytest.approx(0.80)
    assert json.loads(capsys.readouterr().out) == metrics


@pytest.mark.parametrize(
    ("which", "overrides"),
    [
        (0, {"status": "blocked"}),
        (0, {"exit_code": 1}),
        (0, {"repeat_identical": False}),
        (0, {"samples": 499}),
        (0, {"passes": 1}),
        (1, {"status": "running"}),
        (1, {"exit_code": 2}),
        (1, {"repeat_identical": False}),
        (1, {"samples": 100}),
        (2, {"status": "blocked"}),
        (2, {"exit_code": 1}),
        (2, {"passes": 1}),
        (3, {"status": "failed"}),
        (3, {"exit_code": 1}),
        (3, {"config": {}}),
    ],
)
def test_rejects_unhealthy_or_off_protocol_report_without_output(
    tmp_path: Path, which: int, overrides: dict
) -> None:
    paths = list(write_inputs(tmp_path))
    if which == 3:
        bad = smoke_run_report(**overrides)
    else:
        bad = dev_eval_report(0.82, 0.03, 0.85, **overrides)
    paths[which] = write_report(tmp_path / "bad.json", bad)
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 1
    assert not out.exists()


def test_rejects_missing_report_without_output(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    assert main(argv_for(tmp_path / "missing.json", *paths[1:], out)) == 1
    assert not out.exists()


def test_cli_exit_code_one_names_the_bad_report(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    argv = argv_for(paths[0], tmp_path / "gone.json", paths[2], paths[3], out)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.s2_gate", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "gone.json" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("which", [0, 1])
def test_rejects_null_counterfactual_match_rate(tmp_path: Path, which: int) -> None:
    paths = list(write_inputs(tmp_path))
    accuracy = [0.82, 0.70, 0.99][which]
    paths[which] = write_report(tmp_path / "null-cf.json", dev_eval_report(accuracy, cf_match=None))
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 1
    assert not out.exists()


@pytest.mark.parametrize(
    ("image_cf", "screenshot_cf", "expected"),
    [
        (0.85, 0.80, 0.80),
        (0.70, 0.99, 0.70),
        (0.9, 0.9, 0.9),
    ],
)
def test_counterfactual_flip_rate_takes_the_weaker_modality(
    tmp_path: Path, image_cf: float, screenshot_cf: float, expected: float
) -> None:
    paths = list(write_inputs(tmp_path))
    paths[0] = write_report(tmp_path / "image.json", dev_eval_report(0.82, 0.03, image_cf))
    paths[1] = write_report(
        tmp_path / "screenshot.json",
        dev_eval_report(0.70, 0.04, screenshot_cf, cohort="screenshot"),
    )
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 0
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["counterfactual_flip_rate"] == pytest.approx(expected)


@pytest.mark.parametrize("accuracy", [0.98, 1.0])
def test_text_regression_direction(tmp_path: Path, accuracy: float) -> None:
    paths = list(write_inputs(tmp_path))
    paths[2] = write_report(tmp_path / "text.json", dev_eval_report(accuracy, cohort="text"))
    out = tmp_path / "metrics.json"
    assert main(argv_for(*paths, out)) == 0
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["text_regression"] == pytest.approx(S1_TEXT_DEV500_ADAPTER - accuracy)


def test_baseline_overrides_change_deltas(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path)
    out = tmp_path / "metrics.json"
    argv = [
        *argv_for(*paths, out),
        "--baseline-onejev4-image",
        "0.80",
        "--baseline-onejev4-screenshot",
        "0.60",
        "--baseline-s1-text",
        "0.95",
    ]
    assert main(argv) == 0
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert metrics["image_vs_onejev_delta"] == pytest.approx(0.82 - 0.80)
    assert metrics["screenshot_vs_onejev_delta"] == pytest.approx(0.70 - 0.60)
    assert metrics["text_regression"] == pytest.approx(0.95 - 0.99)


@pytest.mark.parametrize(
    "change",
    [
        {"adapter_sha256": "c" * 64},
        {"model_revision": "c" * 40},
        {"data_sha256": "c" * 64},
        {"research_input_profile": {}},
        {"image_max_soft_tokens": {"requested": 1120, "effective": 280}},
        {"image_max_soft_tokens": {"requested": 1120, "effective": 1120}},
    ],
)
def test_rejects_incompatible_candidate_evidence(tmp_path, change):
    paths = write_inputs(tmp_path)
    value = json.loads(paths[0].read_text())
    value.update(change)
    write_report(paths[0], value)
    output = tmp_path / "metrics.json"
    assert main(argv_for(*paths, output)) == 1
    assert not output.exists()


def test_matching_explicit_image_profiles_accepted(tmp_path):
    paths = write_inputs(tmp_path)
    for path in paths[:2]:
        value = json.loads(path.read_text())
        value["image_max_soft_tokens"] = {"requested": 1120, "effective": 1120}
        write_report(path, value)
    assert main(argv_for(*paths, tmp_path / "metrics.json")) == 0
