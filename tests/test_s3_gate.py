"""S3 gate input behavior: regression, same-version evidence and frame controls."""

import json
from pathlib import Path

import pytest

from scripts.s2_gate import GateInputError
from scripts.s3_gate import FROZEN_S3_DEV_SHA256, assemble_metrics


def reports(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path], Path]:
    current, previous = {}, {}
    for stage, destination, scores in [
        (
            "s3",
            current,
            {"audio": 0.38, "video": 0.698, "image": 0.80, "screenshot": 0.95, "text": 0.99},
        ),
        ("s2", previous, {"image": 0.806, "screenshot": 0.96, "text": 0.994}),
    ]:
        for name, score in scores.items():
            path = tmp_path / f"{stage}-{name}.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "exit_code": 0,
                        "samples": 500,
                        "passes": 2,
                        "repeat_identical": True,
                        "model_revision": "a" * 40,
                        "adapter_sha256": ("3" if stage == "s3" else "2") * 64,
                        "data_sha256": FROZEN_S3_DEV_SHA256[name],
                        "video_num_frames": 16,
                        "metrics": {"product_accuracy": score},
                    }
                )
            )
            destination[name] = path
    control = tmp_path / "video8.json"
    data = json.loads(current["video"].read_text())
    data.update(video_num_frames=8, passes=1, repeat_identical=None)
    control.write_text(json.dumps(data))
    return current, previous, control


def test_same_condition_s3_metrics(tmp_path: Path) -> None:
    result = assemble_metrics(*reports(tmp_path))
    assert result["audio_teacher_ratio"] == 1
    assert result["video_teacher_ratio"] == 1
    assert result["old_modalities_max_regression"] == pytest.approx(0.01)
    assert result["frame_ab_complete"] == 1


@pytest.mark.parametrize(
    "which,key,value",
    [
        ("current", "adapter_sha256", "other"),
        ("previous", "adapter_sha256", "other"),
        ("current", "data_sha256", "different-dev"),
        ("current", "repeat_identical", False),
        ("current", "samples", 150),
        ("current", "passes", 1),
        ("control", "adapter_sha256", "other"),
        ("control", "data_sha256", "other"),
        ("control", "video_num_frames", 16),
        ("control", "samples", 10),
        ("control", "passes", 2),
    ],
)
def test_rejects_incomparable_or_partial_evidence(tmp_path: Path, which, key, value) -> None:
    current, previous, control = reports(tmp_path)
    target = {"current": current["image"], "previous": previous["image"], "control": control}[which]
    data = json.loads(target.read_text())
    data[key] = value
    target.write_text(json.dumps(data))
    with pytest.raises(GateInputError):
        assemble_metrics(current, previous, control)


@pytest.mark.parametrize("value", [True, None, float("nan"), 1.1, -0.1])
def test_rejects_invalid_accuracy(tmp_path: Path, value) -> None:
    current, previous, control = reports(tmp_path)
    data = json.loads(current["audio"].read_text())
    data["metrics"]["product_accuracy"] = value
    current["audio"].write_text(json.dumps(data))
    with pytest.raises(GateInputError):
        assemble_metrics(current, previous, control)


def test_cli_writes_metrics_for_gate(tmp_path: Path) -> None:
    from scripts.s3_gate import main

    current, previous, control = reports(tmp_path)
    output = tmp_path / "metrics.json"
    args = []
    for name, path in current.items():
        args.extend([f"--{name}", str(path)])
    for name, path in previous.items():
        args.extend([f"--s2-{name}", str(path)])
    assert main([*args, "--video8", str(control), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["frame_ab_complete"] == 1


def test_regression_rejects_changed_image_input_budget(tmp_path):
    current, previous, control = reports(tmp_path)
    data = json.loads(current["image"].read_text())
    data["image_max_soft_tokens"] = {"requested": 1120, "effective": 1120}
    current["image"].write_text(json.dumps(data))
    with pytest.raises(GateInputError, match="budgets must match"):
        assemble_metrics(current, previous, control)


@pytest.mark.parametrize("cohort", ["audio", "video", "image", "screenshot", "text"])
@pytest.mark.parametrize(
    "change",
    [
        {"data_sha256": None},
        {"data_sha256": "c" * 64},
        {"research_input_profile": {}},
        {"adapter_sha256": None},
    ],
)
def test_rejects_unbound_or_research_stage_reports(tmp_path, cohort, change):
    current, previous, control = reports(tmp_path)
    row = json.loads(current[cohort].read_text())
    row.update(change)
    current[cohort].write_text(json.dumps(row))
    with pytest.raises(GateInputError):
        assemble_metrics(current, previous, control)
