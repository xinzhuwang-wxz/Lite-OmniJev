"""CPU guard tests for root-only first10 launch and paired DEV screening."""

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".scratch/s4"


def load(filename):
    spec = importlib.util.spec_from_file_location(filename, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(module):
    return {
        "decision": "approve_first10",
        "initializer_sha256": module.INIT_SHA,
        "prepared_report_sha256": module.PREPARED_SHA,
        "config_sha256": module.CONFIG_SHA,
        "baseline_report_sha256": module.BASELINE_SHA,
        "control_training_report_sha256": module.CE_REPORT_SHA,
        "max_steps": 10,
        "max_seconds": 240,
    }


def test_explicit_new_objective_decision_required():
    module = load("launch-video-pair-objective-pilot.py")
    module.validate_decision(decision(module))
    for changed in [
        {"max_seconds": 360},
        {"max_steps": 100},
        {"config_sha256": "CE"},
        {"initializer_sha256": "trained CE"},
        {"decision": "baseline_only"},
        {"parity_validated": True},
    ]:
        with pytest.raises(ValueError):
            module.validate_decision(decision(module) | changed)


def test_actual_training_command_and_runtime_snapshot_share_exact_output_directory(tmp_path):
    # Evaluate only inert path/command expressions, never the launcher's main or subprocesses.
    tree = ast.parse((SCRIPTS / "launch-video-pair-objective-pilot.py").read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    expressions = {
        node.targets[0].id: node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"output", "common", "runtime_path", "command"}
    }
    context = {
        "root": tmp_path,
        "pilot_dir": tmp_path / "prepared",
        "prepared_dir": tmp_path / "data",
        "python": "python",
        "chosen_parent": {"path": "initializer", "weights_sha256": "a" * 64},
    }
    for name in ["output", "common", "runtime_path", "command"]:
        context[name] = eval(
            compile(ast.Expression(expressions[name]), "inert_command", "eval"), context
        )
    common, command = context["common"], context["command"]
    assert Path(common[common.index("--output-dir") + 1]) == context["runtime_path"].parent
    assert (
        context["runtime_path"]
        == tmp_path / "artifacts/s4-video-pair-objective-pilot-train-001/run.json"
    )
    assert command[:4] == ["timeout", "--signal=INT", "--kill-after=20s", "220"]
    assert command[command.index("--invocation-steps") + 1] == "10"
    assert "--resume-from" not in command


@pytest.mark.parametrize(
    "accuracy,brier,expected",
    [
        (0.72, 0.44, True),
        (0.70, 0.40, False),
        (0.72, 0.4401167682281881, False),
        (0.74, 0.45, False),
        (float("nan"), 0.4, False),
        (0.72, float("inf"), False),
    ],
)
def test_screen_requires36_and_brier_better_than_actual_ce(accuracy, brier, expected):
    evaluator = load("evaluate-video-pair-objective-dev50.py")
    assert evaluator.passes_screen({"count": 50, "accuracy": accuracy, "brier": brier}) is expected


def test_candidate_cannot_be_old_ce_or_different_data():
    evaluator = load("evaluate-video-pair-objective-dev50.py")
    trained = {
        "status": "paused",
        "exit_code": 0,
        "steps_completed": 10,
        "data_sha256": evaluator.DATA_SHA,
        "init_adapter": {
            "weights_sha256": evaluator.helper.INIT_SHA,
            "config_sha256": evaluator.helper.CONFIG_SHA,
        },
        "config": {
            "rank": 64,
            "alpha": 128,
            "sample_count": 1200,
            "steps": 100,
            "accumulation": 12,
            "seed": 42,
            "learning_rate": 2e-5,
            "video_num_frames": 16,
            "image_max_soft_tokens": 1120,
            "video_pair_loss_weight": 0.1,
            "video_pair_sidecar_sha256": evaluator.SIDECAR_SHA,
            "midway_step": 10,
            "checkpoint_every": 10,
        },
        "video_pairs": {"pairs": 300},
        "invocation": {"next_draw": 120, "global_steps_target": 100},
    }
    launch = {
        "status": "paused",
        "exit_code": 0,
        "config_sha256": evaluator.CONFIG_SHA,
        "policy_sha256": evaluator.POLICY_SHA,
    }
    evaluator.validate_training(trained, launch)
    with pytest.raises(ValueError):
        evaluator.validate_training(trained | {"data_sha256": "new data"}, launch)
    with pytest.raises(ValueError):
        evaluator.validate_training(
            trained | {"config": trained["config"] | {"video_pair_loss_weight": 0}}, launch
        )
