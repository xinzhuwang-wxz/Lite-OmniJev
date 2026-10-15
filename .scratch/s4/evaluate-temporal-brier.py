"""Short CE control retention diagnostics; not stage acceptance."""

import importlib.util
import json
import os
import time
from pathlib import Path

SOURCE = Path("/root/autodl-tmp/s4-monitored-eval-code-001")
os.chdir(SOURCE)
os.sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location(
    "screen", SOURCE / ".scratch/s4/screen-crossmodal-retention.py"
)
assert spec and spec.loader
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
root = screen.ROOT
adapter = root / "artifacts/s4-temporal-brier-train-001/adapter"
training_path = root / "reports/s4-temporal-brier-training-001.json"
training = screen.read(training_path)
assert (
    training["status"] == "completed"
    and training["exit_code"] == 0
    and training["steps_completed"] == 10
)
assert training["config"]["sample_count"] == 120 and training["config"]["accumulation"] == 12
assert training["config"].get("pair_loss_weight", 0) == 0
assert training["config"]["brier_weight"] == 0.5
control = screen.read(root / "reports/s4-temporal-curriculum-training-001.json")
assert training["data_sha256"] == control["data_sha256"]
assert {k: v for k, v in training["config"].items() if k != "brier_weight"} == {
    k: v for k, v in control["config"].items() if k != "brier_weight"
}
assert (
    training["init_adapter"]["weights_sha256"]
    == "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"
)
assert screen.sha(adapter / "adapter_model.safetensors") == training["adapter_sha256"]

output = root / "artifacts/s4-temporal-brier-video150-001"
output.mkdir(exist_ok=False)
prior = root / "artifacts/s4-pilot-five-screen-001"
manifest, commands = screen.read(prior / "manifest.json"), screen.read(prior / "commands.json")
reference = {"video": 110, "audio": 39, "image": 40, "screenshot": 50, "text": 49}
receipt = {
    "status": "running",
    "results": {},
    "stage_gate": False,
    "scope": (
        "fixed video150/other50, one pass, 180s per cohort; >10pp regression stops for diagnosis"
    ),
    "source_commit": "74e96e2",
    "training_report_sha256": screen.sha(training_path),
    "source_archive_sha256": "b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516",
    "manifest_sha256": screen.sha(prior / "manifest.json"),
    "commands_sha256": screen.sha(prior / "commands.json"),
}
start = time.monotonic()


def save():
    receipt["wall_seconds"] = time.monotonic() - start
    temporary = output / "receipt.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(output / "receipt.json")


try:
    for name in ["video"]:
        receipt["active_stage"] = name
        save()
        stage = output / name
        stage.mkdir()
        source = next(s for s in manifest["sources"] if s["cohort"] == name)
        command = dict(next(c for c in commands if c["source"] == name), timeout_seconds=180)
        item = screen.ordinary(name, adapter, stage, source, command)
        item["matched_ce_correct"] = reference[name]
        item["delta_correct"] = item["correct"] - reference[name]
        receipt["results"][name] = item
        save()
        if item["delta_correct"] < -0.1 * item["count"]:
            receipt.update(status="stopped_for_regression_review", exit_code=2)
            break
    else:
        receipt.update(status="completed", exit_code=0)
except Exception as error:
    receipt.update(status="execution_failed", exit_code=1, error=f"{type(error).__name__}: {error}")
finally:
    save()
    print(json.dumps(receipt), flush=True)
raise SystemExit(receipt["exit_code"])
