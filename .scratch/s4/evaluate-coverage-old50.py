"""Four short old-modality diagnostics; historical thresholds do not reject expansion."""

import importlib.util
import json
import os
import time
from pathlib import Path

SOURCE = Path("/root/autodl-tmp/s4-monitored-eval-code-001")
os.chdir(SOURCE)
spec = importlib.util.spec_from_file_location(
    "screen", SOURCE / ".scratch/s4/screen-crossmodal-retention.py"
)
assert spec and spec.loader
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
root = screen.ROOT
adapter = root / "artifacts/s4-crossmodal-retention-train-001/adapter"
assert (
    screen.sha(adapter / "adapter_model.safetensors")
    == "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818"
)
output = root / "artifacts/s4-coverage-old50-001"
output.mkdir(exist_ok=False)
prior = root / "artifacts/s4-pilot-five-screen-001"
manifest, commands = screen.read(prior / "manifest.json"), screen.read(prior / "commands.json")
reference = {"audio": 37, "image": 40, "screenshot": 50, "text": 49}
receipt = {
    "status": "running",
    "results": {},
    "stage_gate": False,
    "scope": "fixed dev50, one pass, 180s per cohort; stop at >10pp regression for diagnosis",
    "source_commit": "74e96e2",
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
    for name in ["audio", "image", "screenshot", "text"]:
        receipt["active_stage"] = name
        save()
        stage = output / name
        stage.mkdir()
        source = next(s for s in manifest["sources"] if s["cohort"] == name)
        command = dict(next(c for c in commands if c["source"] == name), timeout_seconds=180)
        item = screen.ordinary(name, adapter, stage, source, command)
        item["first_correct"] = reference[name]
        item["delta_correct"] = item["correct"] - reference[name]
        receipt["results"][name] = item
        save()
        if item["delta_correct"] < -5:
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
