"""Bounded official video140 preprocessing probe; same retained adapter and dev150."""

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import transformers

from scripts import s1_dev_eval

root = Path("/root/autodl-tmp/s1-text-lora-001")
reference_path = root / "artifacts/s4-pilot-five-screen-001/video-report.json"
reference = json.loads(reference_path.read_text())
command = reference["command"]
data = Path(command[command.index("--data") + 1])
media = Path(command[command.index("--media-root") + 1])
assert hashlib.sha256(data.read_bytes()).hexdigest() == reference["data_sha256"]
adapter = root / "artifacts/s4-joint-pilot-train-001/adapter"
assert (
    hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    == reference["adapter_sha256"]
)
output = root / "artifacts/s4-video140-probe-001"
report = root / "reports/s4-video140-probe-001.json"
original = transformers.AutoProcessor.from_pretrained


class VideoBudget:
    def __init__(self, processor):
        self.processor = processor

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, **kwargs):
        if "videos" in kwargs:
            kwargs["videos_kwargs"] = {**kwargs.get("videos_kwargs", {}), "max_soft_tokens": 140}
        return self.processor(**kwargs)


def factory(*args, **kwargs):
    return VideoBudget(original(*args, **kwargs))


start = time.monotonic()
with patch.object(transformers.AutoProcessor, "from_pretrained", side_effect=factory):
    exit_code = s1_dev_eval.run(
        model_dir=Path(command[command.index("--model-dir") + 1]),
        adapter_dir=adapter,
        data_path=data,
        output_dir=output,
        report_out=report,
        max_length=4096,
        passes=1,
        command=["probe-video140.py", "--official-video-soft-tokens", "140"],
        media_root=media,
        video_frames=16,
    )
result = {
    "scope": "fixed adapter/dev150/16 frames; only video140 changes; not stage acceptance",
    "exit_code": exit_code,
    "video_soft_tokens": 140,
    "reference_video_soft_tokens": 70,
    "reference_report_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "elapsed_seconds": time.monotonic() - start,
    "cost": None,
    "reference_correct": 99,
    "required_screen_correct": 102,
    "maximum_wall_seconds": 600,
}
if exit_code == 0:
    actual = json.loads(report.read_text())
    assert (
        actual["data_sha256"] == reference["data_sha256"]
        and actual["adapter_sha256"] == reference["adapter_sha256"]
    )
    correct = round(actual["metrics"]["product_accuracy"] * actual["samples"])
    result.update(
        correct=correct,
        count=actual["samples"],
        screen_pass=correct >= 102,
        native_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        baseline_job_wall_seconds=reference["timing"]["duration_seconds"],
    )
with (root / "reports/s4-video140-probe-execution-001.json").open("x") as h:
    json.dump(result, h, indent=2)
print(json.dumps(result))
raise SystemExit(exit_code)
