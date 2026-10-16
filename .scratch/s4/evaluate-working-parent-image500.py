"""Measure the working coverage parent on the complete original public image cohort."""

import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/s4-monitored-eval-code-001")
python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert sha(Path("/root/autodl-tmp/omnijev-s4-monitored-eval-001.tar")) == (
    "b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516"
)
previous_path = root / "reports/s3-dev-eval-image-stable-001.json"
assert sha(previous_path) == "2539b353fb2f4e499d3609ad7b9a366f467ec381fe46d2b3bac0295ea4405690"
previous = json.loads(previous_path.read_bytes())
assert previous["image_max_soft_tokens"] == {"requested": 1120, "effective": 1120}
assert previous["status"] == "passed" and previous["samples"] == 500
assert previous["data_sha256"] == "38efa78efe62ebdd8523c7372b8c0ac87a9030a18d429db488f44e5aad08e7de"
prior_command = previous["command"]


def argument(flag):
    return prior_command[prior_command.index(flag) + 1]


data, media, model = [Path(argument(f)) for f in ["--data", "--media-root", "--model-dir"]]
assert sha(data) == previous["data_sha256"]
trained_path = root / "reports/s4-temporal-coverage-training-001.json"
assert sha(trained_path) == "0e29da0e72cba7bcf232593f3babb681f7581af794927398ab6a1b0c47d5b309"
trained = json.loads(trained_path.read_bytes())
assert trained["status"] == "completed" and trained["steps_completed"] == 10
adapter = root / "artifacts/s4-temporal-coverage-train-001/adapter"
assert (
    sha(adapter / "adapter_model.safetensors")
    == trained["adapter_sha256"]
    == ("f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3")
)
assert model.name == previous["model_revision"] == trained["model_revision"]
model_hashes = {}
for name, expected in trained["asset_hashes"].items():
    actual = sha(model / name)
    assert actual == expected, "base model asset changed: " + name
    model_hashes[name] = actual
source_hashes = {}
with tarfile.open("/root/autodl-tmp/omnijev-s4-monitored-eval-001.tar") as archive:
    for name in [
        "scripts/s1_dev_eval.py",
        "train/template.py",
        "train/readout.py",
        "train/trainer.py",
        "train/multimodal.py",
        "scripts/check_tokens.py",
    ]:
        members = [m for m in archive.getmembers() if m.name == name or m.name.endswith("/" + name)]
        assert len(members) == 1
        original = archive.extractfile(members[0])
        assert original is not None
        expected = hashlib.file_digest(original, "sha256").hexdigest()
        assert sha(source / name) == expected, "static source changed: " + name
        source_hashes[name] = expected
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
).strip()
out = root / "artifacts/s4-working-parent-image500-001"
out.mkdir(exist_ok=False)
command = [
    "timeout",
    "--signal=INT",
    "--kill-after=20s",
    "420",
    python,
    "-m",
    "scripts.s1_dev_eval",
    "--model-dir",
    str(model),
    "--adapter",
    str(adapter),
    "--data",
    str(data),
    "--passes",
    "1",
    "--image-max-soft-tokens",
    "1120",
    "--output-dir",
    str(out / "eval"),
    "--report",
    str(out / "report.json"),
    "--media-root",
    str(media),
]
receipt = {
    "status": "running",
    "stage_gate": False,
    "budget_seconds": 420,
    "scope": "Complete original public DEV image500, one pass; no holdout/calibration",
    "source_commit": "74e96e2",
    "source_archive_sha256": "b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516",
    "data_sha256": sha(data),
    "training_report_sha256": sha(trained_path),
    "source_protocol_sha256": sha(previous_path),
    "adapter_sha256": trained["adapter_sha256"],
    "command": command,
    "runner_sha256": sha(Path(__file__)),
    "cost_usd": None,
    "model_asset_sha256": model_hashes,
    "source_file_sha256": source_hashes,
    "model_quality_status": "not_evaluated",
    "baseline_metrics": previous["metrics"],
    "image_max_soft_tokens": 1120,
    "passes": 1,
    "progress_every": 25,
}
(out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
started = time.monotonic()
try:
    with (out / "run.log").open("x") as log:
        result = subprocess.run(
            command,
            cwd=source,
            env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    receipt["exit_code"] = result.returncode
    assert result.returncode == 0, "Full-cohort evaluator failed or exceeded budget"
    report = json.loads((out / "report.json").read_bytes())
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["samples"] == 500 and report["passes"] == 1
    assert report["image_max_soft_tokens"] == {"requested": 1120, "effective": 1120}
    assert report["data_sha256"] == sha(data) == receipt["data_sha256"]
    assert report["adapter_sha256"] == trained["adapter_sha256"]
    receipt.update(
        status="completed", metrics=report["metrics"], report_sha256=sha(out / "report.json")
    )
except Exception as error:
    receipt.update(status="failed", exit_code=receipt.get("exit_code") or 1, error=str(error))
finally:
    receipt["wall_seconds"] = time.monotonic() - started
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
raise SystemExit(receipt["exit_code"])
