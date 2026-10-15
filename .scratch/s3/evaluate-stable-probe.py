"""Bounded S3 candidate screening; never substitutes for the full S3 stage gate."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
model = (
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/"
    "ee0ef6023621cff504d758262d4e04895a5af4a2"
)
train_path = root / "reports/s3-stable-probe-train-001.json"
train = json.loads(train_path.read_text())
assert train["status"] == "completed" and train["exit_code"] == 0
adapter = root / "artifacts/s3-stable-probe-train-001/adapter"
assert (
    hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    == train["adapter_sha256"]
)
reference = json.loads((root / "reports/s3-video150-reference-001.json").read_text())
previous_probe = json.loads((root / "reports/s3-quality-probe-summary-001.json").read_text())
results = {}
summary_path = root / "reports/s3-stable-probe-screening-001.json"
assert not summary_path.exists()
qualified = True
for cohort in ["video", "audio", "image", "screenshot", "text"]:
    if cohort == "video":
        data = Path(reference["data_path"])
        expected_sha = reference["data_sha256"]
        media_root = (
            "/root/autodl-tmp/s0-media-development-002/artifacts/media-dev-002/video-a/media"
        )
        minimum = reference["expansion_required_correct"]
    else:
        source = previous_probe["sources"][cohort]
        data = Path(source["data"])
        expected_sha = source["sha256"]
        media_root = source["media_root"]
        minimum = 18  # ceil(50 * .38 * .9); screening only.
        if cohort in {"image", "screenshot", "text"}:
            baseline = json.loads(
                (root / f"reports/s2-dev-eval-{cohort}-v2-budget1120-001.json").read_text()
            )
            assert baseline["adapter_sha256"] == reference["adapter_sha256"]
            args = baseline["command"]
            sidecar = Path(args[args.index("--output-dir") + 1]) / "records.jsonl"
            assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == baseline["sidecar_sha256"]
            ids = {row["id"] for row in map(json.loads, data.read_text().splitlines())}
            original = [
                r
                for r in map(json.loads, sidecar.read_text().splitlines())
                if r["pass"] == 0 and r["variant"] == "original" and r["id"] in ids
            ]
            assert len(original) == len(ids) == 50
            minimum = sum(r["prediction"] == r["target"] for r in original) - 1
    assert hashlib.sha256(data.read_bytes()).hexdigest() == expected_sha
    report_path = root / f"reports/s3-stable-{cohort}-screen-001.json"
    command = [
        python,
        "-u",
        "-m",
        "scripts.s1_dev_eval",
        "--model-dir",
        model,
        "--adapter",
        str(adapter),
        "--data",
        str(data),
        "--passes",
        "1",
        "--output-dir",
        str(root / f"artifacts/s3-stable-{cohort}-screen-001"),
        "--report",
        str(report_path),
    ]
    if media_root:
        command += ["--media-root", str(media_root)]
    if cohort in {"image", "screenshot"}:
        command += ["--image-max-soft-tokens", "1120"]
    subprocess.run(command, cwd=root, check=True, timeout=300)
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["adapter_sha256"] == train["adapter_sha256"]
    assert report["data_sha256"] == expected_sha
    correct = round(report["metrics"]["product_accuracy"] * report["samples"])
    passed = correct >= minimum
    results[cohort] = {
        "correct": correct,
        "count": report["samples"],
        "required_correct": minimum,
        "screen_passed": passed,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    qualified = qualified and passed
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed" if not passed or cohort == "text" else "running",
                "qualified_for_expansion": qualified if not passed or cohort == "text" else None,
                "scope": "single-pass small development screen, not stage acceptance",
                "training_report_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    if not passed:
        break
print(summary_path.read_text())
