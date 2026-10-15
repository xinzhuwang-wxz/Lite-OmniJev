"""Small same-condition S3 routing probe; never a substitute for dev500 gates."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
py = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
model = (
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef60236"
    "21cff504d758262d4e04895a5af4a2"
)
output = root / "artifacts/s3-quality-probe-001"
output.mkdir(exist_ok=False)
sources = {}
baseline = {}
for cohort in ["image", "screenshot", "text", "audio", "video"]:
    if cohort in ["image", "screenshot", "text"]:
        report = json.loads(
            (root / f"reports/s2-dev-eval-{cohort}-v2-corrected-001.json").read_text()
        )
        args = report["command"]
        source = Path(args[args.index("--data") + 1])
        media = args[args.index("--media-root") + 1] if "--media-root" in args else None
        expected = report["data_sha256"]
    else:
        report = json.loads(
            (root / f"reports/s0-development-{cohort}-gemma4-e4b-it-001.json").read_text()
        )
        source = Path(report["dev_jsonl"]["path"])
        media = report["media_root"]
        expected = report["dev_jsonl"]["sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    rows = list(map(json.loads, source.read_text().splitlines()))
    assert len(rows) == 500 and all(r["meta"]["pool"] == "dev" for r in rows)
    selected = sorted(
        rows, key=lambda r: hashlib.sha256(("s3-quality-probe:42:" + r["id"]).encode()).hexdigest()
    )[:50]
    data = output / f"{cohort}-50.jsonl"
    data.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in selected))
    sources[cohort] = {
        "data": str(data),
        "media_root": media,
        "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "parent_sha256": expected,
    }
    if cohort in ["image", "screenshot", "text"]:
        sidecar = Path(args[args.index("--output-dir") + 1]) / "records.jsonl"
        assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == report["sidecar_sha256"]
        ids = {r["id"] for r in selected}
        records = [
            r
            for r in map(json.loads, sidecar.read_text().splitlines())
            if r["pass"] == 0 and r["variant"] == "original" and r["id"] in ids
        ]
        assert len(records) == 50
        baseline[cohort] = sum(r["prediction"] == r["target"] for r in records) / 50
print("Prepared five fixed dev50 slices", flush=True)
for _attempt in range(720):
    if not Path("/proc/107102").exists():
        break
    time.sleep(2)
else:
    raise SystemExit("prior GPU chain did not finish")
train = json.loads((root / "reports/s3-training-probe-001.json").read_text())
assert train["status"] == "completed"
results = {}
for phase, cohorts, adapter in [
    ("before", ["audio", "video"], root / "artifacts/s2-text-lora-001/joint-v2-001/adapter"),
    (
        "after",
        ["audio", "video", "image", "screenshot", "text"],
        root / "artifacts/s3-training-probe-001/adapter",
    ),
]:
    for cohort in cohorts:
        report_path = root / f"reports/s3-dev50-{phase}-{cohort}-001.json"
        cmd = [
            py,
            "-u",
            "-m",
            "scripts.s1_dev_eval",
            "--model-dir",
            model,
            "--adapter",
            str(adapter),
            "--data",
            sources[cohort]["data"],
            "--passes",
            "1",
            "--output-dir",
            str(output / f"{phase}-{cohort}"),
            "--report",
            str(report_path),
        ]
        if sources[cohort]["media_root"]:
            cmd += ["--media-root", sources[cohort]["media_root"]]
        subprocess.run(cmd, cwd=root, check=True, timeout=300)
        report = json.loads(report_path.read_text())
        score = report["metrics"]["product_accuracy"]
        if phase == "before":
            baseline[cohort] = score
        else:
            results[cohort] = score
        print(json.dumps({"phase": phase, "cohort": cohort, "accuracy": score}), flush=True)
summary = {
    "status": "completed",
    "exit_code": 0,
    "scope": "dev50 single-pass screening only; no phase or gate pass claim",
    "sources": sources,
    "baseline_v2": baseline,
    "s3_probe": results,
    "old_modalities_max_regression": max(
        baseline[c] - results[c] for c in ["image", "screenshot", "text"]
    ),
    "training_report_sha256": hashlib.sha256(
        (root / "reports/s3-training-probe-001.json").read_bytes()
    ).hexdigest(),
}
(root / "reports/s3-quality-probe-summary-001.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary), flush=True)
