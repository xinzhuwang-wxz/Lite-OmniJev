"""Remaining fixed public-dev retention after the second pilot's video/audio screens."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
out = root / "artifacts/s4-video-pilot-screen-001"
training = json.loads((root / "reports/s4-video-pilot-train-001.json").read_text())
initial = json.loads((root / "reports/s4-video-pilot-screening-001.json").read_text())
assert initial["eligible_for_remaining_retention"]
result = {
    "status": "running",
    "scope": "small fixed development retention only, not S4 acceptance",
    "adapter_sha256": training["adapter_sha256"],
    "results": dict(initial["results"]),
    "qualified_for_full_dev": False,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
report = root / "reports/s4-video-pilot-retention-001.json"
assert not report.exists()
for source, minimum, selected_sha in [
    ("original", 35, "1af868df1d442def73765d5cfda0532a8b86884f441d28f844a5f6ee9e3ad972"),
    ("supplemental", 40, "f6a60b8f7a7cf88a303ed6a0765b043cf8c3ccc957eeef03013e96d2133df879"),
]:
    p = root / f"reports/s4-video-pilot-crossmodal-{source}-001.json"
    for _ in range(120):
        if p.exists():
            break
        time.sleep(2)
    else:
        raise SystemExit("crossmodal_report_unavailable")
    r = json.loads(p.read_text())
    assert r["status"] == "completed" and r["exit_code"] == 0
    assert (
        r["adapter_sha256"] == training["adapter_sha256"]
        and r["selected_ids_sha256"] == selected_sha
    )
    correct = round(r["metrics"]["accuracy"] * 50)
    result["results"]["crossmodal_" + source] = {
        "count": 50,
        "correct": correct,
        "required_correct": minimum,
        "passed": correct >= minimum,
        "report_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    if correct < minimum:
        break
cross_pass = len(result["results"]) == 4 and all(r["passed"] for r in result["results"].values())
if cross_pass:
    prior = root / "artifacts/s4-pilot-five-screen-001"
    commands = json.loads((prior / "commands.json").read_text())
    manifest = json.loads((prior / "manifest.json").read_text())
    for cohort, minimum in [("image", 39), ("screenshot", 49), ("text", 48)]:
        source = next(s for s in manifest["sources"] if s["cohort"] == cohort)
        assert (
            hashlib.sha256(Path(source["data"]["path"]).read_bytes()).hexdigest()
            == source["data"]["sha256"]
        )
        argv = list(next(c["argv"] for c in commands if c["source"] == cohort))
        for flag, value in [
            ("--adapter", str(root / "artifacts/s4-video-pilot-train-001/adapter")),
            ("--output-dir", str(out / (cohort + "-eval"))),
            ("--report", str(out / (cohort + "-report.json"))),
        ]:
            argv[argv.index(flag) + 1] = value
        with (out / (cohort + ".log")).open("x") as log:
            p = subprocess.run(argv, stdout=log, stderr=log, timeout=300)
        assert p.returncode == 0
        path = out / (cohort + "-report.json")
        r = json.loads(path.read_text())
        assert (
            r["status"] == "passed"
            and r["exit_code"] == 0
            and r["adapter_sha256"] == training["adapter_sha256"]
            and r["data_sha256"] == source["data"]["sha256"]
        )
        correct = round(r["metrics"]["product_accuracy"] * r["samples"])
        result["results"][cohort] = {
            "count": r["samples"],
            "correct": correct,
            "required_correct": minimum,
            "passed": correct >= minimum,
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        report.write_text(json.dumps(result, indent=2) + "\n")
        if correct < minimum:
            break
result["status"] = "completed"
result["qualified_for_full_dev"] = len(result["results"]) == 7 and all(
    r["passed"] for r in result["results"].values()
)
report.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
