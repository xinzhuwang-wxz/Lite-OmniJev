"""Run bounded old-cohort screens only after the fixed crossmodal trigger passes."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
out = root / "artifacts/s4-pilot-five-screen-001"
report_path = root / "reports/s4-pilot-screening-001.json"
assert not report_path.exists()
second = root / "reports/s4-pilot-crossmodal-supplemental-001.json"
for _ in range(60):
    if second.exists():
        break
    time.sleep(2)
else:
    raise SystemExit("supplemental_report_timeout")
identity = {
    "original": ("1af868df1d442def73765d5cfda0532a8b86884f441d28f844a5f6ee9e3ad972", 33),
    "supplemental": ("f6a60b8f7a7cf88a303ed6a0765b043cf8c3ccc957eeef03013e96d2133df879", 40),
}
training = json.loads((root / "reports/s4-joint-pilot-train-001.json").read_text())
assert training["status"] == "completed" and training["steps_completed"] == 30
cross, deltas = {}, []
for name, (selected_sha, correct) in identity.items():
    path = root / f"reports/s4-pilot-crossmodal-{name}-001.json"
    r = json.loads(path.read_text())
    assert r["status"] == "completed" and r["exit_code"] == 0
    assert r["selected_ids_sha256"] == selected_sha
    assert r["adapter_sha256"] == training["adapter_sha256"]
    count = r["metrics"]["count"]
    assert count == 50
    candidate = round(r["metrics"]["accuracy"] * count)
    deltas.append(candidate - correct)
    cross[name] = {
        "count": count,
        "baseline_correct": correct,
        "candidate_correct": candidate,
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
passed = min(deltas) >= 0 and max(deltas) >= 2
result = {
    "status": "running",
    "quality_scope": "development screening only, not stage acceptance",
    "candidate_adapter_sha256": training["adapter_sha256"],
    "crossmodal": cross,
    "crossmodal_trigger_passed": passed,
    "old_cohorts": {},
    "qualified_for_expansion": False,
    "automatic_training_expansion": False,
}


def save():
    temp = report_path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(report_path)


save()
if passed:
    baseline = {r["cohort"]: r for r in json.loads((out / "baseline-matches.json").read_text())}
    commands = json.loads((out / "commands.json").read_text())
    for item in commands:
        cohort = item["source"]
        with (out / (cohort + "-run.log")).open("x") as log:
            p = subprocess.run(
                item["argv"], stdout=log, stderr=log, timeout=item["timeout_seconds"]
            )
        assert p.returncode == 0, "cohort_evaluator_failed"
        path = Path(item["argv"][item["argv"].index("--report") + 1])
        r = json.loads(path.read_text())
        assert r["status"] == "passed" and r["exit_code"] == 0
        assert r["adapter_sha256"] == training["adapter_sha256"]
        correct = round(r["metrics"]["product_accuracy"] * r["samples"])
        accepted = correct >= baseline[cohort]["minimum_candidate_correct"]
        result["old_cohorts"][cohort] = {
            "count": r["samples"],
            "candidate_correct": correct,
            "baseline_correct": baseline[cohort]["baseline_correct"],
            "minimum_correct": baseline[cohort]["minimum_candidate_correct"],
            "passed": accepted,
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        save()
        if not accepted:
            break
    result["qualified_for_expansion"] = len(result["old_cohorts"]) == 5 and all(
        r["passed"] for r in result["old_cohorts"].values()
    )
result["status"] = "completed"
save()
print(json.dumps(result))
