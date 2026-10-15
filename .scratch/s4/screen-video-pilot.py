"""Video-first fixed dev screening of the bounded second pilot."""

import hashlib
import json
import subprocess
from pathlib import Path

from data.schema import Sample, answer_tokens
from train.readout import read_probabilities

root = Path("/root/autodl-tmp/s1-text-lora-001")
train_path = root / "reports/s4-video-pilot-train-001.json"
train = json.loads(train_path.read_text())
assert train["status"] == "completed" and train["exit_code"] == 0 and train["steps_completed"] == 30
adapter = root / "artifacts/s4-video-pilot-train-001/adapter"
assert (
    hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    == train["adapter_sha256"]
)
assert (
    train["init_adapter"]["weights_sha256"]
    == "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac"
)
prior = root / "artifacts/s4-pilot-five-screen-001"
manifest = json.loads((prior / "manifest.json").read_text())
commands = json.loads((prior / "commands.json").read_text())
out = root / "artifacts/s4-video-pilot-screen-001"
out.mkdir(exist_ok=False, mode=0o700)
result = {
    "status": "running",
    "scope": "fixed public dev screen, not stage acceptance",
    "adapter_sha256": train["adapter_sha256"],
    "results": {},
    "eligible_for_remaining_retention": False,
    "cost": None,
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
report_path = root / "reports/s4-video-pilot-screening-001.json"
assert not report_path.exists()
for cohort, minimum in [("video", 102), ("audio", 37)]:
    source = next(s for s in manifest["sources"] if s["cohort"] == cohort)
    data = Path(source["data"]["path"])
    assert hashlib.sha256(data.read_bytes()).hexdigest() == source["data"]["sha256"]
    samples = {r["id"]: r for r in map(json.loads, data.read_text().splitlines())}
    assert all(r["meta"]["pool"] == "dev" for r in samples.values())
    argv = list(next(c["argv"] for c in commands if c["source"] == cohort))
    for flag, value in [
        ("--adapter", str(adapter)),
        ("--output-dir", str(out / (cohort + "-eval"))),
        ("--report", str(out / (cohort + "-report.json"))),
    ]:
        argv[argv.index(flag) + 1] = value
    with (out / (cohort + ".log")).open("x") as log:
        proc = subprocess.run(argv, stdout=log, stderr=log, timeout=300)
    assert proc.returncode == 0, "evaluator_failed"
    r = json.loads((out / (cohort + "-report.json")).read_text())
    assert (
        r["status"] == "passed"
        and r["exit_code"] == 0
        and r["adapter_sha256"] == train["adapter_sha256"]
    )
    assert r["data_sha256"] == source["data"]["sha256"]
    sidecar = out / (cohort + "-eval") / "records.jsonl"
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == r["sidecar_sha256"]
    groups = {}
    correct = 0
    count = 0
    for row in map(json.loads, sidecar.read_text().splitlines()):
        if row["pass"] != 0 or row["variant"] != "original":
            continue
        sample = Sample.model_validate(samples[row["id"]])
        q = sample.questions[0]
        pred = read_probabilities(
            row["raw_logits"], list(answer_tokens(q)), temperature=1
        ).prediction
        ok = pred == q.target.value
        correct += ok
        count += 1
        words = set(q.instruction.lower().split())
        kind = (
            "temporal"
            if words & {"before", "after", "when", "while"}
            else "motion"
            if words & {"moving", "stationary"}
            else "existence"
        )
        g = groups.setdefault(kind, {"count": 0, "correct": 0})
        g["count"] += 1
        g["correct"] += ok
    assert count == len(samples)
    passed = correct >= minimum
    if cohort == "video":
        passed = passed and groups["temporal"]["correct"] >= 60 and groups["motion"]["correct"] >= 7
    result["results"][cohort] = {
        "count": count,
        "correct": correct,
        "required_correct": minimum,
        "groups": groups if cohort == "video" else None,
        "passed": passed,
        "report_sha256": hashlib.sha256((out / (cohort + "-report.json")).read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        break
result["status"] = "completed"
result["eligible_for_remaining_retention"] = len(result["results"]) == 2 and all(
    r["passed"] for r in result["results"].values()
)
report_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
