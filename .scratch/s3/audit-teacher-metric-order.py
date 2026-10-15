"""Bind teacher baseline answer-order review to its historical executed source."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.s3_gate import AUDIO_TEACHER_ACCURACY, VIDEO_TEACHER_ACCURACY

rows = []
for cohort, accuracy in {"audio": AUDIO_TEACHER_ACCURACY, "video": VIDEO_TEACHER_ACCURACY}.items():
    path = Path(f"reports/s0-development-{cohort}-jev-omni-12b-001.json")
    report = json.loads(path.read_text())
    commit = report["source"]["git"]["commit"]
    source = subprocess.check_output(["git", "show", f"{commit}:scripts/s0_teacher_development.py"])
    assert hashlib.sha256(source).hexdigest() == report["runner_sha256"]
    assert b"tokens = list(answer_tokens(question))" in source
    assert b"values = [float(probabilities[option]) for option in options]" in source
    assert (
        b"targets.append(list(answer_tokens(question)).index(str(question.target.value)))" in source
    )
    assert b'"metrics": evaluate([r["probabilities"] for r in first], targets)' in source
    assert report["metrics"]["accuracy"] == accuracy
    assert report["repeat_exact"] is True and report["metrics"]["count"] == 500
    rows.append(
        {
            "cohort": cohort,
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "historical_runner_sha256": report["runner_sha256"],
            "commit": commit,
            "matched_teacher_accuracy": accuracy,
            "canonical_option_and_target_order": True,
        }
    )
out = Path("reports/s3-teacher-metric-order-audit-001.json")
assert not out.exists()
result = {
    "status": "passed",
    "exit_code": 0,
    "scope": (
        "Historical executed teacher source matches report digest and preserves canonical "
        "option/target order. S3 reference constants unchanged; no new model evaluation."
    ),
    "cohorts": rows,
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
