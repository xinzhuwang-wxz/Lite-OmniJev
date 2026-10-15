"""Exercise S3 evidence-audit rejection against disposable dev artifact copies."""

import copy
import hashlib
import json
import runpy
import tempfile
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
audit = Path("/root/autodl-tmp/s3-verify-final-sidecars.py")
verify = runpy.run_path(str(audit))["verify"]
report_name = "s3-dev-eval-image-stable-001.json"
report = json.loads((root / "reports" / report_name).read_text())
index = report["command"].index("--output-dir") + 1
sidecar = root / report["command"][index] / "records.jsonl"
original = [json.loads(line) for line in sidecar.read_text().splitlines()]
cases = {}
with tempfile.TemporaryDirectory(prefix="s3-audit-rejection-", dir=root / "artifacts") as tmp:
    isolated = Path(tmp)
    (isolated / "reports").mkdir()
    (isolated / "copied").mkdir()
    for case in (
        "valid",
        "target_changed",
        "probability_changed",
        "missing_record",
        "wrong_adapter",
    ):
        candidate = copy.deepcopy(report)
        rows = copy.deepcopy(original)
        if case == "target_changed":
            rows[0]["target"] = "U"
        elif case == "probability_changed":
            token = next(iter(rows[0]["probabilities"]))
            rows[0]["probabilities"][token] += 0.01
        elif case == "missing_record":
            rows.pop()
        elif case == "wrong_adapter":
            candidate["adapter_sha256"] = "0" * 64
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (isolated / "copied/records.jsonl").write_bytes(payload)
        candidate["command"][index] = "copied"
        # A matching digest must not make altered semantics admissible.
        candidate["sidecar_sha256"] = hashlib.sha256(payload).hexdigest()
        (isolated / "reports" / report_name).write_text(json.dumps(candidate))
        try:
            verify(isolated, "image")
        except AssertionError:
            cases[case] = "rejected"
        else:
            cases[case] = "accepted"
assert cases == {
    "valid": "accepted",
    "target_changed": "rejected",
    "probability_changed": "rejected",
    "missing_record": "rejected",
    "wrong_adapter": "rejected",
}
out = root / "reports/s3-sidecar-audit-rejections-001.json"
assert not out.exists()
result = {
    "status": "passed",
    "exit_code": 0,
    "audit_script_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    "cases": cases,
    "scope": "CPU disposable dev copies; original artifacts untouched; no sealed inputs",
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
