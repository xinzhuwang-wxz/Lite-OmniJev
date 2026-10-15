"""Index hash-bound historical training inputs without exporting sample contents."""

import hashlib
import json
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
base = Path("/root/autodl-tmp/lite-omnijev")
report_paths = list((root / "reports").glob("*.json"))
report_paths += list((root / "artifacts").rglob("run.json"))
report_paths += list((base / "artifacts/training").rglob("run.json"))
inputs = {}
unresolved = []
for report_path in sorted(set(report_paths)):
    try:
        report = json.loads(report_path.read_text())
    except (ValueError, OSError):
        continue
    if not isinstance(report, dict) or not isinstance(report.get("steps_completed"), int):
        continue
    if report["steps_completed"] <= 0 or not report.get("data_sha256"):
        continue
    candidates = [report_path.parent / "data.jsonl"]
    if report.get("checkpoint"):
        checkpoint = Path(report["checkpoint"])
        if checkpoint.is_absolute():
            candidates.append(checkpoint.parent.parent / "data.jsonl")
        else:
            candidates.extend(
                parent / checkpoint.parent.parent / "data.jsonl" for parent in [root, base]
            )
    data_path = next(
        (
            p
            for p in candidates
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == report["data_sha256"]
        ),
        None,
    )
    if data_path is None:
        unresolved.append({"report": str(report_path), "data_sha256": report["data_sha256"]})
        continue
    key = str(data_path.resolve())
    if key not in inputs:
        versions = {}
        count = 0
        for line in data_path.read_text().splitlines():
            row = json.loads(line)
            assert row["meta"]["pool"] == "train"
            version = row["meta"]["data_version"]
            versions[version] = versions.get(version, 0) + 1
            count += 1
        inputs[key] = {
            "path": key,
            "sha256": report["data_sha256"],
            "rows": count,
            "data_versions": versions,
            "report_bindings": [],
        }
    inputs[key]["report_bindings"].append(
        {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "steps_completed": report["steps_completed"],
        }
    )
output = {
    "status": "indexed" if not unresolved else "incomplete",
    "scope": (
        "Scoped known reports and run.json discovery; "
        "not proof of all historical jobs or pool isolation"
    ),
    "files": list(inputs.values()),
    "unresolved": unresolved,
}
out = root / "artifacts/s4-gold-inventory-001"
out.mkdir(exist_ok=True)
assert not (out / "historical-training-index.json").exists()
(out / "historical-training-index.json").write_text(json.dumps(output, indent=2) + "\n")
summary = {
    "status": output["status"],
    "input_files": len(inputs),
    "report_bindings": sum(len(v["report_bindings"]) for v in inputs.values()),
    "rows_including_repeated_exposures": sum(v["rows"] for v in inputs.values()),
    "index_sha256": hashlib.sha256(
        (out / "historical-training-index.json").read_bytes()
    ).hexdigest(),
    "unresolved": unresolved,
    "scope": output["scope"],
}
(root / "reports/s4-historical-training-index-001.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary))
