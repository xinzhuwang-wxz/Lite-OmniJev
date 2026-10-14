"""Summarize frozen local development evidence without model execution."""

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "min": values[0],
        "p50": values[math.ceil(len(values) * 0.50) - 1],
        "p95": values[math.ceil(len(values) * 0.95) - 1],
        "max": values[-1],
    }


table_path = Path("reports/s0-development-baselines-table-001.json")
table = json.loads(table_path.read_text())
output = []
for entry in table["rows"]:
    report_path = Path(entry["report"])
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed" and report["exit_code"] == 0
    if "raw_records" in report:
        sidecar = Path(report["raw_records"]["path"])
        expected = report["raw_records"]["sha256"]
    else:
        raw = report.get("raw_report", report.get("local_raw_report"))
        raw_path = Path(raw["path"] if isinstance(raw, dict) else raw)
        sidecar = raw_path.parent / Path(report["records_sidecar"]["path"]).name
        expected = report["records_sidecar"]["sha256"]
    assert digest(sidecar) == expected, str(sidecar)
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(records) == 1000
    warm = [row for row in records if row["pass_index"] == 2]
    assert len(warm) == 500
    tokens = [row.get("prompt_tokens", row.get("prompt_tokens_compute")) for row in warm]
    forwards = [row.get("forward_wall_seconds", row.get("forward_seconds")) for row in warm]
    assert all(value is not None for value in tokens + forwards)
    preparations = [row.get("prepare_wall_seconds", row.get("prepare_seconds")) for row in warm]
    output.append(
        {
            **entry,
            "report_sha256": digest(report_path),
            "records": str(sidecar),
            "records_sha256": expected,
            "warm_prompt_compute_tokens": distribution(tokens),
            "warm_forward_seconds": distribution(forwards),
            "warm_prepare_seconds": distribution(preparations)
            if all(value is not None for value in preparations)
            else None,
            "job_elapsed_seconds": report["elapsed_seconds"],
            "code_commit": report["source"]["git"]["commit"],
        }
    )
result = {
    "status": "complete" if len(output) == table["expected_supported_pairs"] else "partial",
    "matrix_sha256": digest(table_path),
    "summarizer_sha256": digest(Path(__file__)),
    "verified_supported_pairs": len(output),
    "expected_supported_pairs": table["expected_supported_pairs"],
    "pending": table["pending_supported_pairs"],
    "rows": output,
    "unsupported": table["unsupported_pairs"],
    "measurement_boundary": (
        "Pass 2 of shared-GPU quality evaluation; nearest-rank quantiles. "
        "Forward excludes media preparation and gateway; Valen preparation was not "
        "instrumented and remains null. Diagnostic budgets, not isolated release p95 acceptance."
    ),
    "cost_usd": None,
    "whole_s0_complete": False,
}
Path("reports/s0-development-compute-budgets-001.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps({"verified_supported_pairs": len(output), "status": result["status"]}))
