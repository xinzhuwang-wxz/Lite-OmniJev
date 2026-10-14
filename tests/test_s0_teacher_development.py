import json
import math
from pathlib import Path

import pytest

from scripts.s0_teacher_development import load_cohort, normalized_head


def test_head_distribution_keeps_explicit_label_order_and_normalizes_roundoff():
    values = normalized_head(["yes", "no"], {"no": 0.4001, "yes": 0.6})
    assert math.fsum(values) == 1
    assert values[0] == 0.6 / 1.0001
    for bad in ({"no": 0.4}, {"yes": math.nan, "no": 0.4}, {"yes": 0.3, "no": 0.3}):
        with pytest.raises(ValueError):
            normalized_head(["yes", "no"], bad)


def test_teacher_refuses_unbound_development_payload(tmp_path: Path):
    data = tmp_path / "dev.jsonl"
    report = tmp_path / "report.json"
    data.write_text("[]\n")
    report.write_text(
        json.dumps({"status": "passed", "exit_code": 0, "outputs": {"dev_jsonl_sha256": "a" * 64}})
    )
    with pytest.raises(ValueError, match="does not bind"):
        load_cohort({"data": str(data), "report": str(report)})
