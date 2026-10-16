"""Full public cohorts require500original DEV rows and complete legal+Uprobabilities."""

import importlib.util
import math
from pathlib import Path

import pytest
from test_gold_inventory import sample

from data.schema import Sample

path = Path(__file__).resolve().parents[1] / ".scratch/s4/evaluate-program-stratified-public500.py"
spec = importlib.util.spec_from_file_location("public500_runner", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def samples():
    return [Sample.model_validate(sample(str(i), pool="dev", modality="text")) for i in range(500)]


def test_real_count_and_public_pool_guard(tmp_path):
    rows = samples()
    path = tmp_path / "dev.jsonl"
    path.write_text("\n".join(s.model_dump_json() for s in rows[:100]) + "\n")
    with pytest.raises(ValueError, match="500"):
        module.public500(path)
    rows[0].meta.pool = "train"
    path.write_text("\n".join(s.model_dump_json() for s in rows) + "\n")
    with pytest.raises(ValueError, match="public DEV"):
        module.public500(path)


def test_joint_nll_preserves_u_and_conditional_distribution():
    source = samples()
    records = []
    for s in source:
        s.questions[0].target.value = "yes"
        records.append(
            {
                "id": s.id,
                "pool": "dev",
                "target": "yes",
                "raw_logits": {"no": math.log(0.3), "U": math.log(0.1), "yes": math.log(0.6)},
                "probabilities": {"yes": 2 / 3, "no": 1 / 3},
            }
        )
    result = module.losses(records, source)
    assert result["joint_legal_plus_u_nll"] == pytest.approx(-math.log(0.6))
    assert result["conditional_nll"] == pytest.approx(-math.log(2 / 3))
    del records[0]["raw_logits"]["U"]
    with pytest.raises(ValueError, match="legal"):
        module.losses(records, source)
