"""CPU draw ordering and exact TRAIN-ledger binding."""

import importlib.util
from pathlib import Path

import pytest
from test_gold_inventory import sample

from data.schema import Sample
from scripts.s1_text_pools import canonical_sample_sha

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-claim-pair-pilot.py"
spec = importlib.util.spec_from_file_location("claim_prepare", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_inverse_order_preserves_every_fixed_step_offset():
    import random

    desired = list(range(120))
    arranged = module.inverse_order(desired)
    order = list(range(120))
    random.Random(42).shuffle(order)
    assert [arranged[index] for index in order] == desired
    for step in range(10):
        assert [arranged[index] for index in order[step * 12 : step * 12 + 12]] == desired[
            step * 12 : step * 12 + 12
        ]
    with pytest.raises(ValueError, match="draw_count"):
        module.inverse_order(desired[:-1])


def test_ledger_binds_full_sample_and_rejects_gold_or_version_mutation():
    s = Sample.model_validate(sample("train", pool="train", modality="text"))
    ledger = [{"item_id": s.id, "pool": "train", "derived_sample_sha256": canonical_sample_sha(s)}]
    assert module.exact_ledger([s], ledger) == ledger
    for field in ("data_version", "pool"):
        row = s.model_dump(mode="json")
        row["meta"][field] = "dev" if field == "pool" else "changed-version"
        with pytest.raises(ValueError, match="exact_source_ledger"):
            module.exact_ledger([Sample.model_validate(row)], ledger)
    with pytest.raises(ValueError, match="exact_source_ledger"):
        module.exact_ledger([s], [{**ledger[0], "pool": "dev"}])
    with pytest.raises(ValueError, match="conflicting_source_ledger"):
        module.exact_ledger([s], [ledger[0], {**ledger[0], "extra": "conflict"}])
