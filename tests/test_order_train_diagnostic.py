"""Small CPU checks for prediction-independent TRAIN subset and provenance rejection."""

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from data.schema import Sample
from scripts.s1_text_pools import canonical_sample_sha

spec = importlib.util.spec_from_file_location(
    "order_train12", Path(__file__).parents[1] / ".scratch/s4/evaluate-order-train12.py"
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    samples, ledger = [], []
    hashes = {"annotations": "a" * 64, "questions": "b" * 64}
    for family in range(30):
        for target in ["yes", "no"]:
            frames = [1, 33] if target == "yes" else [33, 1]
            program = {
                "operation": "less_frame",
                "events": [{"frame": x} for x in frames],
                "result": target,
            }
            provenance = {
                "program": program,
                "program_sha256": hashlib.sha256(
                    json.dumps(program, sort_keys=True).encode()
                ).hexdigest(),
                "annotations_zip_sha256": hashes["annotations"],
                "official_questions_sha256": hashes["questions"],
            }
            sample = Sample.model_validate(
                {
                    "id": f"synthetic-{family}-{target}",
                    "prefix": {
                        "state": "synthetic",
                        "media": [{"type": "video", "uri": "synthetic.mp4", "sha256": "c" * 64}],
                    },
                    "questions": [
                        {
                            "qid": "q",
                            "type": "noul",
                            "instruction": "Synthetic order?",
                            "target": {"kind": "hard", "value": target, "source": "synthetic"},
                        }
                    ],
                    "meta": {
                        "modality": ["video"],
                        "source_dataset": "synthetic",
                        "source_revision": "test",
                        "source_sha256": hashlib.sha256(
                            json.dumps(provenance, sort_keys=True).encode()
                        ).hexdigest(),
                        "license": "CC0-1.0",
                        "usage": "commercial",
                        "pool": "train",
                        "family_id": f"synthetic:{family}",
                        "data_version": module.ORDER_VERSION,
                    },
                }
            )
            samples.append(sample)
            ledger.append(
                {
                    "item_id": sample.id,
                    "pool": "train",
                    "family_id": sample.meta.family_id,
                    "media_sha256": "c" * 64,
                    "derived_sample_sha256": canonical_sample_sha(sample),
                    "program_provenance": provenance,
                }
            )
    return samples, ledger, hashes


def test_selection_is_fixed_balanced_distinct_family():
    samples, ledger, hashes = fixture()
    selected, rows = module.select_train12(samples, ledger, hashes)
    assert len(selected) == len({s.meta.family_id for s in selected}) == len(rows) == 12
    assert Counter(s.questions[0].target.value for s in selected) == {"yes": 6, "no": 6}
    other, _ = module.select_train12(list(reversed(samples)), list(reversed(ledger)), hashes)
    assert [s.id for s in selected] == [s.id for s in other]


@pytest.mark.parametrize("field", ["derived_sample_sha256", "media_sha256", "family_id", "pool"])
def test_mutated_ledger_rejected(field):
    samples, ledger, hashes = fixture()
    ledger[0][field] = "wrong"
    with pytest.raises(ValueError):
        module.select_train12(samples, ledger, hashes)


def test_raw_provenance_or_wrong_video_scope_rejected():
    samples, ledger, hashes = fixture()
    ledger[0]["program_provenance"]["official_questions_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="raw source"):
        module.select_train12(samples, ledger, hashes)
    with pytest.raises(ValueError, match="fixed60"):
        module.select_train12(samples[:-1], ledger, hashes)
