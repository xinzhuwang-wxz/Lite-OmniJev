"""Fixed TRAIN selection and safe evaluation-preflight rejection; no GPU."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from data.schema import Sample
from train.checkpoint import file_sha256
from train.trainer import data_bytes


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(".scratch/s4") / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sample(i):
    return Sample.model_validate(
        {
            "id": str(i),
            "prefix": {
                "state": "state",
                "media": [{"type": "video", "uri": "video.mp4", "sha256": "a" * 64}],
            },
            "questions": [
                {
                    "qid": "q",
                    "type": "noul",
                    "instruction": "Was object present before event?",
                    "target": {
                        "kind": "hard",
                        "value": "yes" if i % 2 == 0 else "no",
                        "source": "gold",
                    },
                }
            ],
            "meta": {
                "modality": ["video"],
                "source_dataset": "fixture",
                "source_revision": "r1",
                "source_sha256": "a" * 64,
                "license": "MIT",
                "usage": "research",
                "pool": "train",
                "family_id": str(i),
                "data_version": "v1",
            },
        }
    )


def test_fixed_balanced_unique_train_selection_has_no_model_input():
    helper = module("prepare-train-learnability")
    rows = [sample(i) for i in range(48)]
    selected = helper.select_train(rows)
    assert len(selected) == len({s.meta.family_id for s in selected}) == 12
    assert sum(s.questions[0].target.value == "yes" for s in selected) == 6
    assert [s.id for s in selected] == [s.id for s in helper.select_train(list(reversed(rows)))]


def test_nontrain_and_insufficient_gold_selection_rejected():
    helper = module("prepare-train-learnability")
    rows = [sample(i) for i in range(48)]
    with pytest.raises(ValueError, match="TRAIN only"):
        helper.select_train(
            [
                rows[0].model_copy(
                    update={"meta": rows[0].meta.model_copy(update={"pool": "dev"})}
                ),
                *rows[1:],
            ]
        )
    with pytest.raises(ValueError, match="insufficient"):
        helper.select_train([s for s in rows if s.questions[0].target.value == "yes"])


@pytest.mark.parametrize("phase", ["before", "after"])
def test_wrong_phase_identity_fails_before_processor_or_gpu(tmp_path, monkeypatch, phase):
    helper = module("evaluate-train-learnability")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    rows = [sample(i) for i in range(12)]
    (inputs / "train.jsonl").write_bytes(data_bytes(rows))
    (inputs / "allocation-ledger.jsonl").write_text(
        "".join(json.dumps({"item_id": s.id, "pool": "train"}) + "\n" for s in rows)
    )
    config = {
        "sample_count": 12,
        "steps": 10,
        "accumulation": 12,
        "modality": "video",
        "media_root": str(tmp_path),
        "expected_data_version": "v1",
    }
    (inputs / "config-ce.json").write_text(json.dumps(config))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"CPU identity fixture")
    (adapter / "adapter_config.json").write_text("{}")
    identity = {
        "path": str(adapter),
        "weights_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "config_sha256": file_sha256(adapter / "adapter_config.json"),
    }
    source = {
        "status": "ready_for_preflight",
        "exit_code": 0,
        "samples": 12,
        "init_adapter": identity,
        "output_sha256": {
            name: file_sha256(inputs / name)
            for name in ("train.jsonl", "allocation-ledger.jsonl", "config-ce.json")
        },
    }
    (inputs / "report.json").write_text(json.dumps(source))
    output = tmp_path / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--inputs",
            str(inputs),
            "--phase",
            phase,
            "--adapter",
            str(adapter),
            "--expected-adapter-sha256",
            identity["weights_sha256"],
            "--output",
            str(output),
        ],
    )
    assert helper.main() == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["gpu_requested"] is False and receipt["status"] == "failed"
    assert ("frozen CE0" if phase == "before" else "requires training receipt") in receipt["error"]
    assert not (output / "records.jsonl").exists()
