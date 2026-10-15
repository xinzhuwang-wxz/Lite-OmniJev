"""CPU evidence for controlled-claim binding and opt-in accumulated gradients."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from data.schema import Sample
from data.synth import generate_samples
from train import trainer
from train.checkpoint import save_checkpoint, verify_resume_request
from train.pair_objective import claim_pair_loss
from train.trainer import LoopConfig, data_bytes, file_sha256

LEGACY_JSON = (
    '{"seed":42,"sample_count":120,"steps":10,"accumulation":12,"max_length":2048,'
    '"learning_rate":0.0001,"rank":16,"alpha":32,"dtype":"bfloat16","modality":"text",'
    '"video_num_frames":16,"media_root":null,"media_roots":null,"midway_step":null,'
    '"expected_data_version":null,"expected_data_versions":null,"checkpoint_every":null,'
    '"brier_weight":0.0,"preparation_mode":"eager","preparation_cache_bytes":268435456}'
)


def test_zero_weight_preserves_legacy_serialization_and_checkpoint(tmp_path):
    config = LoopConfig(sample_count=120, steps=10, accumulation=12, pair_loss_weight=0.0)
    assert config.model_dump_json() == LEGACY_JSON
    assert config.model_dump() == json.loads(LEGACY_JSON)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = save_checkpoint(
        tmp_path,
        step=1,
        config=json.loads(LEGACY_JSON),
        model_revision="test",
        data_sha256="a" * 64,
        asset_hashes={},
        order=list(range(120)),
        losses=[1.0],
        model=model,
        optimizer=optimizer,
    )
    metadata = verify_resume_request(
        tmp_path,
        checkpoint,
        config=config.model_dump(),
        model_revision="test",
        data_sha256="a" * 64,
        asset_hashes={},
        order=list(range(120)),
    )
    assert metadata["step"] == 1


@pytest.fixture
def paired(tmp_path):
    samples = generate_samples(seed=42, count=120)
    order = trainer.reference_order(LoopConfig(sample_count=120, steps=10, accumulation=12), 120)
    roles = {}
    for step in range(10):
        for offset in (0, 2):
            for delta, gold in ((0, "yes"), (1, "no")):
                roles[order[step * 12 + offset + delta]] = (f"family-{step}-{offset}", gold)
    converted = []
    examples = []
    for i, sample in enumerate(samples):
        family, gold = roles.get(i, (f"old-{i}", "yes"))
        row = sample.model_dump()
        row["prefix"] = {
            "state": f"controlled claim {gold}",
            "media": [
                {"type": kind, "uri": f"shared.{kind}", "sha256": digest * 64}
                for kind, digest in (("image", "a"), ("audio", "b"))
            ],
        }
        row["questions"] = [
            {
                "qid": "q",
                "type": "noul",
                "instruction": "Is claim true?",
                "target": {"kind": "hard", "value": gold, "source": "fixture_gold"},
            }
        ]
        row["meta"].update(pool="train", modality=["image", "audio"], family_id=family)
        converted.append(Sample.model_validate(row))
        examples.append(
            {
                "id": sample.id,
                "input_ids": [i + 1],
                "token_ids": [0, 1, 2],
                "answer_token_ids": {"yes": 0, "no": 1, "U": 2},
                "target": [float(gold == "yes"), float(gold == "no"), 0.0],
                "observed_modalities": ["image", "audio"],
                "pixel_values": torch.tensor([1.0]),
                "input_features": torch.tensor([2.0]),
            }
        )
    samples = converted
    data = tmp_path / "train.jsonl"
    data.write_bytes(data_bytes(samples))
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "".join(json.dumps({"item_id": s.id, "pool": "train"}) + "\n" for s in samples)
    )
    sidecar = tmp_path / "pairs.json"
    config = LoopConfig(
        sample_count=120,
        steps=10,
        accumulation=12,
        modality="mixed",
        media_root=str(tmp_path),
        expected_data_versions=[samples[0].meta.data_version],
        pair_loss_weight=0.1,
        pair_sidecar=str(sidecar),
        pair_sidecar_sha256="0" * 64,
        pair_ledger=str(ledger),
        pair_ledger_sha256=file_sha256(ledger),
    )
    rows = []
    for step in range(10):
        for offset in (0, 2):
            draw = step * 12 + offset
            pi, ni = order[draw : draw + 2]
            rows.append(
                trainer.claim_pair_row(
                    draw, pi, ni, samples[pi], samples[ni], examples[pi], examples[ni]
                )
            )
    payload = {
        "header": {
            "schema": "train-claim-pairs-v1",
            "data_sha256": hashlib.sha256(data_bytes(samples)).hexdigest(),
            "ledger_sha256": file_sha256(ledger),
            "seed": 42,
            "steps": 10,
            "accumulation": 12,
        },
        "pairs": rows,
    }

    def write():
        sidecar.write_text(json.dumps(payload))
        return config.model_copy(update={"pair_sidecar_sha256": file_sha256(sidecar)})

    return samples, examples, data, payload, write, order


def test_fixed_twenty_pairs_are_fully_bound(paired):
    samples, examples, data, payload, write, _ = paired
    rows = trainer.load_claim_pairs(write(), samples, examples, data)
    assert list(rows) == [step * 12 + offset for step in range(10) for offset in (0, 2)]
    assert list(rows.values()) == payload["pairs"]


@pytest.mark.parametrize(
    "change", ["missing", "draw", "index", "sample", "prepared", "family", "component", "boolindex"]
)
def test_rebound_sidecar_cannot_forge_mask_or_identity(paired, change):
    samples, examples, data, payload, write, _ = paired
    row = payload["pairs"][0]
    if change == "missing":
        payload["pairs"].pop()
    elif change == "draw":
        row["positive_draw"] = 2
    elif change == "index":
        row["positive_sample_index"] += 1
    elif change == "sample":
        row["positive_sample_sha256"] = "0" * 64
    elif change == "prepared":
        row["negative_prepared_sha256"] = "0" * 64
    elif change == "family":
        row["family_id"] = "forged"
    elif change == "component":
        row["components"][0]["sha256"] = "0" * 64
    else:
        row["negative_draw"] = True
    with pytest.raises(ValueError, match="binding mismatch"):
        trainer.load_claim_pairs(write(), samples, examples, data)


def test_matching_family_string_does_not_replace_component_or_gold_binding(paired):
    samples, examples, _, _, _, order = paired
    pi, ni = order[:2]
    negative = samples[ni].model_copy(
        update={
            "prefix": samples[ni].prefix.model_copy(
                update={
                    "media": [
                        samples[ni].prefix.media[0].model_copy(update={"sha256": "c" * 64}),
                        samples[ni].prefix.media[1],
                    ]
                }
            )
        }
    )
    with pytest.raises(ValueError, match="component hashes"):
        trainer.claim_pair_row(0, pi, ni, samples[pi], negative, examples[pi], examples[ni])
    with pytest.raises(ValueError, match="gold/mapping"):
        trainer.claim_pair_row(0, pi, ni, samples[ni], samples[pi], examples[ni], examples[pi])


def test_nontrain_ledger_rejected(paired):
    samples, examples, data, payload, write, _ = paired
    config = write()
    ledger = __import__("pathlib").Path(config.pair_ledger)
    text = ledger.read_text().replace('"train"', '"dev"', 1)
    ledger.write_text(text)
    payload["header"]["ledger_sha256"] = file_sha256(ledger)
    config = write().model_copy(update={"pair_ledger_sha256": file_sha256(ledger)})
    with pytest.raises(ValueError, match="ledger violations"):
        trainer.load_claim_pairs(config, samples, examples, data)


@pytest.mark.parametrize(
    "update",
    [
        {"pair_loss_weight": 0.2},
        {"brier_weight": 0.1},
        {"reference_kl_weight": 0.5},
        {"steps": 9},
        {"accumulation": 10},
        {"sample_count": 119},
        {"pair_sidecar": None},
    ],
)
def test_pair_pilot_config_rejects_scope_changes(paired, update):
    config = paired[4]()
    raw = config.model_dump()
    raw.update(update)
    with pytest.raises(ValueError):
        LoopConfig.model_validate(raw)


def test_cpu_training_accumulates_exact_two_pair_objectives_without_extra_forwards(
    paired, tmp_path, monkeypatch
):
    samples, examples, data, _, write, order = paired
    config = write()
    rows = trainer.load_claim_pairs(config, samples, examples, data)
    original_tensor = torch.tensor
    weight = torch.nn.Parameter(torch.tensor([0.1, -0.05, 0.0]))
    expected = []
    for step in range(10):
        terms = []
        for offset in range(12):
            example = examples[order[step * 12 + offset]]
            logits = weight * (example["input_ids"][0] / 120)
            terms.append(trainer.decision_loss(logits, example["token_ids"], example["target"]))
            if offset in (1, 3):
                positive = examples[order[step * 12 + offset - 1]]
                pair = claim_pair_loss(
                    weight * (positive["input_ids"][0] / 120),
                    logits,
                    positive_token_ids=positive["answer_token_ids"],
                    negative_token_ids=example["answer_token_ids"],
                    positive_gold="yes",
                    negative_gold="no",
                    positive_family_id="family",
                    negative_family_id="family",
                )
                terms.append(0.1 * pair)
        objective = sum(terms) / 12
        expected.append(torch.autograd.grad(objective, weight)[0])
    calls = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = weight

        def to(self, device):
            return self

        def forward(self, **kwargs):
            calls.append(int(kwargs["input_ids"][0, 0]))
            return SimpleNamespace(logits=(self.weight * (calls[-1] / 120)).reshape(1, 1, 3))

        def save_pretrained(self, directory, **kwargs):
            from pathlib import Path

            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.safetensors").write_bytes(b"cpu-only")

    class Optimizer:
        def __init__(self, *args, **kwargs):
            self.step_index = 0

        def zero_grad(self, **kwargs):
            weight.grad = None

        def step(self):
            torch.testing.assert_close(weight.grad, expected[self.step_index])
            self.step_index += 1

    model = Model()
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: original_tensor(*args, **{**kwargs, "device": "cpu"}),
    )
    monkeypatch.setattr(torch.optim, "AdamW", Optimizer)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU double")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: model)
        ),
    )
    monkeypatch.setattr(trainer, "attach_lora", lambda *a, **k: model)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *a, **k: tmp_path / "checkpoint")
    report = {
        "examples": 120,
        "max_prompt_tokens": 1,
        "config": config.model_dump(),
        "model_revision": "test",
        "data_sha256": "test",
        "asset_hashes": {},
    }
    trainer.train_smoke(
        config,
        tmp_path,
        examples,
        tmp_path,
        report,
        SimpleNamespace(log=lambda *a, **k: None),
        pair_rows=rows,
    )
    assert calls == [examples[index]["input_ids"][0] for index in order]
    assert len(calls) == 120
    # A late preparation change is rejected before forward/backward/optimizer updates.
    examples[order[0]]["input_ids"] = [999]
    with pytest.raises(ValueError, match="runtime prepared/draw"):
        trainer.train_smoke(
            config,
            tmp_path,
            examples,
            tmp_path,
            report,
            SimpleNamespace(log=lambda *a, **k: None),
            pair_rows=rows,
        )
    assert len(calls) == 120
    assert len(report["step_metrics"]) == 10
    for metric in report["step_metrics"]:
        assert metric["pair_count"] == 2 and metric["pair_loss_weight"] == 0.1
        assert metric["gold_ce"] > 0 and metric["pair_loss"] > 0
        assert metric["peak_cuda_bytes"] == 1024
        assert metric["step_wall_seconds"] >= 0
