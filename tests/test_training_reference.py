"""CPU evidence for TRAIN reference binding and loss gradients, not model quality."""

import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from data.schema import Sample
from data.synth import generate_samples
from scripts.train_reference import precompute
from train.trainer import (
    LoopConfig,
    data_bytes,
    decision_loss,
    file_sha256,
    load_reference,
    reference_header,
    reference_order,
    reference_prepared_hash,
    reference_row,
    validate_reference_row,
)


def test_zero_weight_is_exact_legacy_loss_and_gradient():
    x = torch.tensor([7.0, 0.4, -0.2, 0.1], requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    old = decision_loss(x, [1, 2, 3], [1.0, 0.0, 0.0])
    new = decision_loss(
        y, [1, 2, 3], [1.0, 0.0, 0.0], reference_probs=[float("nan")], reference_kl_weight=0.0
    )
    assert torch.equal(old, new)
    old.backward()
    new.backward()
    assert torch.equal(x.grad, y.grad)
    assert not any(
        k.startswith("reference_")
        for k in LoopConfig(sample_count=300, steps=30, accumulation=10).model_dump()
    )


@pytest.mark.parametrize(
    "target,q,enabled,active",
    [
        ([1.0, 0.0, 0.0], [0.7, 0.2, 0.1], True, True),
        ([1.0, 0.0, 0.0], [0.1, 0.8, 0.1], True, False),
        ([1.0, 0.0, 0.0], [0.45, 0.45, 0.1], True, False),
        ([1.0, 0.0, 0.0], [0.7, 0.2, 0.1], False, False),
        ([0.0, 0.0, 1.0], [0.1, 0.2, 0.7], True, True),
    ],
)
def test_reference_mask_gradient_and_detached_reference(target, q, enabled, active):
    x = torch.tensor([100.0, 0.2, -0.4, 0.7], requires_grad=True)
    reference = torch.tensor(q, requires_grad=True)
    loss = decision_loss(
        x,
        [1, 2, 3],
        target,
        reference_probs=reference,
        reference_kl_weight=0.5,
        reference_enabled=enabled,
    )
    loss.backward()
    p = x.detach()[1:].softmax(-1)
    expected = p - torch.tensor(target)
    if active:
        expected += 0.5 * (p - reference.detach())
    torch.testing.assert_close(x.grad[1:], expected)
    assert reference.grad is None
    assert x.grad[0] == 0


def test_reference_equal_student_has_zero_added_loss_and_gradient():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    q = x.detach().softmax(-1)
    ce = decision_loss(x, [0, 1, 2], [0.0, 0.0, 1.0])
    loss = decision_loss(
        x,
        [0, 1, 2],
        [0.0, 0.0, 1.0],
        reference_probs=q,
        reference_kl_weight=0.5,
        reference_enabled=True,
    )
    torch.testing.assert_close(loss, ce)
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], torch.autograd.grad(ce, x)[0])


@pytest.fixture
def bound_reference(tmp_path):
    samples = generate_samples(seed=42, count=300)
    # Explicit public-interface TRAIN cases; no benchmark rows or sealed inputs.
    samples = [
        Sample.model_validate({**s.model_dump(), "meta": {**s.meta.model_dump(), "pool": "train"}})
        for s in samples
    ]
    data = tmp_path / "train.jsonl"
    data.write_bytes(data_bytes(samples))
    replay = tmp_path / "replay.jsonl"
    replay.write_bytes(data_bytes(samples))
    ledger = tmp_path / "allocation.jsonl"
    ledger.write_text(
        "".join(json.dumps({"item_id": s.id, "pool": "train"}) + "\n" for s in samples)
    )
    sidecar = tmp_path / "reference.json"
    config = LoopConfig(
        sample_count=300,
        steps=30,
        accumulation=10,
        reference_kl_weight=0.5,
        reference_sidecar=str(sidecar),
        reference_sidecar_sha256="0" * 64,
        reference_ledger=str(ledger),
        reference_ledger_sha256=file_sha256(ledger),
        reference_replay_data=str(replay),
        reference_replay_sha256=file_sha256(replay),
    )
    examples = [
        {
            "id": s.id,
            "input_ids": [i + 10],
            "token_ids": [1, 2, 3],
            "answer_token_ids": {"A": 1, "B": 2, "U": 3},
            "target": [1.0, 0.0, 0.0],
            "pixel_values": torch.tensor([float(i)]),
        }
        for i, s in enumerate(samples)
    ]
    report = {
        "init_adapter": {"weights_sha256": "a" * 64, "config_sha256": "b" * 64},
        "asset_hashes": {"config": "c" * 64},
    }
    payload = {
        "header": reference_header(config, samples, report["init_adapter"], report["asset_hashes"]),
        "rows": [
            reference_row(draw, index, samples[index], examples[index], [0.7, 0.2, 0.1])
            for draw, index in enumerate(reference_order(config, len(samples)))
        ],
    }

    def write():
        sidecar.write_text(json.dumps(payload))
        return config.model_copy(update={"reference_sidecar_sha256": file_sha256(sidecar)})

    return samples, examples, report, data, payload, write


def test_actual_draw_order_and_valid_sidecar(bound_reference):
    samples, examples, report, data, payload, write = bound_reference
    config = write()
    order = list(range(300))
    random.Random(42).shuffle(order)
    assert reference_order(config, 300) == order
    assert len(set(order)) == 300
    rows = load_reference(config, samples, examples, report, data)
    assert [row["id"] for row in rows] == [samples[i].id for i in order]
    video = samples[0].model_copy(
        update={"meta": samples[0].meta.model_copy(update={"modality": ["video"]})}
    )
    row = reference_row(0, 0, video, examples[0], [0.7, 0.2, 0.1])
    assert not validate_reference_row(row, 0, 0, video, examples[0])


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "draw",
        "token",
        "prepared",
        "data",
        "adapter",
        "ledger",
        "prob",
        "U",
        "replay",
        "config",
        "adapter_config",
        "sidecar_sha",
        "ledger_file",
        "dev",
        "bool_draw",
    ],
)
def test_binding_rejects_mismatch(bound_reference, change):
    samples, examples, report, data, payload, write = bound_reference
    if change == "missing":
        payload["rows"].pop()
    elif change == "extra":
        payload["rows"].append(copy.deepcopy(payload["rows"][0]))
    elif change == "draw":
        payload["rows"][0]["draw"] = 1
    elif change == "token":
        payload["rows"][0]["token_ids"].reverse()
    elif change == "prepared":
        examples[0]["pixel_values"][0] += 1
    elif change == "data":
        payload["header"]["data_sha256"] = "f" * 64
    elif change == "adapter":
        report["init_adapter"]["weights_sha256"] = "f" * 64
    elif change == "ledger":
        payload["header"]["ledger_sha256"] = "f" * 64
    elif change == "prob":
        payload["rows"][0]["reference_probs"] = [float("nan"), 0.2, 0.8]
    elif change == "U":
        examples[0]["answer_token_ids"] = {"A": 1, "B": 2, "C": 3}
    elif change == "replay":
        samples[0] = samples[0].model_copy(update={"id": "new-nonreplay"})
    config = write()
    if change == "config":
        config = config.model_copy(update={"max_length": config.max_length + 1})
    elif change == "adapter_config":
        report["init_adapter"]["config_sha256"] = "f" * 64
    elif change == "sidecar_sha":
        config = config.model_copy(update={"reference_sidecar_sha256": "f" * 64})
    elif change == "dev":
        samples[0] = samples[0].model_copy(
            update={"meta": samples[0].meta.model_copy(update={"pool": "dev"})}
        )
    elif change == "bool_draw":
        payload["rows"][0]["draw"] = False
        config = write()
    elif change == "ledger_file":
        __import__("pathlib").Path(config.reference_ledger).write_text("")
    with pytest.raises((ValueError, KeyError)):
        load_reference(config, samples, examples, report, data)


def test_precompute_freezes_model_uses_eval_and_no_grad(bound_reference):
    samples, examples, report, data, payload, write = bound_reference

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(4))
            self.seen = []

        def forward(self, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            self.seen.append(kwargs["input_ids"].item())
            return SimpleNamespace(logits=self.parameter.reshape(1, 1, 4))

    model = Model()
    result = precompute(write(), samples, examples, model, payload["header"], "cpu")
    assert not model.parameter.requires_grad and model.parameter.grad is None
    assert model.seen == [examples[i]["input_ids"][0] for i in reference_order(write(), 300)]
    assert len(result["rows"]) == 300
    assert reference_prepared_hash(examples[0]) != reference_prepared_hash(examples[1])
