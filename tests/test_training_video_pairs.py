"""CPU-only fixed-course binding and real autograd integration; no GPU quality claims."""

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.synth import generate_samples
from scripts.s0_media_dev import video_sample
from scripts.s1_text_pools import canonical_sample_sha
from train import trainer
from train.video_pairs import build_video_pairs, digest, load_video_pairs, member, question_hash


def video_config(tmp_path, **updates):
    values = dict(
        sample_count=1200,
        steps=100,
        accumulation=12,
        seed=42,
        modality="mixed",
        rank=64,
        alpha=128,
        learning_rate=2e-5,
        video_num_frames=16,
        image_max_soft_tokens=1120,
        media_root=str(tmp_path),
        expected_data_versions=["fixture"],
        video_pair_loss_weight=0.1,
        video_pair_sidecar=str(tmp_path / "sidecar.json"),
        video_pair_sidecar_sha256="a" * 64,
        video_pair_ledger=str(tmp_path / "ledger.jsonl"),
        video_pair_ledger_sha256="b" * 64,
    )
    return trainer.LoopConfig(**(values | updates))


@pytest.fixture
def course(tmp_path):
    replay = generate_samples(600, 42)
    for sample in replay:
        sample.questions = [sample.questions[0]]
    draws, metadata, official = [], [], {}
    for pair in range(300):
        samples, records = [], []
        for offset in range(2):
            scene = pair * 2 + offset
            record = {
                "question_id": 1,
                "question": "Does the sphere move?",
                "program": ["scene", "filter_sphere", "filter_moving", "exist"],
                "answer": "yes" if (pair + offset) % 2 else "no",
            }
            sample = video_sample(
                {"scene_index": scene, "family_id": f"clevrer:{scene}"},
                record,
                media_uri=f"{scene}.mp4",
                media_sha=hashlib.sha256(str(scene).encode()).hexdigest(),
            )
            sample.meta.pool = "train"
            samples.append(sample)
            records.append(record)
            official[sample.id] = record
        metadata.append(
            {
                "pair": pair,
                "metadata_only": True,
                "model_nonmedia_question_sha256": question_hash(samples[0]),
                "official_program_sha256": digest(
                    json.dumps(records[0]["program"], sort_keys=True).encode()
                ),
                "text_sha256": digest(samples[0].questions[0].instruction.encode()),
                "members": [member(s, r) for s, r in zip(samples, records, strict=True)],
            }
        )
        draws.extend(samples)
        if pair % 3 == 2:
            start = pair // 3 * 6
            draws.extend(replay[start : start + 6])
    order = list(range(1200))
    random.Random(42).shuffle(order)
    samples = [None] * 1200
    for index, sample in zip(order, draws, strict=True):
        samples[index] = sample
    data = tmp_path / "train.jsonl"
    data.write_bytes(trainer.data_bytes(samples))
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        "\n".join(
            json.dumps(
                {"item_id": s.id, "pool": "train", "derived_sample_sha256": canonical_sample_sha(s)}
            )
            for s in samples
        )
        + "\n"
    )
    config = video_config(tmp_path, video_pair_ledger_sha256=trainer.file_sha256(ledger))
    payload = build_video_pairs(
        samples,
        json.dumps({"metadata_only": True, "rows": metadata}),
        official,
        config.video_pair_ledger_sha256,
    )

    def write():
        path = Path(config.video_pair_sidecar)
        path.write_text(json.dumps(payload))
        return config.model_copy(update={"video_pair_sidecar_sha256": trainer.file_sha256(path)})

    return samples, payload, data, write, order


def test_complete_actual_shuffle_binding_both_polarity_orders(course):
    samples, payload, data, write, _ = course
    rows = load_video_pairs(write(), samples, data)
    assert len(rows) == 300
    assert list(rows) == [step * 12 + offset for step in range(100) for offset in (0, 2, 4)]
    assert {row["positive_offset"] for row in rows.values()} == {0, 1}
    assert rows[0]["members"][0]["family_id"] != rows[0]["members"][1]["family_id"]


@pytest.mark.parametrize(
    "drift", ["program", "gold", "question", "mask", "header", "canonical_ledger", "pool"]
)
def test_rebound_sidecar_cannot_forge_official_gold_program_mask_or_train(course, drift):
    samples, payload, data, write, _ = course
    question = next(iter(payload["official_questions"].values()))
    if drift == "program":
        question["program"].append("forged")
    elif drift == "gold":
        question["answer"] = "yes" if question["answer"] == "no" else "no"
    elif drift == "question":
        question["question"] += " different"
    elif drift == "mask":
        payload["pairs"][0]["draw"] = 6
    elif drift == "header":
        payload["header"]["seed"] = 43
    else:
        path = data.with_name("ledger.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["pool" if drift == "pool" else "derived_sample_sha256"] = (
            "dev" if drift == "pool" else "0" * 64
        )
        path.write_text("\n".join(json.dumps(row) for row in rows))
        config = write().model_copy(update={"video_pair_ledger_sha256": trainer.file_sha256(path)})
        with pytest.raises(ValueError):
            load_video_pairs(config, samples, data)
        return
    with pytest.raises(ValueError):
        load_video_pairs(write(), samples, data)


@pytest.mark.parametrize(
    "updates",
    [
        {"video_pair_loss_weight": 0.2},
        {"rank": 16, "alpha": 32},
        {"steps": 10},
        {"pair_loss_weight": 0.1},
        {"brier_weight": 0.1},
        {"video_pair_sidecar_sha256": "bad"},
        {"learning_rate": 1e-4},
    ],
)
def test_video_config_rejects_unmatched_experiment(tmp_path, updates):
    with pytest.raises(ValueError):
        video_config(tmp_path, **updates)


def test_zero_opt_in_leaves_config_byte_identical_and_rejects_orphan_bindings():
    plain = trainer.LoopConfig(sample_count=1200, steps=100, accumulation=12)
    zero = trainer.LoopConfig(
        sample_count=1200, steps=100, accumulation=12, video_pair_loss_weight=0
    )
    assert plain.model_dump_json() == zero.model_dump_json()
    assert not any(key.startswith("video_pair_") for key in zero.model_dump())
    with pytest.raises(ValueError):
        trainer.LoopConfig(sample_count=1200, steps=100, accumulation=12, video_pair_sidecar="x")


def test_real_cpu_loop_three_pairs_one_backward_each_and_six_ce_replays(
    course, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    samples, _, data, write, order = course
    config = write()
    rows = load_video_pairs(config, samples, data)
    weight = torch.nn.Parameter(torch.tensor([0.1, -0.05, 0.02]))
    examples = []
    for i, sample in enumerate(samples):
        gold = sample.questions[0].target.value if sample.prefix.media else "yes"
        examples.append(
            {
                "id": sample.id,
                "input_ids": [i + 1],
                "token_ids": [0, 1, 2],
                "answer_token_ids": {"yes": 0, "no": 1, "U": 2},
                "target": [float(gold == "yes"), float(gold == "no"), 0.0],
                "pixel_values_videos": torch.tensor([0.0]),
            }
        )
    terms, logits = [], []
    for offset in range(12):
        example = examples[order[offset]]
        logit = weight * example["input_ids"][0] / 1200
        logits.append(logit)
        terms.append(trainer.decision_loss(logit, example["token_ids"], example["target"]))
        if offset in (1, 3, 5):
            pi = rows[offset - 1]["positive_offset"]
            positive, negative = [logits[offset - 1 + i] for i in (pi, 1 - pi)]
            delta = positive[0] - positive[1] - negative[0] + negative[1]
            terms.append(0.1 * torch.nn.functional.softplus(-delta))
    expected = torch.autograd.grad(sum(terms) / 12, weight)[0]
    calls, gradients = [], []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = weight

        def to(self, *args):
            return self

        def forward(self, input_ids, **kwargs):
            calls.append(int(input_ids[0, 0]))
            return SimpleNamespace(logits=(weight * calls[-1] / 1200).reshape(1, 1, 3))

        def save_pretrained(self, directory, **kwargs):
            path = Path(directory)
            path.mkdir(exist_ok=True, parents=True)
            (path / "adapter_model.safetensors").write_bytes(b"cpu")

    class Optimizer:
        def __init__(self, *args, **kwargs):
            pass

        def zero_grad(self, **kwargs):
            weight.grad = None

        def step(self):
            gradients.append(weight.grad.clone())

    original = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: original(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.optim, "AdamW", Optimizer)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "CPU test")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    # Let existing forwarding move fixture tensors to the CPU selected by ids.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: Model())
        ),
    )
    monkeypatch.setattr(trainer, "attach_lora", lambda model, *a: model)
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *a, **k: tmp_path / "checkpoint")
    report = {
        "examples": 1200,
        "max_prompt_tokens": 1,
        "config": config.model_dump(),
        "model_revision": "cpu",
        "data_sha256": "cpu",
        "asset_hashes": {},
    }
    trainer.train_smoke(
        config,
        tmp_path,
        examples,
        tmp_path,
        report,
        SimpleNamespace(log=lambda *a, **k: None),
        video_pair_rows=rows,
        invocation_steps=1,
    )
    assert calls == [examples[index]["input_ids"][0] for index in order[:12]]
    assert len(gradients) == 1
    torch.testing.assert_close(gradients[0], expected)
    metric = report["step_metrics"][0]
    assert metric["video_pair_count"] == 3 and metric["video_pair_loss_weight"] == 0.1
    assert metric["gold_ce"] > 0 and metric["video_pair_loss"] > 0
    assert metric["peak_cuda_bytes"] == 1024 and metric["step_wall_seconds"] >= 0
    assert metric["grad_norm"] > 0


def test_sidecar_sha_changes_invalidate_resume_identity(course, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from train.checkpoint import save_checkpoint, verify_resume_request

    samples, _, data, write, order = course
    config = write()
    model = torch.nn.Linear(2, 3)
    checkpoint = save_checkpoint(
        tmp_path,
        step=1,
        config=config.model_dump(),
        model_revision="cpu",
        data_sha256=digest(trainer.data_bytes(samples)),
        asset_hashes={},
        order=order,
        losses=[1.0],
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
    )
    common = dict(
        model_revision="cpu", data_sha256=digest(data.read_bytes()), asset_hashes={}, order=order
    )
    verify_resume_request(tmp_path, checkpoint, config=config.model_dump(), **common)
    changed = config.model_dump() | {"video_pair_sidecar_sha256": "f" * 64}
    with pytest.raises(ValueError, match="config does not match"):
        verify_resume_request(tmp_path, checkpoint, config=changed, **common)


def test_same_media_cannot_be_rebound_as_distinct_video_pair(course):
    from scripts.s0_media_dev import source_hash_for_video

    samples, payload, _, _, order = course
    left, right = samples[order[0]], samples[order[1]]
    right.prefix.media[0].sha256 = left.prefix.media[0].sha256
    record = payload["official_questions"][right.id]
    right.meta.source_sha256 = source_hash_for_video(
        {"scene_index": int(right.meta.family_id.split(":")[1])},
        record,
        right.prefix.media[0].sha256,
    )
    metadata = json.loads(payload["matched_metadata_json"])
    metadata["rows"][0]["members"][1] = member(right, record)
    with pytest.raises(ValueError, match="distinct video families/media"):
        build_video_pairs(
            samples,
            json.dumps(metadata),
            payload["official_questions"],
            payload["header"]["ledger_sha256"],
        )
