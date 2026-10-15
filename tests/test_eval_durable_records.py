"""CPU doubles prove durable evidence, not GPU quality or throughput."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import s1_dev_eval as runner
from train.readout import read_probabilities


@pytest.fixture
def evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    rows = [
        {
            "id": f"sample-{i}",
            "prefix": {"state": f"state {i}", "media": []},
            "questions": [
                {
                    "qid": "q",
                    "type": "choice",
                    "instruction": "Choose",
                    "options": ["alpha", "beta"],
                    "target": {"kind": "hard", "value": "A", "source": "gold"},
                }
            ],
            "meta": {
                "modality": ["text"],
                "source_dataset": "test",
                "source_revision": "revision",
                "source_sha256": "a" * 64,
                "license": "Apache-2.0",
                "usage": "research",
                "pool": "dev",
                "family_id": str(i),
                "data_version": "test-v1",
            },
        }
        for i in range(30)
    ]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    model = SimpleNamespace()
    model.to = lambda device: model
    model.eval = lambda: model
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            use_deterministic_algorithms=lambda value: None,
            cuda=SimpleNamespace(get_device_name=lambda index: "test-double"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **k: model)
        ),
    )
    monkeypatch.setattr(runner, "OfficialTokenizerAdapter", lambda path: None)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda package: "test")
    monkeypatch.setattr(runner, "forward_readout", predict)
    return dict(
        model_dir=tmp_path / "model-revision",
        adapter_dir=None,
        data_path=data_path,
        output_dir=tmp_path / "output",
        report_out=tmp_path / "report.json",
        max_length=2048,
        passes=2,
        command=["test"],
        counterfactual=True,
    )


def predict(*args, **kwargs):
    logits = {"A": 8.0, "B": 0.0, "U": -8.0}
    return logits, read_probabilities(logits, ["A", "B"], temperature=1.0)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError, ValueError])
@pytest.mark.parametrize("completed", [27, 35, 63])
def test_interruption_retains_completed_rows(evaluation, monkeypatch, failure, completed):
    count = 0

    def interrupted(*args, **kwargs):
        nonlocal count
        if count == completed:
            raise failure("interrupted")
        count += 1
        return predict()

    monkeypatch.setattr(runner, "forward_readout", interrupted)
    if failure is ValueError:
        assert runner.run(**evaluation) == 1
    else:
        with pytest.raises(failure):
            runner.run(**evaluation)
    output = evaluation["output_dir"]
    rows = [
        json.loads(line) for line in (output / "records.partial.jsonl").read_text().splitlines()
    ]
    assert len(rows) == completed
    assert [row["pass"] for row in rows] == [i // 60 for i in range(completed)]
    assert rows[-1]["variant"] == ("counterfactual" if completed == 35 else "original")
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "incomplete"
    assert progress["records_completed"] == completed
    assert progress["passes_completed"] == completed // 60
    assert not (output / "records.jsonl").exists()
    report = evaluation["report_out"]
    assert not report.exists() or json.loads(report.read_text())["status"] == "blocked"


def test_live_flush_bound_and_success_preserve_final_records(evaluation, monkeypatch):
    count = 0
    output = evaluation["output_dir"]

    def observe(*args, **kwargs):
        nonlocal count
        if count == 25:
            assert len((output / "records.partial.jsonl").read_text().splitlines()) == 25
            progress = json.loads((output / "progress.json").read_text())
            assert progress["records_completed"] == 25
            assert progress["status"] == "incomplete"
        if count == 60:
            progress = json.loads((output / "progress.json").read_text())
            assert progress["records_completed"] == 60
            assert progress["passes_completed"] == 1
        count += 1
        return predict()

    monkeypatch.setattr(runner, "forward_readout", observe)
    assert runner.run(**evaluation) == 0
    final = (output / "records.jsonl").read_bytes()
    assert final == (output / "records.partial.jsonl").read_bytes()
    rows = [json.loads(line) for line in final.splitlines()]
    assert len(rows) == 120
    assert [(row["pass"], row["variant"], row["id"]) for row in rows] == [
        (p, variant, f"sample-{i}")
        for p in range(2)
        for variant in ("original", "counterfactual")
        for i in range(30)
    ]
    originals = []
    for sample in runner.load_samples(evaluation["data_path"]):
        logits, readout = predict()
        originals.append({**runner.record_for(sample, readout, logits), "variant": "original"})
    assert final.splitlines()[:30] == [
        runner.canonical_json({**row, "pass": 0}).encode() for row in originals
    ]
    report = json.loads(evaluation["report_out"].read_text())
    assert report["status"] == "passed"
    assert report["repeat_identical"] is True
    for key, value in runner.metrics_from_records(originals).items():
        assert report["metrics"][key] == value
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["passes_completed"] == 2
    identity = progress["identity"]
    for key in (
        "data_sha256",
        "adapter_sha256",
        "model_revision",
        "script_sha256",
        "image_max_soft_tokens",
        "video_num_frames",
    ):
        assert identity[key] == report[key]


def test_existing_output_is_rejected_without_modification(evaluation):
    output = evaluation["output_dir"]
    output.mkdir()
    sentinel = output / "records.partial.jsonl"
    sentinel.write_text("existing evidence\n")
    with pytest.raises(FileExistsError):
        runner.run(**evaluation)
    assert sentinel.read_text() == "existing evidence\n"
    assert not evaluation["report_out"].exists()


def test_report_write_failure_keeps_progress_incomplete(evaluation):
    evaluation["report_out"].mkdir()
    with pytest.raises(IsADirectoryError):
        runner.run(**evaluation)
    output = evaluation["output_dir"]
    assert len((output / "records.partial.jsonl").read_text().splitlines()) == 120
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "incomplete"


def test_live_original_metrics_and_loading_phases(evaluation, monkeypatch):
    output = evaluation["output_dir"]
    seen = 0

    def loading(*args, **kwargs):
        progress = json.loads((output / "progress.json").read_text())
        assert progress["phase"] == "loading_model"
        assert progress["original_completed"] == 0
        assert progress["original_total"] == 30
        model = SimpleNamespace()
        model.to = lambda device: model
        model.eval = lambda: model
        return model

    monkeypatch.setattr(
        sys.modules["transformers"].Gemma4ForConditionalGeneration, "from_pretrained", loading
    )

    def observe(*args, **kwargs):
        nonlocal seen
        if seen == 25:
            progress = json.loads((output / "progress.json").read_text())
            assert progress["phase"] == "evaluating"
            assert progress["original_completed"] == progress["original_correct"] == 25
            assert progress["original_accuracy"] == 1
            assert progress["elapsed_seconds"] >= progress["last_original_batch_seconds"] >= 0
            assert "sample-" not in json.dumps(progress)
        if seen == 35:
            # Five counterfactual records do not change original correctness or denominator.
            progress = json.loads((output / "progress.json").read_text())
            assert progress["original_completed"] == 25  # last original batch checkpoint
            raise RuntimeError("stop within CF")
        seen += 1
        return predict()

    monkeypatch.setattr(runner, "forward_readout", observe)
    with pytest.raises(RuntimeError):
        runner.run(**evaluation)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["original_completed"] == progress["original_correct"] == 30
    assert progress["original_accuracy"] == 1
    assert progress["records_completed"] == 35
    assert progress["status"] == "incomplete"


@pytest.mark.parametrize("pool", ["calibration", "holdout"])
def test_private_pool_creates_no_public_progress_or_partial_journal(evaluation, pool):
    path = evaluation["data_path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["meta"]["pool"] = pool
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert runner.run(**evaluation) == 0
    assert not (evaluation["output_dir"] / "progress.json").exists()
    assert not (evaluation["output_dir"] / "records.partial.jsonl").exists()
