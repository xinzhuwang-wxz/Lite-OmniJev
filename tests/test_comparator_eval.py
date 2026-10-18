"""Behavioral tests for the comparator harness: contracts in, reports out."""

import importlib
import json
import types
from pathlib import Path

import pytest
import torch

from data.schema import HardTarget, Media, Metadata, Prefix, Question, Sample
from scripts.comparator_eval import DevEvalError, PreparedRow, letter_mapping, main, score_row

_REAL_IMPORT = importlib.import_module


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


class FakeModel(torch.nn.Module):
    """Last-position logits table indexed by token id."""

    def __init__(self, table: dict[int, float]) -> None:
        super().__init__()
        self.table = table

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> types.SimpleNamespace:
        row = torch.zeros(256)
        for token_id, value in self.table.items():
            row[token_id] = value
        return types.SimpleNamespace(logits=row.unsqueeze(0).unsqueeze(0))


class FakeAdapter:
    def load(self, model_dir: Path) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            tokenizer=FakeTokenizer(), model=FakeModel({65: 3.0, 66: 1.0, 85: 0.5})
        )

    def prepare(
        self, loaded: types.SimpleNamespace, sample: Sample, media_root: Path, *, max_length: int
    ) -> PreparedRow:
        return PreparedRow(input_ids=[112], prompt_text="p", forward_kwargs={})


def install_fake_family(monkeypatch: pytest.MonkeyPatch, adapter: FakeAdapter) -> None:
    def fake_import(name: str) -> object:
        if name == "torch":
            return _REAL_IMPORT(name)
        return types.SimpleNamespace(build=lambda: adapter)

    monkeypatch.setattr(importlib, "import_module", fake_import)


def sample_row() -> Sample:
    return Sample(
        id="mmau-test-mini:test-0",
        prefix=Prefix(
            state="Answer the multiple-choice question using the attached audio.",
            media=[Media(type="audio", uri="audio.wav", sha256="a" * 64)],
        ),
        questions=[
            Question(
                qid="q0",
                type="choice",
                instruction="Which is it?",
                options=["alpha", "beta"],
                target=HardTarget(kind="hard", value="A", source="test"),
            )
        ],
        meta=Metadata(
            modality=["audio"],
            source_dataset="test",
            source_revision="r1",
            source_sha256="0" * 64,
            license="CC0-1.0",
            usage="research",
            pool="dev",
            family_id="test-0",
            data_version="test-v1",
        ),
    )


def write_contract(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text(sample_row().model_dump_json() + "\n", encoding="utf-8")
    return path


def cli_args(tmp_path: Path, data: Path, out: str, *, with_limit: bool = True) -> list[str]:
    args = [
        "--family",
        "fake",
        "--model-dir",
        str(tmp_path),
        "--data",
        str(data),
        "--media-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / out),
        "--report",
        str(tmp_path / f"{out}-report.json"),
        "--device",
        "cpu",
    ]
    if with_limit:
        args += ["--limit", "5"]
    return args


def test_scored_run_requires_smoke_or_limit(tmp_path: Path) -> None:
    data = write_contract(tmp_path)
    with pytest.raises(SystemExit, match="smoke"):
        main(cli_args(tmp_path, data, "out", with_limit=False))


def test_end_to_end_report_and_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = write_contract(tmp_path)
    install_fake_family(monkeypatch, FakeAdapter())
    report_path = tmp_path / "out-report.json"
    code = main(cli_args(tmp_path, data, "out"))
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["harness"] == "comparator-eval-v1"
    assert report["scored"] == 1
    assert report["metrics"]["accuracy"] == 1.0
    records = [json.loads(line) for line in (tmp_path / "out" / "records.jsonl").open()]
    assert records[0]["correct"] is True
    assert records[0]["predicted"] == "A"


def test_score_row_reads_last_position_logits() -> None:
    model = FakeModel({65: 3.0, 66: 1.0, 85: 0.5})
    logits = score_row(model, [112], {"A": 65, "B": 66, "U": 85}, {}, device="cpu")
    assert logits == {"A": 3.0, "B": 1.0, "U": 0.5}


def test_letter_mapping_rejects_broken_boundary() -> None:
    class BreakingTokenizer(FakeTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            if text.endswith("A"):
                return [ord(char) + 2 for char in text]
            return super().encode(text, add_special_tokens=add_special_tokens)

    sample = sample_row()
    with pytest.raises(DevEvalError, match="boundary"):
        letter_mapping(BreakingTokenizer(), sample, [112], "p")


def test_letter_mapping_rejects_multi_token_letter() -> None:
    class MultiTokenizer(FakeTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            if text == "A":
                return [65, 66]
            return super().encode(text, add_special_tokens=add_special_tokens)

    sample = sample_row()
    with pytest.raises(DevEvalError, match="single token"):
        letter_mapping(MultiTokenizer(), sample, [112], "p")


def test_boundary_mismatch_becomes_protocol_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MismatchAdapter(FakeAdapter):
        def prepare(
            self,
            loaded: types.SimpleNamespace,
            sample: Sample,
            media_root: Path,
            *,
            max_length: int,
        ) -> PreparedRow:
            return PreparedRow(input_ids=[999], prompt_text="p", forward_kwargs={})

    data = write_contract(tmp_path)
    install_fake_family(monkeypatch, MismatchAdapter())
    report_path = tmp_path / "v-report.json"
    main(cli_args(tmp_path, data, "v"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scored"] == 0
    assert report["protocol_violations"] == 1
