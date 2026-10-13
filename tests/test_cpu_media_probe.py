from pathlib import Path

import pytest

from scripts import cpu_media_probe


def test_missing_model_dir_blocks_before_processor(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"
    output = tmp_path / "report.json"
    assert (
        cpu_media_probe.main(
            [
                "--model-dir",
                str(missing),
                "--work-dir",
                str(tmp_path / "work"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert '"status": "blocked"' in output.read_text()


def test_answer_boundary_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadTokenizer:
        def __init__(self) -> None:
            self.config = {"unk_token": "<unk>"}
            self.tokenizer = self

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            if text in {"A", "B", "U", "0", "1", "yes", "no"}:
                return [1]
            return [ord(char) for char in text]

        def token_to_id(self, token: str) -> int:
            return -1

        def get_added_tokens_decoder(self) -> dict[int, object]:
            return {}

        def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
            return "not-the-token"

    bindings = cpu_media_probe.answer_boundary_bindings(BadTokenizer(), "prompt")
    assert bindings["_failures"]
    assert bindings["choice"]["failures"]


def test_tensor_summary_records_shape_dtype_and_hash() -> None:
    np = pytest.importorskip("numpy")
    summary = cpu_media_probe.tensor_summary(np.asarray([[1, 2]], dtype=np.int64))
    assert summary["shape"] == [1, 2]
    assert summary["dtype"] == "int64"
    assert len(summary["sha256"]) == 64
