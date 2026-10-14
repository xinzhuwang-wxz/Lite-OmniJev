"""Builder binding tests with CPU-independent processor doubles."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.freeze_prefix_cache import freeze
from scripts.s0_prepare import read_manifest


class Row:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


@pytest.mark.parametrize("defect", [None, "image", "mapping", "prefix"])
def test_freeze_uses_exact_prepared_media_mapping_and_expanded_prefix(
    monkeypatch: pytest.MonkeyPatch,
    defect: str | None,
) -> None:
    cases = read_manifest(Path("tests/fixtures/s0-prepare-public-001/manifest.json"))
    before = [case.model_dump() for case in cases]
    selected = next(case for case in cases if case.category == "image")
    calls = []

    def prepared(case: Any, processor: Any, root: Path, *, profile: str) -> Any:
        index = len(calls)
        calls.append(case)
        png = b"\x89PNG\r\n\x1a\nSYNTHETIC" + (bytes([index]) if defect == "image" else b"")
        ids = [10, 20, index + 100]
        if defect == "prefix" and index:
            ids[1] += index
        return SimpleNamespace(
            http_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64.b64encode(png).decode(),
                            },
                        },
                        {"type": "text", "text": "common-prefix " + str(index)},
                    ],
                }
            ],
            answer_token_ids={"A": 1, "B": 2, "U": 3 + (index if defect == "mapping" else 0)},
            inputs={"input_ids": [Row(ids)], "mm_token_type_ids": [Row([0, 1, 0])]},
            summary={"probe": index},
        )

    monkeypatch.setattr("scripts.freeze_prefix_cache.prepare_case", prepared)
    if defect is not None:
        with pytest.raises(ValueError):
            freeze(cases, object(), Path("unused"))
    else:
        protocol, gateway, summary = freeze(cases, object(), Path("unused"))
        assert protocol.media_prefix_end == 2
        assert protocol.shared_prompt_token_ids == [10, 20]
        assert len(gateway["questions"]) == 10
        assert summary["source_family"] == selected.meta.family_id
        assert summary["s0_acceptance"] is False
        assert all(case.meta == selected.meta for case in calls)
        assert len({case.question.instruction for case in calls}) == 10
    assert [case.model_dump() for case in cases] == before
