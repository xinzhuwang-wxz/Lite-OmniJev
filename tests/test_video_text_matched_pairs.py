"""Matching capacity requires exact text/program, opposite gold and distinct media/family."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / ".scratch/s4/inspect-video-text-matched-pairs.py"
spec = importlib.util.spec_from_file_location("text_matches", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(family, gold, text="same", program=None, media=None):
    return {
        "family": family,
        "gold": gold,
        "text": text,
        "program": program or ["objects", "exist"],
        "media": media or family,
    }


def test_exact_matching_rejects_similar_text_program_or_same_media():
    assert (
        module.capacity([row("a", "yes"), row("b", "no")])["max_disjoint_distinct_media_pairs"] == 1
    )
    for other in [
        row("b", "no", text="same?"),
        row("b", "no", program=["different"]),
        row("b", "no", media="a"),
    ]:
        assert module.capacity([row("a", "yes"), other])["max_disjoint_distinct_media_pairs"] == 0
    with pytest.raises(ValueError, match="family"):
        module.capacity([row("a", "yes"), row("a", "no")])


def test_capacity_is_disjoint_matching_not_cartesian_count():
    rows = [row(str(i), "yes" if i < 3 else "no") for i in range(5)]
    assert module.capacity(rows)["max_disjoint_distinct_media_pairs"] == 2
