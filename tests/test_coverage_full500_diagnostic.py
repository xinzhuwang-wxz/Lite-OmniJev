"""Public comparison boundaries; probability math reuses tested paired/ranking helpers."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / ".scratch/s4/compare-temporal-coverage-full500.py"
spec = importlib.util.spec_from_file_location("full500_comparison", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_forward_comparison_ignores_format_and_progress_function():
    old = "def forward_readout(x):\n return x\n"
    new = "def progress():\n return 25\ndef forward_readout(x):\n    return x\n"
    assert module.same_forward(old, new)


def test_changed_forward_is_not_declared_equal():
    assert not module.same_forward(
        "def forward_readout(x):\n return x\n", "def forward_readout(x):\n return x + 1\n"
    )


def test_misaligned_source_order_fails_before_scoring():
    with pytest.raises(ValueError, match="IDs/order"):
        module.analyze(
            [SimpleNamespace(id="original")], [{"id": "different"}], [{"id": "original"}]
        )
