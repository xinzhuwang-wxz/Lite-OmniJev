"""Third-segment exclusion contract uses family identity, not experiment labels."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).with_name("prepare-temporal-coverage-segment003.py")
spec = importlib.util.spec_from_file_location("segment003", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sources():
    old = set(range(350))
    first = {0, 1} | set(range(350, 408))
    second = set(range(408, 468))
    return old, first, second


def test_frozen_union468_and_namespace():
    assert module.exclusion_union(*sources()) == frozenset(range(468))
    assert module.NAMESPACE == "s4-temporal-coverage-segment003:42:"


def test_overlapping_second_is_rejected():
    old, first, second = sources()
    second.remove(467)
    second.add(0)
    with pytest.raises(ValueError, match="overlap"):
        module.exclusion_union(old, first, second)


def test_missing_family_is_rejected():
    old, first, second = sources()
    second.remove(467)
    with pytest.raises(ValueError, match="counts"):
        module.exclusion_union(old, first, second)
