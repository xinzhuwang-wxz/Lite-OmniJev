from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from train.request_media_cache import request_image_processor


class FakeImage:
    def __init__(
        self, color: tuple[int, ...] = (1, 2, 3), size: tuple[int, int] = (2, 2), mode: str = "RGB"
    ) -> None:
        self.color = color
        self.size = size
        self.mode = mode

    def getbands(self) -> tuple[str, ...]:
        return tuple(self.mode)

    def tobytes(self) -> bytes:
        return bytes(self.color) * (self.size[0] * self.size[1])

    def getpixel(self, _position: tuple[int, int]) -> tuple[int, ...]:
        return self.color

    def convert(self, mode: str) -> FakeImage:
        if mode == "L":
            return FakeImage((sum(self.color) // len(self.color),), self.size, "L")
        return FakeImage(self.color, self.size, mode)


class ExplodingBytesImage(FakeImage):
    def tobytes(self) -> bytes:
        raise AssertionError("oversized image must bypass before pixel hashing")


@pytest.fixture(autouse=True)
def fake_pil(monkeypatch: pytest.MonkeyPatch) -> None:
    pil_package = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.__dict__["Image"] = FakeImage
    pil_package.__dict__["Image"] = pil_image
    monkeypatch.setitem(sys.modules, "PIL", pil_package)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)


def rgb_image(color: tuple[int, int, int] = (1, 2, 3), size: tuple[int, int] = (2, 2)) -> Any:
    return FakeImage(color, size)


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = 0
        self.tokenizer = object()

    def _process_images(self, images: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        self.calls += 1
        flat = _flatten(images)
        return (
            {
                "processed_images": [
                    {
                        "mode": getattr(image, "mode", None),
                        "size": getattr(image, "size", None),
                        "first_pixel": image.getpixel((0, 0))
                        if hasattr(image, "getpixel")
                        else None,
                        "kwargs": dict(kwargs),
                        "call": self.calls,
                    }
                    for image in flat
                ]
            },
            [f"replacement-{index}-call-{self.calls}" for index, _ in enumerate(flat)],
        )

    def __call__(self, *, images: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        return self._process_images(images, **kwargs)


class FakeTensorBytes:
    def __init__(self, size: int) -> None:
        self.size = size

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1


class LargeOutputProcessor(FakeProcessor):
    def _process_images(self, images: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        self.calls += 1
        return {"processed_images": FakeTensorBytes(33 * 1024 * 1024)}, ["large"]


class SelfCopyProcessor(FakeProcessor):
    def __copy__(self) -> SelfCopyProcessor:
        return self


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list | tuple):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def test_same_media_across_questions_hits_request_cache() -> None:
    processor = FakeProcessor()
    wrapped = request_image_processor(processor)
    image = rgb_image()

    first = wrapped(images=[[image]], images_kwargs={"do_normalize": True})
    second = wrapped(images=[[image]], images_kwargs={"do_normalize": True})

    assert processor.calls == 0
    assert wrapped.calls == 1
    assert first == second
    assert wrapped.request_image_cache_stats.hits == 1
    assert wrapped.request_image_cache_stats.misses == 1
    assert wrapped.request_image_cache_stats.bypasses == 0


def test_pixel_and_kwargs_changes_miss_without_false_hits() -> None:
    wrapped = request_image_processor(FakeProcessor())
    image = rgb_image((1, 2, 3))
    changed = rgb_image((4, 5, 6))

    wrapped(images=[[image]], images_kwargs={"do_normalize": True})
    wrapped(images=[[changed]], images_kwargs={"do_normalize": True})
    wrapped(images=[[changed]], images_kwargs={"do_normalize": False})

    assert wrapped.calls == 3
    assert wrapped.request_image_cache_stats.hits == 0
    assert wrapped.request_image_cache_stats.misses == 3


def test_nested_image_layout_change_misses_even_with_same_flat_pixels() -> None:
    wrapped = request_image_processor(FakeProcessor())
    first = rgb_image((1, 2, 3))
    second = rgb_image((4, 5, 6))

    wrapped(images=[[first, second]])
    wrapped(images=[[first], [second]])
    wrapped(images=[[first], [second]])

    assert wrapped.calls == 2
    assert wrapped.request_image_cache_stats.hits == 1
    assert wrapped.request_image_cache_stats.misses == 2


def test_original_processor_is_not_modified_and_call_path_is_preserved() -> None:
    processor = FakeProcessor()
    original_process = processor._process_images
    wrapped = request_image_processor(processor)

    assert processor._process_images == original_process
    assert not hasattr(processor, "request_image_cache_stats")
    assert wrapped.tokenizer is processor.tokenizer
    assert wrapped(images=[[rgb_image()]])[0]["processed_images"][0]["call"] == 1


def test_deepcopy_boundary_prevents_return_mutation_from_polluting_cache() -> None:
    wrapped = request_image_processor(FakeProcessor())
    image = rgb_image()

    first = wrapped(images=[[image]])
    first[0]["processed_images"][0]["first_pixel"] = "mutated"
    first[1].append("mutated")
    second = wrapped(images=[[image]])

    assert second[0]["processed_images"][0]["first_pixel"] == (1, 2, 3)
    assert second[1] == ["replacement-0-call-1"]
    assert wrapped.calls == 1
    assert wrapped.request_image_cache_stats.hits == 1


def test_single_entry_eviction_and_new_factory_do_not_share_cache() -> None:
    processor = FakeProcessor()
    wrapped = request_image_processor(processor)
    first = rgb_image((1, 2, 3))
    second = rgb_image((7, 8, 9))

    wrapped(images=[[first]])
    wrapped(images=[[second]])
    wrapped(images=[[first]])

    assert wrapped.calls == 3
    assert wrapped.request_image_cache_stats.misses == 3

    fresh = request_image_processor(processor)
    fresh(images=[[first]])
    assert fresh.calls == 1
    assert fresh.request_image_cache_stats.hits == 0
    assert fresh.request_image_cache_stats.misses == 1


def test_large_processed_output_bypasses_without_caching_or_deepcopying() -> None:
    wrapped = request_image_processor(LargeOutputProcessor())
    image = rgb_image()

    first = wrapped(images=[[image]])
    second = wrapped(images=[[image]])

    assert first[0]["processed_images"] is not second[0]["processed_images"]
    assert wrapped.calls == 2
    assert wrapped.request_image_cache_stats.bypasses == 2
    assert wrapped.request_image_cache_stats.output_bypasses == 2
    assert wrapped.request_image_cache_stats.hits == 0
    assert wrapped.request_image_cache_stats.misses == 0


def test_unsupported_processor_rejected() -> None:
    with pytest.raises(ValueError, match="_process_images"):
        request_image_processor(object())


def test_processor_copy_must_not_return_original_instance() -> None:
    processor = SelfCopyProcessor()

    with pytest.raises(ValueError, match="original instance"):
        request_image_processor(processor)

    assert not hasattr(processor, "request_image_cache_stats")


def test_non_pil_non_rgb_or_unstable_kwargs_bypass_without_fake_hits() -> None:
    wrapped = request_image_processor(FakeProcessor())
    non_rgb = rgb_image().convert("L")
    rgb = rgb_image()

    wrapped(images=[[non_rgb]])
    wrapped(images=[["not an image"]])
    wrapped(images=[[rgb]], image_token=object())

    assert wrapped.calls == 3
    assert wrapped.request_image_cache_stats.bypasses == 3
    assert wrapped.request_image_cache_stats.output_bypasses == 0
    assert wrapped.request_image_cache_stats.hits == 0
    assert wrapped.request_image_cache_stats.misses == 0


def test_oversized_input_bypasses_before_hashing_pixels() -> None:
    wrapped = request_image_processor(FakeProcessor())
    oversized = ExplodingBytesImage(size=(40_000_001, 1))

    result = wrapped(images=[[oversized]])

    assert result[0]["processed_images"][0]["size"] == (40_000_001, 1)
    assert wrapped.calls == 1
    assert wrapped.request_image_cache_stats.bypasses == 1
    assert wrapped.request_image_cache_stats.hits == 0
    assert wrapped.request_image_cache_stats.misses == 0
