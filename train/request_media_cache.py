"""Request-local image preprocessing reuse for official Transformers processors."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any


@dataclass
class RequestImageCacheStats:
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    output_bypasses: int = 0


@dataclass(frozen=True)
class _ImageDigest:
    mode: str
    size: tuple[int, int]
    sha256: str


@dataclass(frozen=True)
class _CacheKey:
    layout: Any
    images: tuple[_ImageDigest, ...]
    kwargs: str


_MAX_IMAGES = 8
_MAX_TOTAL_PIXELS = 40_000_000
_MAX_IMAGE_BYTES = 256 * 1024 * 1024
_MAX_OUTPUT_BYTES = 32 * 1024 * 1024


def request_image_processor(processor: Any) -> Any:
    """Return a shallow processor copy with a one-entry, request-local image cache."""
    official = getattr(processor, "_process_images", None)
    if official is None or not callable(official):
        raise ValueError("processor does not expose official _process_images")
    try:
        wrapped = copy.copy(processor)
    except Exception as exc:  # pragma: no cover - depends on third-party processor internals.
        raise ValueError("processor cannot be shallow-copied") from exc
    if wrapped is processor:
        raise ValueError("processor shallow copy returned the original instance")

    official_copy = getattr(wrapped, "_process_images", None)
    if official_copy is None or not callable(official_copy):
        raise ValueError("processor copy does not expose official _process_images")

    stats = RequestImageCacheStats()
    lock = threading.Lock()
    cache: dict[str, tuple[_CacheKey, Any] | None] = {"entry": None}

    def cached_process_images(self: Any, images: Any, **kwargs: Any) -> Any:
        key = _cache_key(images, kwargs)
        if key is None:
            with lock:
                stats.bypasses += 1
            return official_copy(images, **kwargs)

        with lock:
            entry = cache["entry"]
            if entry is not None and entry[0] == key:
                stats.hits += 1
                return copy.deepcopy(entry[1])

        result = official_copy(images, **kwargs)
        if not _cacheable_output(result):
            with lock:
                stats.bypasses += 1
                stats.output_bypasses += 1
            return result
        cached = copy.deepcopy(result)
        with lock:
            stats.misses += 1
            cache["entry"] = (key, cached)
            return copy.deepcopy(cached)

    wrapped._process_images = MethodType(cached_process_images, wrapped)
    wrapped.request_image_cache_stats = stats
    return wrapped


def _cache_key(images: Any, kwargs: Mapping[str, Any]) -> _CacheKey | None:
    layout, flattened = _pil_image_tree(images)
    if layout is None or not flattened:
        return None
    stable_kwargs = _stable_json(kwargs)
    if stable_kwargs is None:
        return None

    total_pixels = 0
    total_bytes = 0
    for index, image in enumerate(flattened):
        if image.mode != "RGB":
            return None
        width, height = image.size
        pixels = width * height
        total_pixels += pixels
        total_bytes += pixels * len(image.getbands())
        if (
            index >= _MAX_IMAGES
            or total_pixels > _MAX_TOTAL_PIXELS
            or total_bytes > _MAX_IMAGE_BYTES
        ):
            return None

    digests: list[_ImageDigest] = []
    for image in flattened:
        width, height = image.size
        digests.append(
            _ImageDigest(
                mode=image.mode,
                size=(width, height),
                sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            )
        )
    return _CacheKey(layout=layout, images=tuple(digests), kwargs=stable_kwargs)


def _pil_image_tree(images: Any) -> tuple[Any | None, list[Any]]:
    try:
        image_class = __import__("PIL.Image", fromlist=["Image"]).Image
    except ImportError:
        return None, []
    if isinstance(images, image_class):
        return "image", [images]
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        nodes: list[Any] = []
        flattened: list[Any] = []
        for item in images:
            nested_tree, nested_images = _pil_image_tree(item)
            if nested_tree is None:
                return None, []
            nodes.append(nested_tree)
            flattened.extend(nested_images)
        return tuple(nodes), flattened
    return None, []


def _cacheable_output(value: Any) -> bool:
    size = _estimated_output_bytes(value)
    return size is not None and size <= _MAX_OUTPUT_BYTES


def _estimated_output_bytes(value: Any) -> int | None:
    if value is None or isinstance(value, bool | int | float):
        return 0
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, bytes | bytearray | memoryview):
        return len(value)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    if isinstance(value, Mapping):
        total = 0
        for key, item in value.items():
            key_size = _estimated_output_bytes(key)
            item_size = _estimated_output_bytes(item)
            if key_size is None or item_size is None:
                return None
            total += key_size + item_size
        return total
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        total = 0
        for item in value:
            item_size = _estimated_output_bytes(item)
            if item_size is None:
                return None
            total += item_size
        return total
    return None


def _stable_json(value: Any) -> str | None:
    try:
        stable = _stable_value(value)
        return json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_stable_value(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("kwargs keys must be strings")
            result[key] = _stable_value(item)
        return result
    raise TypeError("unstable value")
