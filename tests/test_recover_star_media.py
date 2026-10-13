"""HTTP metadata and byte-budget failures must never become whole-ZIP downloads."""

import io
import re
import zipfile
from typing import Any
from urllib.request import Request

import pytest

from scripts.recover_star_media import HTTPRangeFile


class Response:
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 206) -> None:
        self.body, self.headers, self.status = body, headers, status

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def read(self, size: int) -> bytes:
        return self.body[:size]


def range_server(monkeypatch: pytest.MonkeyPatch, data: bytes, change: str = "") -> list[Request]:
    requests = []

    def open_range(request: Request, **kwargs: Any) -> Response:
        requests.append(request)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match is not None
        start, end = map(int, match.groups())
        assert request.get_header("If-match") == '"pinned"'
        headers = {
            "Content-Range": f"bytes {start}-{end}/{len(data)}",
            "ETag": '"pinned"',
            "x-amz-version-id": "immutable-v1",
        }
        status = 206
        body = data[start : end + 1]
        if change == "status":
            status = 200
        elif change in headers:
            headers[change] = "wrong-object"
        elif change == "short":
            body = body[:-1]
        elif change == "long":
            body += b"extra"
        return Response(body, headers, status)

    monkeypatch.setattr("urllib.request.urlopen", open_range)
    return requests


def test_stdlib_zipfile_reads_selected_member_over_bounded_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("Charades_v1_480/AAAAA.mp4", b"chosen-media" * 100)
        zipped.writestr("Charades_v1_480/BBBBB.mp4", b"other-media" * 100)
    data = buffer.getvalue()
    requests = range_server(monkeypatch, data)
    source = HTTPRangeFile(
        "https://official.example/archive.zip",
        len(data),
        '"pinned"',
        "immutable-v1",
        budget=10000,
        page_size=64,
    )
    with zipfile.ZipFile(source) as zipped:
        assert zipped.read("Charades_v1_480/AAAAA.mp4") == b"chosen-media" * 100
    assert requests
    assert source.downloaded <= 10000
    assert all(request.get_header("Range") for request in requests)


@pytest.mark.parametrize("change", ["status", "ETag", "x-amz-version-id", "Content-Range"])
def test_response_must_bind_exact_range_etag_and_generation(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    range_server(monkeypatch, b"a" * 64, change)
    source = HTTPRangeFile(
        "https://official.example/zip", 64, '"pinned"', "immutable-v1", budget=100, page_size=8
    )
    with pytest.raises(ValueError, match="not_exact_pinned_object"):
        source.read(1)
    assert source.downloaded == 0


@pytest.mark.parametrize("change", ["short", "long"])
def test_body_length_must_match_range(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    range_server(monkeypatch, b"a" * 64, change)
    source = HTTPRangeFile(
        "https://official.example/zip", 64, '"pinned"', "immutable-v1", budget=100, page_size=8
    )
    with pytest.raises(ValueError, match="body_length_mismatch"):
        source.read(1)


def test_budget_rejection_precedes_network(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = range_server(monkeypatch, b"a" * 64)
    source = HTTPRangeFile(
        "https://official.example/zip", 64, '"pinned"', "immutable-v1", budget=7, page_size=8
    )
    with pytest.raises(ValueError, match="budget_exceeded"):
        source.read(1)
    assert requests == []


def test_large_parser_read_is_rejected_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = range_server(monkeypatch, b"a" * 64)
    source = HTTPRangeFile("https://official.example/zip", 16 * 1024**2, '"pinned"', "immutable-v1")
    with pytest.raises(ValueError, match="single_read_limit"):
        source.read()
    assert requests == []
