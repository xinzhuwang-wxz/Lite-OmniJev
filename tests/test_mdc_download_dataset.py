"""Public download failures preserve data and do not disclose authentication material."""

import hashlib
import io
import os
import socket
import urllib.error
import urllib.request

import pytest

from scripts import mdc_download_dataset as mdc


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "mdc.key"
    path.write_text("synthetic-test-credential-only\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )


def asset_transport(monkeypatch, payload):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    monkeypatch.setattr(
        mdc,
        "request_json",
        lambda *args: {"downloadUrl": "https://cdn.example/data?signature=fake"},
    )
    return calls


def download(key_file, output, expected_payload):
    return mdc.download_dataset(
        dataset_id="synthetic-dataset",
        key_file=key_file,
        output_path=output,
        expected_size_bytes=len(expected_payload),
        expected_sha256=hashlib.sha256(expected_payload).hexdigest(),
    )


def test_verified_archive_uses_unauthenticated_asset_request(
    tmp_path,
    key_file,
    public_dns,
    monkeypatch,
):
    calls = asset_transport(monkeypatch, b"verified-public-data")
    output = tmp_path / "archive"
    report = download(key_file, output, b"verified-public-data")
    assert report["status"] == "downloaded_verified"
    assert output.read_bytes() == b"verified-public-data"
    assert all(isinstance(call, str) for call in calls)
    assert "signature" not in str(report)
    assert key_file.read_text().strip() not in str(report)
    assert not list(tmp_path.glob(".archive.*.partial"))


@pytest.mark.parametrize("payload", [b"different", b"x", b"much-longer-than-expected"])
def test_bad_size_or_hash_preserves_existing_destination(
    tmp_path,
    key_file,
    public_dns,
    monkeypatch,
    payload,
):
    asset_transport(monkeypatch, payload)
    output = tmp_path / "archive"
    output.write_bytes(b"previous-good")
    with pytest.raises(mdc.DownloadError):
        download(key_file, output, b"expected!")
    assert output.read_bytes() == b"previous-good"
    assert not list(tmp_path.glob(".archive.*.partial"))


def test_success_does_not_replace_existing_user_file(tmp_path, key_file, public_dns, monkeypatch):
    asset_transport(monkeypatch, b"expected")
    output = tmp_path / "archive"
    output.write_bytes(b"previous-good")
    with pytest.raises(mdc.DownloadError, match="output_already_exists"):
        download(key_file, output, b"expected")
    assert output.read_bytes() == b"previous-good"


def test_secret_symlink_fifo_and_oversize_are_rejected(tmp_path, key_file):
    link = tmp_path / "link"
    link.symlink_to(key_file)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"x" * (mdc.MAX_KEY_BYTES + 1))
    oversized.chmod(0o600)
    for path in (link, fifo, oversized):
        with pytest.raises(mdc.DownloadError, match="private_key_file_rejected"):
            mdc.read_secret(path)


def test_secret_permissions_and_header_injection_rejected(key_file):
    key_file.chmod(0o644)
    with pytest.raises(mdc.DownloadError, match="private_key_file_rejected"):
        mdc.read_secret(key_file)
    key_file.chmod(0o600)
    key_file.write_text("synthetic-key\nInjected: private-value")
    with pytest.raises(mdc.DownloadError) as error:
        mdc.read_secret(key_file)
    assert "private-value" not in str(error.value)


def test_api_error_body_and_signed_url_are_redacted(monkeypatch):
    secret = "synthetic-private-credential"
    body = secret + " https://cdn.example/data?signature=private-signed-url"

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, body, {}, io.BytesIO(body.encode()))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(mdc.DownloadError) as error:
        mdc.request_json(mdc.API_BASE_URL + "/datasets/synthetic/download", secret)
    assert str(error.value) == "mdc_api_http_403"
    assert error.value.__suppress_context__


def test_api_only_accepts_official_origin_before_sending_key():
    for url in [
        "https://example.com/api/datasets/id/download",
        "http://127.0.0.1/api/datasets/id/download",
    ]:
        with pytest.raises(mdc.DownloadError, match="api_origin_rejected"):
            mdc.request_json(url, "synthetic-key")


def test_authenticated_redirect_is_refused():
    request = urllib.request.Request(
        mdc.API_BASE_URL + "/datasets/id/download",
        data=b"{}",
        headers={"Authorization": "Bearer synthetic-key"},
    )
    with pytest.raises(mdc.DownloadError, match="redirect_refused"):
        mdc.RejectRedirect().redirect_request(
            request, None, 302, "Found", {}, "https://other.example"
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/private",
        "http://example.com/archive",
        "https://user:password@example.com/archive",
    ],
)
def test_non_https_and_userinfo_assets_rejected(url):
    with pytest.raises(mdc.DownloadError, match="asset_url_rejected"):
        mdc.validate_asset_url(url)


def test_private_asset_addresses_rejected(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(mdc.DownloadError, match="asset_url_rejected"):
        mdc.validate_asset_url("https://private.example/archive")
