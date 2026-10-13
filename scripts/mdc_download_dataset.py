"""Download MDC data after human authorization, without logging credentials or URLs.

Asset archive requests intentionally disable proxy use so the validated public
origin address is the address connected by the TLS socket.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

API_BASE_URL = "https://mozilladatacollective.com/api"
MAX_KEY_BYTES = 8192
MAX_METADATA_BYTES = 65536
AddressInfo = tuple[int, int, int, str, tuple[Any, ...]]


class DownloadError(RuntimeError):
    """A fixed error code safe for public failure reports."""


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise DownloadError("redirect_refused")


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, pinned_addresses: list[AddressInfo], **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._pinned_addresses = pinned_addresses
        self._create_connection = self._create_pinned_connection

    def _create_pinned_connection(
        self,
        address: tuple[str, int],
        timeout: Any = None,
        source_address: tuple[str, int] | None = None,
        *,
        all_errors: bool = False,
    ) -> socket.socket:
        errors: list[Exception] = []
        for family, socktype, proto, _canonname, sockaddr in self._pinned_addresses:
            sock = socket.socket(family, socktype, proto)
            try:
                if isinstance(timeout, (int, float)):
                    sock.settimeout(float(timeout))
                if source_address is not None:
                    sock.bind(source_address)
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                errors.append(exc)
                sock.close()
        if all_errors:
            raise ExceptionGroup("asset_connection_failed", errors)
        if errors:
            raise errors[-1]
        raise OSError("asset_connection_failed")


class PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, pinned_addresses: list[AddressInfo]) -> None:
        super().__init__()
        self._context: Any
        self._pinned_addresses = pinned_addresses

    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(self._connection_factory, req, context=self._context)

    def _connection_factory(self, host: str, **kwargs: Any) -> PinnedHTTPSConnection:
        return PinnedHTTPSConnection(host, self._pinned_addresses, **kwargs)


def read_secret(path: Path) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) & 0o177
                or info.st_uid != os.geteuid()
                or not 0 < info.st_size <= MAX_KEY_BYTES
            ):
                raise DownloadError("private_key_file_rejected")
            raw = os.read(descriptor, MAX_KEY_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw) > MAX_KEY_BYTES:
            raise DownloadError("private_key_file_rejected")
        secret = raw.decode("ascii").strip()
        if not secret or any(not 33 <= ord(character) <= 126 for character in secret):
            raise DownloadError("private_key_format_rejected")
        return secret
    except (OSError, UnicodeError):
        raise DownloadError("private_key_file_rejected") from None


def request_json(url: str, api_key: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "mozilladatacollective.com"
        or not re.fullmatch(r"/api/datasets/[A-Za-z0-9_-]{1,128}/download", parsed.path)
        or parsed.query
        or parsed.fragment
    ):
        raise DownloadError("api_origin_rejected")
    request = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.build_opener(RejectRedirect()).open(request, timeout=60) as response:
            raw = response.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise DownloadError("api_metadata_too_large")
        payload = json.loads(raw)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise DownloadError(f"mdc_api_http_{code}") from None
    except (urllib.error.URLError, OSError, ValueError, UnicodeError, RecursionError):
        raise DownloadError("mdc_api_request_failed") from None
    if not isinstance(payload, dict):
        raise DownloadError("api_metadata_invalid")
    return cast(dict[str, Any], payload)


def validate_asset_url(url: str) -> list[AddressInfo]:
    try:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname
        if (
            parsed.scheme != "https"
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or parsed.fragment
        ):
            raise DownloadError("asset_url_rejected")
        addresses = cast(
            list[AddressInfo],
            socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM),
        )
        if not addresses or any(
            not ipaddress.ip_address(address[4][0]).is_global for address in addresses
        ):
            raise DownloadError("asset_url_rejected")
        return addresses
    except (ValueError, OSError):
        raise DownloadError("asset_url_rejected") from None


def stream_download(
    url: str,
    output_path: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
) -> tuple[int, str]:
    pinned_addresses = validate_asset_url(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_path.parent).free < expected_size_bytes + 64 * 1024**2:
        raise DownloadError("insufficient_disk")
    staged: Path | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent, delete=False
        ) as handle:
            staged = Path(handle.name)
            # A separate unauthenticated request; never send API credentials to asset hosts.
            opener = urllib.request.build_opener(
                RejectRedirect(),
                PinnedHTTPSHandler(pinned_addresses),
                urllib.request.ProxyHandler({}),
            )
            with opener.open(url, timeout=60) as response:
                while True:
                    chunk = response.read(min(1024**2, expected_size_bytes - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_size_bytes:
                        raise DownloadError("archive_size_mismatch")
                    digest.update(chunk)
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if total != expected_size_bytes:
            raise DownloadError("archive_size_mismatch")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise DownloadError("archive_sha256_mismatch")
        # Atomic exclusive publication cannot overwrite an existing file, including races.
        os.link(staged, output_path)
        return total, actual_sha256
    except FileExistsError:
        raise DownloadError("output_already_exists") from None
    except (urllib.error.URLError, OSError, ValueError):
        raise DownloadError("archive_request_failed") from None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def download_dataset(
    *,
    dataset_id: str,
    key_file: Path,
    output_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
    api_base_url: str = API_BASE_URL,
) -> dict[str, Any]:
    if api_base_url != API_BASE_URL or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", dataset_id):
        raise DownloadError("api_origin_rejected")
    if (
        type(expected_size_bytes) is not int
        or expected_size_bytes <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise DownloadError("archive_contract_invalid")
    api_key = read_secret(key_file)
    metadata = request_json(f"{API_BASE_URL}/datasets/{dataset_id}/download", api_key)
    download_url = metadata.get("downloadUrl")
    if not isinstance(download_url, str) or not download_url:
        raise DownloadError("api_metadata_invalid")
    size, actual_sha256 = stream_download(
        download_url,
        output_path,
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
    )
    return {
        "status": "downloaded_verified",
        "dataset_id": dataset_id,
        "output_path": str(output_path),
        "size_bytes": size,
        "sha256": actual_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size-bytes", required=True, type=int)
    args = parser.parse_args()
    try:
        result = download_dataset(
            dataset_id=args.dataset_id,
            key_file=args.key_file,
            output_path=args.output,
            expected_sha256=args.expected_sha256,
            expected_size_bytes=args.expected_size_bytes,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "exit_code": 1,
            "error_code": str(exc) if isinstance(exc, DownloadError) else "download_failed",
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
