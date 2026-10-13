"""Request and download an authorized Mozilla Data Collective dataset archive.

This helper assumes the human has already signed in to MDC, accepted the dataset
terms in the web UI, and stored an API key in a local 0600 file. It never prints
the key and does not attempt to accept terms through the API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

API_BASE_URL = "https://mozilladatacollective.com/api"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_secret(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"refusing to read {path}; expected mode 0600 or stricter")
    secret = path.read_text().strip()
    if not secret:
        raise ValueError(f"empty API key file: {path}")
    return secret


def request_json(url: str, api_key: str) -> dict[str, Any]:
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
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MDC API returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MDC API returned a non-object JSON payload")
    return cast(dict[str, Any], payload)


def stream_download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp_path, output_path)


def download_dataset(
    *,
    dataset_id: str,
    key_file: Path,
    output_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
    api_base_url: str = API_BASE_URL,
) -> dict[str, Any]:
    api_key = read_secret(key_file)
    download_metadata = request_json(f"{api_base_url}/datasets/{dataset_id}/download", api_key)
    download_url = download_metadata.get("downloadUrl")
    if not isinstance(download_url, str) or not download_url:
        raise ValueError("MDC API response did not include a downloadUrl")
    stream_download(download_url, output_path)
    actual_size = output_path.stat().st_size
    if actual_size != expected_size_bytes:
        raise ValueError(f"size mismatch: expected {expected_size_bytes}, got {actual_size}")
    actual_sha256 = sha256(output_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"sha256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    return {
        "status": "downloaded_verified",
        "dataset_id": dataset_id,
        "output_path": str(output_path),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "filename": download_metadata.get("filename"),
        "expires_at": download_metadata.get("expiresAt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size-bytes", required=True, type=int)
    args = parser.parse_args()
    result = download_dataset(
        dataset_id=args.dataset_id,
        key_file=args.key_file,
        output_path=args.output,
        expected_sha256=args.expected_sha256,
        expected_size_bytes=args.expected_size_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
