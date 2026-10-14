"""Bounded public-wheel fetch and isolated CPU processor Docker stage runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import http.client
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ARM_IMAGE = "python@sha256:1847ca1d1bc792b4b58fff39c82c67942112e3e1ebc0ee1d06b9014873152ce7"
WHEEL_URL = "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0%2Bcpu-cp38-abi3-manylinux_2_39_aarch64.whl"
WHEEL_NAME = "vllm-0.31.0+cpu-cp38-abi3-manylinux_2_39_aarch64.whl"
WHEEL_SIZE = 68640121
WHEEL_SHA256 = "8a9f86447a8ebcd1dadeae81b4b52587d8d967f67500bc7c00d2c6383c1df9cb"
DISK_LIMIT = 4 * 1024**3
FREE_RESERVE = 6 * 1024**3


def read_prefix(response: Any, limit: int | None) -> bytes:
    """Keep contiguous validated response bytes when a transfer ends early."""
    data = bytearray()
    while limit is None or len(data) < limit:
        width = 65536 if limit is None else min(65536, limit - len(data))
        try:
            block = response.read(width)
        except http.client.IncompleteRead as error:
            data.extend(error.partial)
            break
        except (TimeoutError, OSError):
            if not data:
                raise
            break
        if not block:
            break
        data.extend(block)
    return bytes(data)


def require_budget(root: Path, extra_bytes: int = 0) -> None:
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if used + extra_bytes > DISK_LIMIT or shutil.disk_usage(root).free < FREE_RESERVE:
        raise ValueError("diagnostic_disk_budget_exceeded")


def fetch_wheel(root: Path) -> dict[str, object]:
    """Range-verify each bounded response, then the official release asset SHA."""
    target = root / WHEEL_NAME
    chunks = root / "wheel-chunks"
    chunks.mkdir(exist_ok=True)
    size = 1024**2
    deadline = time.monotonic() + 600

    def fetch(start: int) -> Path:
        end = min(start + size, WHEEL_SIZE) - 1
        part = chunks / str(start)
        data = bytearray()
        for _ in range(3):
            offset = start + len(data)
            if offset > end:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("release_transfer_deadline")
            request = urllib.request.Request(
                WHEEL_URL + f"?diagnostic_bytes={offset}-{end}",
                headers={"Range": f"bytes={offset}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{WHEEL_SIZE}"
                ):
                    raise ValueError("invalid_release_asset_range")
                block = read_prefix(response, end - offset + 1)
                if not block:
                    raise ValueError("release_transfer_no_progress")
                data.extend(block)
        if len(data) != end - start + 1:
            raise ValueError("incomplete_release_asset_range")
        part.write_bytes(data)
        return part

    require_budget(root, WHEEL_SIZE * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        parts = list(executor.map(fetch, range(0, WHEEL_SIZE, size)))
    digest = hashlib.sha256()
    with target.open("wb") as output:
        for part in parts:
            data = part.read_bytes()
            output.write(data)
            digest.update(data)
    if target.stat().st_size != WHEEL_SIZE or digest.hexdigest() != WHEEL_SHA256:
        raise ValueError("release_asset_sha256_mismatch")
    return {"name": WHEEL_NAME, "bytes": WHEEL_SIZE, "sha256": WHEEL_SHA256, "url": WHEEL_URL}


def fetch_arm_image(root: Path) -> dict[str, object]:
    """Restore the pinned official image from digest-checked public registry blobs."""
    token_url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull"
    with urllib.request.urlopen(token_url, timeout=30) as response:
        token = json.load(response)["token"]
    deadline = time.monotonic() + 600

    def get(
        path: str, *, start: int | None = None, end: int | None = None, total: int | None = None
    ) -> bytes:
        if start is not None and end is None:
            raise ValueError("range_end_required")
        stop = end if end is not None else -1
        data = bytearray()
        for _ in range(3):
            offset = start + len(data) if start is not None else None
            if offset is not None and offset > stop:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("image_transfer_deadline")
            headers = {
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.oci.image.manifest.v1+json",
            }
            request_path = path
            if offset is not None:
                headers["Range"] = f"bytes={offset}-{end}"
                request_path += f"?diagnostic_bytes={offset}-{end}"
            with urllib.request.urlopen(
                urllib.request.Request(
                    "https://registry-1.docker.io/v2/library/python/" + request_path,
                    headers=headers,
                ),
                timeout=30,
            ) as response:
                if offset is not None and (
                    response.status != 206
                    or response.headers.get("Content-Range", "") != f"bytes {offset}-{end}/{total}"
                ):
                    raise ValueError("invalid_image_blob_range")
                block = read_prefix(response, stop - offset + 1 if offset is not None else None)
                if not block:
                    raise ValueError("image_transfer_no_progress")
                data.extend(block)
            if start is None:
                return bytes(data)
        if start is None:
            return bytes(data)
        if len(data) != stop - start + 1:
            raise ValueError("incomplete_image_blob_range")
        return bytes(data)

    manifest_data = get("manifests/" + ARM_IMAGE.split("@", 1)[1])
    if hashlib.sha256(manifest_data).hexdigest() != ARM_IMAGE.split(":", 1)[1]:
        raise ValueError("image_manifest_sha256_mismatch")
    manifest = json.loads(manifest_data)
    directory = root / "arm-image"
    directory.mkdir(exist_ok=True)
    config_data = get("blobs/" + manifest["config"]["digest"])
    config_sha = hashlib.sha256(config_data).hexdigest()
    if config_sha != manifest["config"]["digest"].split(":", 1)[1]:
        raise ValueError("image_config_sha256_mismatch")
    config = json.loads(config_data)
    if config["architecture"] != "arm64":
        raise ValueError("native_arm_image_required")
    config_name = config_sha + ".json"
    (directory / config_name).write_bytes(config_data)
    layer_names = []
    for index, layer in enumerate(manifest["layers"]):
        require_budget(root, layer["size"] * 5)
        width = 1024**2
        starts = list(range(0, layer["size"], width))
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            parts = list(
                pool.map(
                    lambda start, layer=layer, width=width: get(
                        "blobs/" + layer["digest"],
                        start=start,
                        end=min(start + width, layer["size"]) - 1,
                        total=layer["size"],
                    ),
                    starts,
                )
            )
        compressed = b"".join(parts)
        if hashlib.sha256(compressed).hexdigest() != layer["digest"].split(":", 1)[1]:
            raise ValueError("image_layer_sha256_mismatch")
        plain = gzip.decompress(compressed)
        require_budget(root, len(plain))
        if "sha256:" + hashlib.sha256(plain).hexdigest() != config["rootfs"]["diff_ids"][index]:
            raise ValueError("image_layer_diff_sha256_mismatch")
        name = f"layer-{index}.tar"
        (directory / name).write_bytes(plain)
        layer_names.append(name)
    tag = "lite-omnijev-python-arm-base-001:local"
    (directory / "manifest.json").write_text(
        json.dumps([{"Config": config_name, "RepoTags": [tag], "Layers": layer_names}])
    )
    archive = root / "python-arm-base-001.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name in [config_name, "manifest.json", *layer_names]:
            output.add(directory / name, arcname=name)
    subprocess.run(["docker", "load", "-i", str(archive)], check=True, timeout=120)
    return {
        "tag": tag,
        "source_manifest_sha256": ARM_IMAGE,
        "config_sha256": config_sha,
        "layers": manifest["layers"],
        "archive_bytes": archive.stat().st_size,
    }


def docker_run(root: Path, *, image: str, name: str, command: list[str], offline: bool) -> int:
    """Mount only the owned snapshot/output, with no GPU or ambient host secrets."""
    if not name.startswith("lite-omnijev-"):
        raise ValueError("diagnostic_container_name_required")
    if (
        subprocess.run(
            ["docker", "container", "inspect", name], capture_output=True, check=False
        ).returncode
        == 0
    ):
        raise ValueError("diagnostic_container_already_exists")
    image_size = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    require_budget(root, int(image_size.stdout.strip()))
    args = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/arm64",
        "--name",
        name,
        "--memory=3g",
        "--memory-swap=3g",
        "--cpus=2",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--network=none" if offline else "--network=bridge",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "MKL_NUM_THREADS=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "TOKENIZERS_PARALLELISM=false",
        "-e",
        "S0_CODE_VERSION=cf7ceb3",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-v",
        f"{root.resolve()}:/work",
        "-w",
        "/work",
    ]
    if offline:
        args += [
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            "-e",
            "VLLM_NO_USAGE_STATS=1",
        ]
    started = time.monotonic()
    try:
        with (root / f"{name}.stdout.txt").open("w") as output:
            result = subprocess.run(
                [*args, image, *command],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        code = result.returncode
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "stop", "-t", "2", name], check=False, capture_output=True)
        code = 124
    (root / f"{name}.exit.json").write_text(
        json.dumps(
            {
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - started,
                "offline": offline,
                "image": image,
                "memory_bytes": 3 * 1024**3,
                "cpus": 2,
            }
        )
        + "\n"
    )
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fetch-wheel", action="store_true")
    parser.add_argument("--fetch-image", action="store_true")
    parser.add_argument("--image", default=ARM_IMAGE)
    parser.add_argument("--name", default="lite-omnijev-arm-processor-001")
    parser.add_argument("--network", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.fetch_wheel:
        (args.root / "official-wheel.json").write_text(
            json.dumps(fetch_wheel(args.root), indent=2) + "\n"
        )
        return 0
    if args.fetch_image:
        (args.root / "official-arm-image.json").write_text(
            json.dumps(fetch_arm_image(args.root), indent=2) + "\n"
        )
        return 0
    if not args.command:
        parser.error("a stage command is required")
    return docker_run(
        args.root, image=args.image, name=args.name, command=args.command, offline=not args.network
    )


if __name__ == "__main__":
    raise SystemExit(main())
