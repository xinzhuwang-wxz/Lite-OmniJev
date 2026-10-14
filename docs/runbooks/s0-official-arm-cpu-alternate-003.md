# Official ARM CPU alternate-source recovery, 003

This slice recovered the complete vLLM CPU ARM wheel through a newly verified official endpoint. It stopped before installation because the pinned official Python base image did not finish downloading. There are zero HF/native processor cases and no full-100 result. This is not Linux x86, CUDA, GPU, probability, service, quality or release acceptance.

## Verified wheel

The [official vLLM 0.31 CPU documentation](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/cpu/) identifies `wheels.vllm.ai` as an official wheel host. The actual [0.31.0 CPU index](https://wheels.vllm.ai/0.31.0/cpu/vllm/) returned HTTP 200 and listed this [ARM wheel](https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm-0.31.0%2Bcpu-cp38-abi3-manylinux_2_39_aarch64.whl). No URL was inferred from a filename convention.

- File: `vllm-0.31.0+cpu-cp38-abi3-manylinux_2_39_aarch64.whl`
- Complete size: `68640121` bytes
- Complete SHA256: `8a9f86447a8ebcd1dadeae81b4b52587d8d967f67500bc7c00d2c6383c1df9cb`
- Retained local path: `artifacts/native-arm-cpu-official-alternate-003/` followed by that filename.
- Retained remote path: `/root/autodl-tmp/lite-omnijev/artifacts/native-arm-cpu-official-alternate-003/` followed by that filename.

The local curl attempt stopped after approximately 44 seconds at 121445 bytes under its low-speed bound. A single bounded remote new-source relay obtained all bytes and verified the expected release SHA. SSH then copied that already verified file locally, where size and SHA were independently checked again. The initial remote command failed with exit 127 because `python3` was absent; it made no remote source request. The corrected relay used the existing environment's Python. No credential was stored, and the earlier failing GitHub/DockerHub paths were not retried.

## Base-image result

The [official uv Docker documentation](https://docs.astral.sh/uv/guides/integration/docker/) explicitly lists `ghcr.io/astral-sh/uv:python3.11-trixie-slim`. Its real GHCR index was obtained and its bytes matched the registry digest:

- Index: `sha256:ddda768d6445e90828a663cf7bb7896a8da2ea4ca679abb2b5a3450e8aa32758`
- Linux ARM64 manifest: `sha256:29ebfd64068a5ed091591693a8e9f28a49ff6bc4411611b3c0000ce6d5281e91`
- Config: `sha256:92aca8d0f4758e9930ccb84d880f1766a50682d258597cf978993219ed11aa18`
- Five compressed layers: `64771792` bytes total.

Docker manifest inspection succeeded for that exact digest. A 180-second bounded `docker pull --platform linux/arm64` exited 124 after layer transfer retries. The exact image was absent in the final image inspection. Consequently Python 3.11 and glibc >=2.39 were **not runtime verified**. An official tag/manifest is not a prepared runtime.

No container, package installation, source compilation, model weights or engine was started. Docker daemon settings and other containers were untouched. No cleanup/prune of shared layer content was attempted. The bounded branch stopped here without another image source or relay download attempt.

## Remaining acceptance

Retain the approved limits for any separately authorized continuation: 30 minutes total, at most 4 GiB new disk, Mac free space >=6 GiB, and an owned `lite-omnijev-*` container capped at 3 GiB RAM, 3 GiB RAM+swap and 2 CPUs. This slice ended within 14 minutes with roughly 10 GiB Mac free space; Docker's loaded image count and size remained unchanged. The resource snapshots do not attribute unrelated concurrent host allocation.

A later runnable base must prove Python **3.11** and glibc >=2.39 before binary-only CPU dependency preparation. Pin Torch `2.13.0+cpu` and Transformers `5.17`, record official dependency hashes, and refuse CUDA packages, source builds, weights or engine initialization. No such closure was downloaded in this slice.

Use the actual same-container Transformers 5.17 HF canonical six-case reference, retaining every tensor field and shape. Compare native six cases before a full-100 run. Do not squeeze batch dimensions, synthesize frame counts, or relax exact-field comparisons to make a result pass. Transformers 5.13/5.19 evidence from another environment cannot substitute for this reference.

The complete machine-readable evidence and artifact hashes are in `reports/s0-official-arm-cpu-alternate-003.json`. The wheel is a successful prerequisite recovery; processor acceptance remains blocked.
