# Bounded Docker CPU processor diagnostic

This is an auxiliary Linux ARM CPU diagnostic, not acceptance of the remote x86 CUDA runtime. Never create an engine, load backbone weights, change model classes or model info, bypass architecture checks, compile packages, change Docker daemon settings, or affect existing containers.

The approved limits are 30 minutes, 4 GiB newly allocated disk, at least 6 GiB Mac free space, and a dedicated `lite-omnijev-*` container with `--memory=3g --memory-swap=3g --cpus=2`. Only an owned source snapshot plus the existing 32 MB metadata and 5 MB licensed public fixture media may be mounted. No HOME, .env, Docker socket or weights mount is permitted. The helper refuses an existing container name and bounds per-stage execution to 600 seconds. The operator must retain the overall 30-minute deadline and account for both the image and owned files.

Official fixed artifacts:

- Python ARM64 image manifest: `sha256:1847ca1d1bc792b4b58fff39c82c67942112e3e1ebc0ee1d06b9014873152ce7`.
- vLLM `0.31.0+cpu`, `cp38-abi3-manylinux_2_39_aarch64`: 68,640,121 bytes, SHA256 `8a9f86447a8ebcd1dadeae81b4b52587d8d967f67500bc7c00d2c6383c1df9cb`. Its URL and constants are in `scripts/s0_docker_processor.py`.

The cached `python:3.11-slim` on this machine is **amd64**, despite an ARM Docker host. Its successful Python probe is not an ARM result. A native ARM image must be obtained and verified separately. The helper can fetch a pinned official registry manifest/config/layers, verify compressed layer SHA and uncompressed diff IDs, then load an owned image tag. Interrupted range responses retain their contiguous prefix in memory and continue from the next offset; each requested range start/end/total and the final complete SHA must match. Empty progress or TLS failures stop; never disable TLS verification.

Only after complete verified binaries are available may a dedicated installation container fetch official Torch 2.13 CPU ARM wheels, Transformers 5.17 and the source-METADATA dependency closure, with `--only-binary=:all:` and recorded wheel SHAs. Refuse CUDA stacks, source builds, missing key ARM wheels or disk-budget excess. No dependency installation occurred in the recorded attempt below, so no prepared image is claimed.

For a future successfully prepared owned image, the stage runner starts `--network=none`, clears `CUDA_VISIBLE_DEVICES`, enables both HF offline flags, sets one numerical thread and explicit `S0_CODE_VERSION`. First run `scripts.s0_prepare --profile canonical_http` against the licensed fixture inside the container. Then run `scripts.s0_vllm_processor` with that same-container HF report as `--hf-reference`, first six cases and only after success `--full`. Preserve any Mac/container input differences instead of calling FFmpeg-version differences native processor parity failures. The original Mac reference remains independently frozen; no numerical or source gate is relaxed.

## Actual result: 2026-10-14

The attempt stopped before dependency installation because public transport did not produce a verified complete ARM base or wheel. Docker pull exited 1 with `unexpected EOF`. The first wheel download ended at 5 MiB and failed the official size/SHA; bounded range recovery encountered incomplete bodies and TLS `record layer failure`. Raw official OCI recovery also encountered truncated bodies and TLS failure. The previous SSH ControlMaster was unavailable and the authorized SSH command returned authentication failure; no credentials were read or requested.

No processor canary, full-100 comparison, backbone, engine, CUDA operation or package installation ran. There are no owned running containers. Existing containers and Docker daemon settings were not changed. Partial owned artifacts are retained for evidence, not installed. This is `download_transport_blocked`, not proof of ARM CPU incompatibility. See `reports/s0-docker-processor-001.json`.

Official reference: [vLLM 0.31 CPU installation](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/cpu/), [fixed v0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0).
