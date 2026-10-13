# Runtime CPU Builds and Package Caches

This runbook records the `offline-prep:03` CPU-build slice for the S6 runtimes. It is pre-GPU evidence only: no model forward pass, generated quantized weight, GPU benchmark, mobile package, or production readiness claim is made here.

## Current state

The remote runtime source cache already contains pinned source archives for `llama.cpp v0.6.0`, `Ollama v0.35.0`, `LiteRT-LM v0.9.0`, and `llm-compressor 0.14.0`. This slice added a direct wheel cache for llm-compressor and started a CPU-only llama.cpp build. The llama.cpp build is paused, not failed, because the remote cgroup is capped at 2 GiB and 0.5 CPU and root asked this agent to stop only its compiler subtree while short checks run.

Paused remote PIDs: `8791`, `8962`, `8963`, `8965`, `8968`, `12081`, `14088`, `14089`, `14090`. The preserved build directory is `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/build/llama.cpp-v0.6.0-cpu-lowmem`. Resume only under `flock /tmp/omnijev-cpu-heavy.lock`.

## Runtime support check

Binary availability alone is not enough for this project. The pinned runtime source must also recognize the Gemma4/E2B and modality/conversion paths needed by the original S6 plan.

| Runtime | Pinned version | Source support evidence | Build/cache state |
| --- | --- | --- | --- |
| `llama.cpp` | `v0.6.0` | `common/parsers/gemma4.cpp`; `conversion/gemma.py` contains Gemma4/E2B/multimodal/GGUF paths; GGUF conversion scripts are present; quant names include `Q4_K_M` and `Q8_0`. | CPU build configured with CUDA/BLAS/native off and tests off; paused before binary staging. |
| `Ollama` | `v0.35.0` | `create/gemma4.go`, Gemma4 parser/renderer files, `gemma4_e2b_chat_template.jinja2`, and MLX Gemma4 audio/vision/media code are present. | Source cached. Official Linux amd64 asset is known with SHA-256 but remote download failed before receiving bytes. Source build is blocked by missing Go 1.26 on the remote. |
| `LiteRT-LM` | `v0.9.0` | README/API docs mention E2B, audio, vision, and multimodal; runtime embedding lookup and Python multimodal example paths are present. | Source cached. Host build blocked by upstream toolchain requirements: CMake 3.25+, GCC/G++ 13+, Bazel 7.6.1/Java for Bazel path. Android/JDK/SDK terms were not accepted automatically. |
| `llm-compressor` | `0.14.0` | Gemma4 FP8/NVFP4 docs and examples are present, including model-free PTQ, multimodal vision, and W4A16/NVFP4 examples. | Direct wheel cache complete for 13 direct artifacts, 41,166,938 bytes, with hashes in the JSON ledger. |

## Cached package evidence

`/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/llmcompressor-0.14.0-direct` contains direct wheels for `llmcompressor==0.14.0`, `transformers==5.17.0`, `datasets==5.0.1`, `auto-round==0.15.1`, `accelerate==1.15.0`, `compressed-tensors==0.19.0`, and supporting direct packages. These were downloaded without installing into or replacing any existing environment. Torch is intentionally not duplicated because the release requires `torch<=2.14.0`; `serve-next` already has `torch==2.13.0+cu130`, while the root training env has `torch==2.14.1` and must not be used as-is for an llm-compressor 0.14 release env.

## Blockers and next action

- Finish the paused `llama.cpp` build after root’s short checks. Use `flock /tmp/omnijev-cpu-heavy.lock` and resume the recorded PIDs with `SIGCONT`; do not delete the build directory or restart from scratch.
- Retry caching `ollama-linux-amd64.tar.zst` when the remote GitHub path is stable, then verify SHA-256 `1c114a6b220c5efca2ef2b1e5f01d1e535e26f6cd6d1678c8489325d2835e525` and run `ollama --version` from the extracted binary without installing a service.
- Do not automate Android SDK/JDK terms. The non-gated alternatives currently available are source-level LiteRT-LM evidence, local iOS/Xcode inspection, and llama.cpp/Ollama GGUF paths until the user explicitly accepts the Android toolchain terms.
- GPU runtime and quantization validation remains pending; this slice only prepares CPU-build and package-cache evidence.
