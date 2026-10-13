# Runtime CPU Builds and Package Caches

This runbook records the `offline-prep:03` CPU-build slice for the S6 runtimes. It is pre-GPU evidence only: no model forward pass, generated quantized weight, GPU benchmark, mobile package, or production readiness claim is made here.

## Current state

The remote runtime source cache contains pinned source archives for `llama.cpp v0.6.0`, `Ollama v0.35.0`, `LiteRT-LM v0.9.0`, and `llm-compressor 0.14.0`. This slice now also has verified remote runtime/toolchain assets, a completed `llama.cpp` CPU build, and a standalone offline quant wheelhouse:

- Official `ollama-linux-amd64.tar.zst` for `v0.35.0` is staged at `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/packages/ollama-v0.35.0`, size `1,427,765,407`, SHA-256 `1c114a6b220c5efca2ef2b1e5f01d1e535e26f6cd6d1678c8489325d2835e525`. It was extracted and `extract/bin/ollama --version` reported client version `0.35.0` without installing a daemon.
- LiteRT host toolchain archives are staged at `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/toolchains`: Bazel `7.6.1`, CMake `4.4.4`, and Temurin Java/Javac `17.0.20.1` all have version evidence. Bazelisk is cached and hashed, but direct Bazel is the usable no-download proof path.
- `llama.cpp v0.6.0` CPU binaries are staged at `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/bin/llama.cpp-v0.6.0-cpu`. The resumed build ran under `flock /tmp/omnijev-cpu-heavy.lock`, exited `rc=0`, and `llama-cli --version` reported `version: 0.6.0 (build 24, commit 45ebe32)` built with GNU 11.4.0 for Linux x86_64.
- `/root/autodl-tmp/envs/quant` still imports `llmcompressor==0.14.0`, `compressed-tensors==0.19.0`, `auto-round==0.15.1`, `transformers==5.17.0`, and `torch==2.13.0` with CUDA metadata `13.0` on CPU; this env reuses serve-next through a `.pth` file and remains an import probe only.
- The standalone quant wheelhouse is now complete at `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/llmcompressor-0.14.0-full`: 85 wheels, 2.7 GiB, official PyPI resolution, SHA-256 manifest, and a separate offline install/import check in `/root/autodl-tmp/envs/quant-offline-check` with `sys_path_has_serve_next=false`.

## Runtime support check

Binary availability alone is not enough for this project. The pinned runtime source must also recognize the Gemma4/E2B and modality/conversion paths needed by the original S6 plan.

| Runtime | Pinned version | Source support evidence | Build/cache state |
| --- | --- | --- | --- |
| `llama.cpp` | `v0.6.0` | `common/parsers/gemma4.cpp`; `conversion/gemma.py` contains Gemma4/E2B/multimodal/GGUF paths; GGUF conversion scripts are present; quant names include `Q4_K_M` and `Q8_0`. | CPU build completed under the heavy lock; `llama-cli`, `llama-quantize`, `llama-gguf`, and `llama-gguf-split` are staged and hashed. Official `v0.6.0` release had no Linux x64 CPU prebuilt asset beyond `nightly-tag.txt`, so source build was required. |
| `Ollama` | `v0.35.0` | `create/gemma4.go`, Gemma4 parser/renderer files, `gemma4_e2b_chat_template.jinja2`, and MLX Gemma4 audio/vision/media code are present. | Official Linux amd64 binary cached, hashed, extracted, and version-checked. |
| `LiteRT-LM` | `v0.9.0` | README/API docs mention E2B, audio, vision, and multimodal; runtime embedding lookup and Python multimodal example paths are present. | Source cached. Bazel/JDK/CMake host toolchains are cached and version-checked. Ubuntu Noble GCC/G++ 13 packages are cached as `.deb` files only, not installed. Android NDK download still requires the Android SDK License acceptance and was not automated. |
| `llm-compressor` | `0.14.0` | Gemma4 FP8/NVFP4 docs and examples are present, including model-free PTQ, multimodal vision, and W4A16/NVFP4 examples. | Direct wheel cache complete; `/root/autodl-tmp/envs/quant` imports the stack on CPU by reusing serve-next torch/CUDA packages. |

## Cached package evidence

`/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/llmcompressor-0.14.0-direct` contains direct wheels for the first quant environment probe. The standalone closure is `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/llmcompressor-0.14.0-full`: 85 wheels, 2.7 GiB, `quant-closure.files.txt`, and `quant-closure.sha256sums.txt`. It uses the official PyPI `torch-2.13.0-cp311-cp311-manylinux_2_28_x86_64.whl` with SHA-256 `60fcdcb2f3876e21146cb4524ef06397d727ca9ad5f020818547e25075fe3cb7`.

The import probe reported:

```json
2026-10-14T03:57:11.4202 | get_main_device | WARNING - No accelerator available! Compressing model on CPU instead
{
  "packages": {
    "PIL": "12.0.0",
    "accelerate": "1.15.0",
    "auto_round": "0.15.1",
    "compressed_tensors": "0.19.0",
    "datasets": "5.0.1",
    "llmcompressor": "0.14.0",
    "multiprocess": "0.70.19",
    "numpy": "2.2.6",
    "pandas": "3.0.6",
    "pyarrow": "26.0.0",
    "torch": "2.13.0",
    "transformers": "5.17.0"
  },
  "python": "/root/autodl-tmp/envs/quant/bin/python",
  "sys_path_contains_serve_next": true,
  "torch_cuda": "13.0",
  "torch_cuda_available": false
}
```

## Blockers and next action

- GPU runtime and quantization validation remains pending; this slice only prepares CPU-build, official-binary, toolchain, wheelhouse, and import evidence.
- LiteRT host/mobile build is still not marked ready. Cached non-Android prerequisites now include Bazel `7.6.1`, CMake `4.4.4`, Temurin JDK `17.0.20.1`, and Ubuntu Noble GCC/G++ 13 `.deb` packages. Those GCC packages are cache-only and were not installed into the Ubuntu 22.04 host.
- Android NDK/SDK downloads were not automated because Google’s Android NDK download page requires accepting the Android SDK License before downloading. The concrete required human action is to accept the Android SDK License on the official NDK download page or through the Android command-line tools license flow before caching an NDK archive.
- The standalone quant wheelhouse passed an offline `--no-index --find-links` install/import check in `/root/autodl-tmp/envs/quant-offline-check`; no model forward pass or GPU quant run was attempted.
