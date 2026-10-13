# Runtime CPU Builds and Package Caches

This runbook records the `offline-prep:03` CPU-build slice for the S6 runtimes. It is pre-GPU evidence only: no model forward pass, generated quantized weight, GPU benchmark, mobile package, or production readiness claim is made here.

## Current state

The remote runtime source cache contains pinned source archives for `llama.cpp v0.6.0`, `Ollama v0.35.0`, `LiteRT-LM v0.9.0`, and `llm-compressor 0.14.0`. This slice now also has verified remote runtime/toolchain assets and a CPU import-checked quant environment:

- Official `ollama-linux-amd64.tar.zst` for `v0.35.0` is staged at `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/packages/ollama-v0.35.0`, size `1,427,765,407`, SHA-256 `1c114a6b220c5efca2ef2b1e5f01d1e535e26f6cd6d1678c8489325d2835e525`. It was extracted and `extract/bin/ollama --version` reported client version `0.35.0` without installing a daemon.
- LiteRT host toolchain archives are staged at `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/toolchains`: Bazel `7.6.1`, CMake `4.4.4`, and Temurin Java/Javac `17.0.20.1` all have version evidence. Bazelisk is cached and hashed, but direct Bazel is the usable no-download proof path.
- `/root/autodl-tmp/envs/quant` imports `llmcompressor==0.14.0`, `compressed-tensors==0.19.0`, `auto-round==0.15.1`, `transformers==5.17.0`, and `torch==2.13.0` with CUDA metadata `13.0` on CPU. This env reuses `/root/autodl-tmp/envs/serve-next/lib/python3.11/site-packages` through a `.pth` file; `serve-next` itself was not modified.
- The standalone quant wheelhouse closure is still not complete because the `torch==2.13.0` wheel download was too slow/stalled on both remote pip and local mirror attempts. The prepared quant env is therefore an import-checked isolated env that reuses existing serve-next torch/CUDA packages, not a self-contained wheelhouse.

The `llama.cpp` CPU source build is still paused, not failed, because the remote cgroup is capped at 2 GiB and 0.5 CPU and root asked this agent to stop only its compiler subtree while higher-priority checks run. Paused remote PIDs: `8791`, `8962`, `8963`, `8965`, `8968`, `12081`, `14088`, `14089`, `14090`. The preserved build directory is `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/build/llama.cpp-v0.6.0-cpu-lowmem`. Resume only under `flock /tmp/omnijev-cpu-heavy.lock`.

## Runtime support check

Binary availability alone is not enough for this project. The pinned runtime source must also recognize the Gemma4/E2B and modality/conversion paths needed by the original S6 plan.

| Runtime | Pinned version | Source support evidence | Build/cache state |
| --- | --- | --- | --- |
| `llama.cpp` | `v0.6.0` | `common/parsers/gemma4.cpp`; `conversion/gemma.py` contains Gemma4/E2B/multimodal/GGUF paths; GGUF conversion scripts are present; quant names include `Q4_K_M` and `Q8_0`. | CPU build configured with CUDA/BLAS/native off and tests off; paused before binary staging. Official `v0.6.0` release had no Linux x64 CPU prebuilt asset beyond `nightly-tag.txt`, so no compatible prebuilt was available. |
| `Ollama` | `v0.35.0` | `create/gemma4.go`, Gemma4 parser/renderer files, `gemma4_e2b_chat_template.jinja2`, and MLX Gemma4 audio/vision/media code are present. | Official Linux amd64 binary cached, hashed, extracted, and version-checked. |
| `LiteRT-LM` | `v0.9.0` | README/API docs mention E2B, audio, vision, and multimodal; runtime embedding lookup and Python multimodal example paths are present. | Source cached. Bazel/JDK/CMake host toolchains are cached and version-checked, but remote GCC/G++ remains 11.4 while upstream CMake guide expects GCC/G++ 13+ on Ubuntu 24.04. Android SDK terms were not accepted automatically. |
| `llm-compressor` | `0.14.0` | Gemma4 FP8/NVFP4 docs and examples are present, including model-free PTQ, multimodal vision, and W4A16/NVFP4 examples. | Direct wheel cache complete; `/root/autodl-tmp/envs/quant` imports the stack on CPU by reusing serve-next torch/CUDA packages. |

## Cached package evidence

`/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/llmcompressor-0.14.0-direct` contains direct wheels for `llmcompressor==0.14.0`, `transformers==5.17.0`, `datasets==5.0.1`, `auto-round==0.15.1`, `accelerate==1.15.0`, `compressed-tensors==0.19.0`, and supporting direct packages. The quant env added missing transitive packages into `/root/autodl-tmp/envs/quant` through pip without modifying `serve-next`; evidence files are `quant-env-install.log`, `quant-env-deps-repair.log`, `quant-env-imports.json`, `quant-env-freeze.txt`, and `quant-env-evidence.json`.

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

- Resume the paused `llama.cpp` build only after root clears CPU contention. Use `flock /tmp/omnijev-cpu-heavy.lock` and resume the recorded PIDs with `SIGCONT`; do not delete the build directory or restart from scratch.
- The quant env is ready for CPU import/preflight, but a fully standalone torch wheelhouse remains a documented gap. Existing `serve-next` torch/CUDA packages are reused through a `.pth` link; GPU quant validation remains pending.
- LiteRT host/mobile build is still not marked ready because the remote compiler is below upstream guidance and Android SDK terms must not be automated. Cached non-Android prerequisites are available: Bazel `7.6.1`, CMake `4.4.4`, and Temurin JDK `17.0.20.1`.
- GPU runtime and quantization validation remains pending; this slice only prepares CPU-build, official-binary, toolchain, and import evidence.
