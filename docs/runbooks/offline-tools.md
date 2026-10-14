# Offline tools preparation

This runbook records the isolated tooling locked for Lite-OmniJev S0-S6 offline preparation. It does not change the project runtime lock, service environments, ticket state, or product code.

## Python tool lock

`infra/offline-tools/pyproject.toml` is a standalone uv project for auxiliary OSS tools that may be needed before the next GPU run:

- `py-data-juicer==1.6.0` for Data-Juicer recipes when corpus filtering needs them.
- `datasketch==2.0.0` for MinHash/LSH duplicate and contamination checks.
- `imagededup==0.3.3.post2` for image duplicate candidate generation.
- `scenedetect==0.7.2` for deterministic scene ranges in video media QA.
- `lmms-eval==0.7.3` for optional multimodal evaluation harness work.
- `torchmetrics==1.9.0` for metric cross-checks in evaluation reports.

The lock is `infra/offline-tools/uv.lock`. The exported full requirements file is `reports/offline-tools.requirements.txt`. The light remote cache file is `reports/offline-tools.light-requirements.txt`, which removes torch, torchvision, triton, and `nvidia-*`/`cuda-*` wheels so this slice reuses the existing torch/quant/root caches instead of duplicating multi-GB CUDA payloads.

`decord` is not selected for this stage because the current media validation path already uses ffmpeg/ffprobe and the official Transformers processor path. DeepSpeed and Unsloth are documented as optional only; neither is required for the frozen Gemma/PEFT/vLLM baseline before the next GPU run.

## Frontend seed lock

There is no product frontend app in this slice. `infra/offline-tools/frontend/package.json` and `pnpm-lock.yaml` only pin a future Playground dependency seed: Node `v24.21.0`, pnpm `12.11.2`, React `19.3.0`, Vite `8.3.4`, shadcn `4.21.4`, Tailwind `4.3.3`, and related UI utilities. The remote cache contains the official Linux x64 Node archive and the pnpm tarball; Node was extracted only under `artifacts/runtime-builds/offline-tools/node` for version probing.

Remote evidence:

```text
node --version -> v24.21.0
npm --version -> 11.19.0
corepack --version -> 0.36.0
pnpm-12.11.2.tgz sha256 -> eb57341b8b54fddaeededb15dbb77cc30a11750cecca301bfda02478fdff3a00
pnpm fetch -> 322 packages, supply-chain policy passed
pnpm offline install -> 322 reused, 0 downloaded, React 19.3.0 / framer-motion 14.1.0 / lucide-react 1.55.0 / Vite 8.3.4
```

The Vite probe is deliberately a build-tool canary under the isolated artifact directory, not a product UI implementation. Local evidence is `reports/frontend-vite-offline-probe.json`.

The bounded UI source gap is now covered for the selected Playground foundation controls. Official shadcn/ui `new-york-v4` registry JSON is cached for `button`, `card`, `input`, `textarea`, `select`, `tabs`, `alert`, and `dialog`; official Animate UI registry JSON is cached for `primitives-animate-tabs` plus its three registry dependencies (`primitives-effects-highlight`, `primitives-animate-slot`, and `lib-get-strict-context`). The raw registry source cache is 58,139 bytes, below the 100 MB budget, with per-item source URLs, MIT license notes, JSON SHA-256, inline file SHA-256, and dependency lists in `reports/frontend-registry-sources.json`. The selected registry npm additions are only `radix-ui==1.8.1` and `motion==14.2.0`; `cn` is treated as the local utility alias, not an npm package.

A synthetic Vite app was generated under `artifacts/runtime-builds/offline-tools/frontend/registry-build-canary` using only the cached registry source files and local utility glue. `npm exec --offline --yes pnpm@12.11.2 -- install --offline --frozen-lockfile --ignore-scripts` reused the pnpm store with `downloaded 0`, then `vite build` passed with 2 dist files. Local evidence is `reports/frontend-registry-offline-build.json`. This is still a source/build canary only: no product UI, browser visual QA, full registry crawl, account, or paid API is claimed.

## Serve-next audio compatibility

The existing remote `serve-next` environment was inspected read-only because it pins `torch==2.13.0+cu130` with `torchaudio==2.11.0+cu130`. CPU imports succeeded for torch, torchaudio, torchvision, Transformers, and vLLM, and a CPU tensor sum returned `2.0`. No existing GPU service environment was modified.

The isolated correction is to document the current lock as semver-skewed but CPU-import-valid; official PyPI metadata currently reports `torchaudio` latest as `2.11.0`, so no matching `torchaudio==2.13.0` wheel was installed into an existing env.

## Remote cache paths

- Python light wheelhouse: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/offline-tools-light`.
- Node archive: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/offline-tools/node/node-v24.21.0-linux-x64.tar.xz`.
- pnpm archive: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/offline-tools/frontend/pnpm-12.11.2.tgz`.
- Root training/dev/monitoring wheelhouse, owned by the coordinating root task: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/root-training-dev-monitoring`.

## Offline install canaries

The light Python wheelhouse has a full no-index canary in `/root/autodl-tmp/envs/offline-tools-canary-full`. The canary installed from `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/offline-tools-light` and reused the existing `serve-next` torch path for CPU imports instead of copying another Torch/CUDA stack. The recorded imports are Data-Juicer, datasketch, imagededup, lmms-eval, PySceneDetect, and torchmetrics; the CPU torch probe returned `2.0` with CUDA unavailable. Local evidence is `reports/offline-tools-canary-full.json`.

## Android NDK and LiteRT-LM Android cache

The user approved the Android SDK License path through the root session for NDK r28b `28.1.13356709`. The official archive `android-ndk-r28b-linux.zip` is cached under `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/android-tools`, and its SHA1 matches the official value `f574d3165405bd59ffc5edaadac02689075a729f`. The extracted NDK is 2.2G. A tiny `hello.c` aarch64 API24 compile probe proves the NDK toolchain is usable and produced an AArch64 ELF, but it is not a LiteRT-LM runtime claim. Local evidence is `reports/android-tools-preparation.json`.

LiteRT-LM v0.9.0 has an official Android arm64 CLI release asset, `litert_lm_main.android_arm64`. It is cached under `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/bin/litert-lm-v0.9.0-android_arm64`, with SHA256 `606a933f3971b7c3c23a00e016cbf020ccd6d35f04a1ee8d1072387f361bb629`; `file` and `llvm-readelf` identify it as an AArch64 Android PIE executable. The v0.9.0 release metadata has platform CLI binaries and no Android AAR asset. Kotlin/JNI packaging targets exist in source (`//kotlin/java/com/google/ai/edge/litertlm:litertlm-android` and `//kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni`), so an AAR/API artifact still requires Bazel/Android SDK dependency fetch and build work if that packaging path becomes required. Local evidence is `reports/litert-lm-android-runtime-cache.json`.

## Pending checks

The frontend pnpm store is cached and has an offline install canary. The light Python wheelhouse is cached, hashed, and import-checked. The root training/dev/monitoring wheelhouse is still downloading on the constrained 2 GiB / 0.5 CPU remote; PID 25391 and supplemental shard PID 29601 were last observed on the same `torch==2.14.1` transfer. Hash manifests and root no-index canaries must be generated after those processes exit. Playwright Chromium browser binaries and current serve-next vLLM cache closure still need final proof; Python package imports alone are not counted as browser/runtime closure.
