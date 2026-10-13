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
```

## Serve-next audio compatibility

The existing remote `serve-next` environment was inspected read-only because it pins `torch==2.13.0+cu130` with `torchaudio==2.11.0+cu130`. CPU imports succeeded for torch, torchaudio, torchvision, Transformers, and vLLM, and a CPU tensor sum returned `2.0`. No existing GPU service environment was modified.

The isolated correction is to document the current lock as semver-skewed but CPU-import-valid; official PyPI metadata currently reports `torchaudio` latest as `2.11.0`, so no matching `torchaudio==2.13.0` wheel was installed into an existing env.

## Remote cache paths

- Python light wheelhouse: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/offline-tools-light`.
- Node archive: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/offline-tools/node/node-v24.21.0-linux-x64.tar.xz`.
- pnpm archive: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/offline-tools/frontend/pnpm-12.11.2.tgz`.
- Root training/dev/monitoring wheelhouse, owned by the coordinating root task: `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/root-training-dev-monitoring`.

## Pending checks

The light Python wheelhouse and the root training/dev/monitoring wheelhouse are background downloads on the constrained 2 GiB / 0.5 CPU remote. Hash manifests and no-index canaries must be generated after those processes exit. The report JSON records their PIDs and last observed status.
