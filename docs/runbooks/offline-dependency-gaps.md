# Offline Dependency Gap Audit

This audit covers the Python/runtime dependency closure needed before the next GPU run. It does not admit model weights, datasets, generated LoRA adapters, quantized weights, mobile packages, Android SDK/NDK terms, or frontend implementation work.

## Evidence inputs

- Local lockfiles: `uv.lock`, `infra/data/uv.lock`, `infra/serve/uv.lock`, and `infra/serve-next/uv.lock`.
- Remote project: `/root/autodl-tmp/lite-omnijev` on `root@region-42.seetacloud.com:48926`.
- Remote installed envs probed: `.venv`, `/root/autodl-tmp/envs/data`, `/root/autodl-tmp/envs/serve`, `/root/autodl-tmp/envs/serve-next`, `/root/autodl-tmp/envs/media-cpu`, `/root/autodl-tmp/envs/quant`, and `/root/autodl-tmp/envs/quant-offline-check`.
- Original plan text, parsed with `HTMLParser`, names Data-Juicer, datasketch, imagededup, ffmpeg, decord, PySceneDetect, torchaudio, lmms-eval, Playwright, and future frontend work, but the current repository only locks the Python envs listed above.

## Current must-have coverage

| Area | Required now | Current evidence | Wheelhouse state | Gap |
| --- | --- | --- | --- | --- |
| Training / gateway / dev / monitoring | Yes for P0/S0 and next GPU entrypoints | Remote `.venv` imports `torch==2.14.1`, `transformers==5.19.0`, `peft==0.21.2`, `accelerate==1.15.0`, `safetensors==0.8.0`, `fastapi==0.143.0`, `playwright==1.63.0`, and `swanlab==0.10.2`. | `artifacts/runtime-builds/python-wheels/root-training-dev-monitoring/requirements.txt` exported. Full wheel download started as remote PID `25391`; no wheel files had been committed to that directory at the checkpoint. | Installed env is usable; explicit offline rebuild wheelhouse is still in progress. Do not claim root wheelhouse complete until the job exits and hashes are written. |
| Data PyArrow reader | Yes for current admitted CLINC/data probes | `/root/autodl-tmp/envs/data` imports `pyarrow==26.0.0`. | Complete: `artifacts/runtime-builds/python-wheels/data-pyarrow`, 52 MiB, one wheel plus requirements/hash files. `pyarrow-26.0.0-cp311-cp311-manylinux_2_28_x86_64.whl` SHA-256 `6e89dee53aaeb50505ed6152ea55bc7ddfd4f4df264f5427ea255288d8f0e580`. | None for current data reader. |
| vLLM service baseline | Yes for service fallback/compat checks | `/root/autodl-tmp/envs/serve` imports `vllm==0.26.0`, `torch==2.11.0`, `transformers==5.13.0`, `torchaudio==2.11.0`, `torchvision==0.26.0`. | Requirements exported to `artifacts/runtime-builds/python-wheels/serve-vllm026/requirements.txt`; no full wheelhouse. | Existing env is installed, but offline reprovisioning is not wheelhouse-complete. Full vLLM closure is large; do not duplicate it without a scoped storage/time decision. |
| vLLM service candidate / teacher candidate | Yes for current preferred service candidate | `/root/autodl-tmp/envs/serve-next` imports `vllm==0.31.0`, `torch==2.13.0`, `transformers==5.17.0`, `torchaudio==2.11.0`, `torchvision==0.28.0`. | Requirements exported to `artifacts/runtime-builds/python-wheels/serve-next-vllm031/requirements.txt`; no dedicated full wheelhouse. The quant full wheelhouse includes overlapping `torch==2.13.0` and many CUDA 13 packages, but it is not a full vLLM wheelhouse. | Existing env is installed; offline reprovisioning remains a gap if the env is lost. |
| Quantization | Yes for later S6 tooling prep, but GPU quant run is future | `/root/autodl-tmp/envs/quant-offline-check` imported `llmcompressor==0.14.0`, `compressed-tensors==0.19.0`, `auto-round==0.15.1`, `transformers==5.17.0`, `torch==2.13.0`, `datasets==5.0.1`, `pandas==3.0.6`, and `pyarrow==26.0.0` without serve-next on `sys.path`. | Complete: `artifacts/runtime-builds/python-wheels/llmcompressor-0.14.0-full`, 2.7 GiB, 88 files including `quant-closure.requirements.txt` and hash manifest. | No new action in this audit; do not redo the quant wheelhouse. GPU quant validation and generated artifacts remain future work. |
| CPU media processor canary | Yes for S0 media template/processor checks | `/root/autodl-tmp/envs/media-cpu` imports `transformers==5.19.0`, `torch==2.14.1`, `torchvision==0.29.0`, `pillow==11.3.0`, and `soundfile==0.14.0`; remote `ffmpeg/ffprobe==4.4.2` passed prior synthetic media probe. | Only `/root/autodl-tmp/cache/media-wheels/torchvision-0.29.0...whl` is explicitly cached; the env is installed. | Current S0 media CPU probe is covered by installed env evidence, but there is no separate pinned media wheelhouse for `pillow/soundfile` and no locked `decord`/`PySceneDetect`. |

## Not currently pinned; do not download blindly

The original plan mentions the following stacks, but the current repo has no lockfile entries or implementation imports for them:

- Data-Juicer for recipe-driven multimodal cleaning.
- datasketch and imagededup for text/image/video-frame deduplication.
- decord and PySceneDetect for future video frame extraction and scene splitting.
- lmms-eval and torchmetrics for later general multimodal evaluation/regression.
- DeepSpeed/Unsloth for optional multi-GPU or small-card experiments.
- Frontend packages for a future Playground. There is no `package.json`, `pnpm-lock.yaml`, or frontend source tree in this worktree, and remote `node/npm/pnpm/yarn` are absent.

These are real future plan items, but not current locked must-haves. Caching them now would create an unreviewed dependency decision. Pin versions and licenses in a scoped ticket before downloading.

## Current remote handles

- Root wheelhouse download: `PID 25391`, log `/root/autodl-tmp/lite-omnijev/logs/root-wheelhouse-download.log`, destination `/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/python-wheels/root-training-dev-monitoring`.
- At the checkpoint, PID `25391` was still running and the destination contained only the exported requirements file. The log tail was downloading `mypy-1.20.2`; large torch/CUDA packages had not yet been materialized as wheel files in the destination.
- Data wheelhouse completed and was hashed.

## Safe next actions

1. Let PID `25391` finish or explicitly stop it if root decides the installed `.venv` plus uv cache is enough. After it exits, write `files.txt`, `sha256sums.txt`, and a concise completion report before calling that wheelhouse ready.
2. Keep using the installed `/root/autodl-tmp/envs/serve` and `/root/autodl-tmp/envs/serve-next` for the next GPU checks. Only build full vLLM wheelhouses if reprovisioning from scratch is required; they are large enough to need a separate budget/time decision.
3. Do not fetch Data-Juicer, lmms-eval, decord, PySceneDetect, datasketch, imagededup, DeepSpeed, Unsloth, or frontend packages until the project pins versions and licenses.
4. Treat future LoRA adapters, distilled E2B checkpoints, calibration tables, quantized GGUF/FP8/W4A16 files, mobile packages, and browser/frontend bundles as generated outputs, not prerequisite downloads.
