# Offline Runtime Preparation

This runbook is the `offline-prep:03` evidence trail for runtime and build dependencies. It does not admit models, datasets, teacher weights, generated LoRA adapters, calibration artifacts, quantized weights, mobile packages, or GPU results. The machine-readable ledger is `configs/project-runtime-assets.json`.

## Scope

The original S6 plan requires:

- E4B server artifacts in FP8, W4A16, GGUF Q8_0, and GGUF Q4_K_M.
- E2B distilled and LoRA-merged edge artifacts, including GGUF Q4_K_M and a fallback to Q5_K_M or Q8_0 if Q4 loses too much quality.
- vLLM, Ollama, and llama.cpp one-command delivery checks.
- llama.cpp and LiteRT-LM checks for option logits and media support.
- API latency/error-rate checks and mobile memory checks.

Those are future S6 validation items. The current offline-prep task only records whether the software needed to rebuild or probe those paths is locked, installed, cached, or missing before the next GPU run.

## Current Evidence

Checked lockfiles:

| Path | SHA256 | Status |
| --- | --- | --- |
| `uv.lock` | `65eeed6e0b0db033dcc6f25bebbbd7c392fac629fc904cc16bcd9fd6b0781c60` | Installed in the remote root `.venv` for training/gateway packages |
| `infra/data/uv.lock` | `81ce12213cfc69b30d210faed6267275d860f371ea349543e8c9ff404929aa4b` | Installed in `/root/autodl-tmp/envs/data` as `pyarrow==26.0.0` |
| `infra/serve/uv.lock` | `673de0eeb90c4117424868de331874d606106192abc844d07e1d4f2aa969ef39` | Installed in `/root/autodl-tmp/envs/serve` as `vllm==0.26.0`, `torch==2.11.0`, `transformers==5.13.0` |
| `infra/serve-next/uv.lock` | `df2242e7620d0c85202c0bcffcc31d8e3a077280fc0579ec472c36559bac02ab` | Installed in `/root/autodl-tmp/envs/serve-next` as `vllm==0.31.0`, `torch==2.13.0`, `transformers==5.17.0` |

Remote no-GPU host evidence:

- Host: `root@region-42.seetacloud.com:48926`, project `/root/autodl-tmp/lite-omnijev`.
- Disk: `/root/autodl-tmp` is an 800G filesystem with 725G available after staging runtime sources.
- Root `.venv` imports package metadata for `torch==2.14.1`, `transformers==5.19.0`, `peft==0.21.2`, `accelerate==1.15.0`, `safetensors==0.8.0`, `fastapi==0.143.0`, `uvicorn==0.54.0`, `httpx==0.28.1`, `pydantic==2.14.1`, `huggingface-hub==1.33.0`, `tokenizers==0.23.3`, `playwright==1.63.0`, and `swanlab==0.10.2`.
- After sourcing `infra/bootstrap/autodl-env.sh`, `uv` resolves to `/root/miniconda3/bin/uv`. The service envs include `ninja==1.13.2`.
- `ffmpeg==4.4.2-0ubuntu0.22.04.1` and `ffprobe==4.4.2-0ubuntu0.22.04.1` are installed from the configured Ubuntu jammy repositories. A remote one-second synthetic H.264/AAC file was generated and probed successfully, including a 16 kHz audio stream.
- `cmake` and `git` are present. Java, Android SDK, and remote Ollama binary are still absent.
- Root model-assets coordination reports `Valen-Team/Valen-Preview-0923` `checkpoint.pt` is being downloaded into quarantine with frozen SHA prefix `30096aaa`; that asset needs audited tensor-only conversion and nested checkpoint mapping before runtime use. This ticket did not load or execute the raw `.pt` file.

Local macOS evidence:

- `ffmpeg==8.0.1` and `ffprobe==8.0.1` are installed locally. A one-second synthetic H.264/AAC file was generated and probed successfully, including a 16 kHz audio stream.
- Playwright `1.63.0` is installed locally, with cached Chromium and Playwright FFmpeg artifacts.
- Xcode `26.5`, iPhoneOS SDK `26.5`, and CMake `4.1.1` are installed locally.
- Java/Android SDK were not found locally. Ollama client `0.24.0` is installed locally, but no daemon was running.

Official upstream source checks on 2026-10-14:

- `llm-compressor`: GitHub releases show `v0.14.0` as latest, commit `6e4b96e`. Source archive cached remotely with SHA-256 `232c24db6a488b3bbeaed9dd8935941c4063c81e6ebc7cb8cba856e4f60cbefb`. The archive contains `docs/key-models/gemma4` and Gemma4 FP8/NVFP4/W4A16-style examples.
- `llama.cpp`: GitHub releases show `b11583`, commit `9880157`, as the latest release entry; `git ls-remote --tags` also observed semantic tag `v0.6.0` at `69ecae922d2b60c1fa8c590279f4e2099af75111`. The `v0.6.0` source archive is cached remotely with SHA-256 `09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12`.
- `Ollama`: GitHub releases show `v0.35.0`, commit `cc40693`, as latest; the local client is older at `0.24.0`. The `v0.35.0` source archive is cached remotely with SHA-256 `2d24c59fe4aec09fa68994816cd24e233dff4d57cbc31ee9b46973fb0b3f55e7`.
- `LiteRT-LM`: the official repository says source builds should check out a stable tag. `git ls-remote --tags` observed stable tag `v0.9.0` at `e515431480b60a92a9a841d613a9c5404d059675` plus `v0.9.0-alpha*`, `beta`, and `rc` tags. The `v0.9.0` source archive is cached remotely with SHA-256 `bda82d61677965daa3b839cb70073f1ce9b0e151057ede54031df841a5d6c97f`.

Remote runtime source cache:

| Runtime | Cached Path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `llm-compressor 0.14.0` | `artifacts/runtime-sources/llm-compressor-0.14.0.tar.gz` | 3,044,870 | `232c24db6a488b3bbeaed9dd8935941c4063c81e6ebc7cb8cba856e4f60cbefb` |
| `llama.cpp v0.6.0` | `artifacts/runtime-sources/llama.cpp-v0.6.0.tar.gz` | 37,874,349 | `09b36dba235fcac180efff18514da7038e65d8e5b9e4aefa59238468abfdec12` |
| `Ollama v0.35.0` | `artifacts/runtime-sources/ollama-v0.35.0.tar.gz` | 26,420,689 | `2d24c59fe4aec09fa68994816cd24e233dff4d57cbc31ee9b46973fb0b3f55e7` |
| `LiteRT-LM v0.9.0` | `artifacts/runtime-sources/litert-lm-v0.9.0.tar.gz` | 339,861,017 | `bda82d61677965daa3b839cb70073f1ce9b0e151057ede54031df841a5d6c97f` |

## Readiness Matrix

| Area | Status | Evidence | Next Action |
| --- | --- | --- | --- |
| Training runtime | Installed on remote root `.venv` | Root lock and import metadata | GPU CUDA check remains pending |
| Teacher candidate runtime | Installed in isolated remote env | `/root/autodl-tmp/envs/serve-next`, `vllm==0.31.0`, `transformers==5.17.0` | GPU/model execution remains pending |
| vLLM service runtime | Installed in isolated remote envs | `/root/autodl-tmp/envs/serve`, `/root/autodl-tmp/envs/serve-next` | Run `collect_env` and startup only on an enabled GPU host |
| FastAPI gateway | Installed in remote root `.venv` | Import metadata | Service smoke still depends on model backend |
| Browser runtime | Cached locally, remote unchecked | Playwright CLI/cache | Coordinate remote browser cache only if needed |
| Media tools | Installed locally and remotely | Local and remote FFmpeg probes passed | Keep package versions in run evidence |
| CPU media processor probe | Passed | `reports/cpu-media-probe-offline-runtime.json`; Transformers 5.19 `Gemma4Processor` output tensors for image/audio/video | GPU/model parity still pending |
| Data runtime | Installed in isolated remote env | `/root/autodl-tmp/envs/data`, `pyarrow==26.0.0` | None for CPU data runtime |
| Valen runtime conversion | Asset quarantined, conversion missing | Root model-assets coordination; SHA prefix `30096aaa` | Audit tensor-only conversion and nested checkpoint mapping; never load raw `.pt` in normal inference |
| FP8/W4A16 quantization | Source cached, independent env pending | `llm-compressor 0.14.0` source archive and Gemma4 examples present | Build/install only in an independent quantization env |
| llama.cpp/GGUF | Source cached, build pending | `llama.cpp v0.6.0` source archive cached | Build per platform after model artifacts exist |
| Ollama | Source cached, local client only | `Ollama v0.35.0` source archive cached; local client `0.24.0`; remote binary absent | Pin/build binary or installer before S6 delivery |
| LiteRT-LM | Source cached, build pending | `LiteRT-LM v0.9.0` source archive cached | Resolve platform terms/toolchains and verify logits/audio support |
| iOS build prerequisites | Local installed | Xcode/iPhoneOS SDK probes | Mobile runtime source still missing |
| Android build prerequisites | Missing | Java and SDK absent | Install JDK/Android SDK only after explicit dependency decision |

## CPU Media Processor Probe

`scripts/cpu_media_probe.py` is the no-GPU S0 media preparation check. It creates development-only canaries: a PIL image, a screenshot-like PIL image, a 16 kHz sine WAV, and a 3-frame MP4 generated by FFmpeg. The script extracts the MP4 frames with FFmpeg, renders the official processor chat template with `enable_thinking=False`, and calls the real Transformers 5.19 `Gemma4Processor` with `images=`, `audio=`, `videos=`, `num_frames=3`, `return_tensors="pt"`, and `return_mm_token_type_ids=True`.

The current evidence report is `reports/cpu-media-probe-offline-runtime.json`, SHA-256 `76e697e375e8717e32f208a2773f8a48b6097cf89230fe390b77baf087d29701`. It passed on `/root/autodl-tmp/envs/media-cpu` with `transformers==5.19.0`, `torch==2.14.1`, `torchvision==0.29.0`, `pillow==11.3.0`, and `soundfile==0.14.0`. The report records model artifact hashes, media hashes, template placeholder counts, tensor dtypes/shapes/hashes, and answer-token boundary bindings for Choice A-T+U, Score 0-9+U, and Noul yes/no+U.

This is only a CPU processor/tokenizer canary. It does not load model weights as a model, produce logits, use GPU, compare HF/vLLM probabilities, or pass S0 parity.

## Reproduce The Probes

Run these from the repository root. They are read-only except for temporary files created by the media probe.

```sh
shasum -a 256 uv.lock infra/data/uv.lock infra/serve/uv.lock infra/serve-next/uv.lock
```

```sh
uv run python - <<'PY'
import tomllib
from pathlib import Path

for path in ["uv.lock", "infra/data/uv.lock", "infra/serve/uv.lock", "infra/serve-next/uv.lock"]:
    data = tomllib.loads(Path(path).read_text())
    packages = {package["name"].lower(): package.get("version") for package in data.get("package", [])}
    wanted = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "vllm",
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "huggingface-hub",
        "tokenizers",
        "pyarrow",
        "playwright",
        "swanlab",
        "triton",
    ]
    print(path)
    for name in wanted:
        if name in packages:
            print(f"  {name}=={packages[name]}")
PY
```

```sh
uv run python - <<'PY'
import json
import subprocess
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="omnijev-media-probe-") as temp_dir:
    path = Path(temp_dir) / "probe.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=16x16:rate=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:sample_rate=16000",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    result = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    data = json.loads(result.stdout)

print(
    {
        "streams": [
            {
                "codec_type": stream.get("codec_type"),
                "codec_name": stream.get("codec_name"),
                "sample_rate": stream.get("sample_rate"),
                "width": stream.get("width"),
                "height": stream.get("height"),
            }
            for stream in data.get("streams", [])
        ]
    }
)
PY
```

Remote metadata probe:

```sh
ssh -o ControlMaster=auto \
  -o ControlPath=/tmp/omnijev-ssh-%C \
  -o ControlPersist=10m \
  -p 48926 root@region-42.seetacloud.com \
  'cd /root/autodl-tmp/lite-omnijev &&
   source infra/bootstrap/autodl-env.sh &&
   for py in .venv/bin/python /root/autodl-tmp/envs/data/bin/python /root/autodl-tmp/envs/serve-next/bin/python /root/autodl-tmp/envs/serve/bin/python; do
     if [ -x "$py" ]; then
       echo "$py"
       "$py" - <<'"'"'PY'"'"'
import importlib.metadata as md

names = (
    "torch", "transformers", "peft", "accelerate", "safetensors", "vllm",
    "fastapi", "uvicorn", "httpx", "pydantic", "huggingface-hub",
    "tokenizers", "pyarrow", "playwright", "swanlab",
)
versions = {}
missing = []
for name in names:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        missing.append(name)
print({"versions": versions, "missing": missing})
PY
     fi
   done'
```

## Before The Next GPU Run

Do not claim offline runtime readiness until these are resolved or explicitly accepted as blocked:

- Valen `checkpoint.pt` must remain quarantined until a tensor-only conversion and nested checkpoint mapping are audited. Do not load the raw `.pt` checkpoint in normal inference or service paths.
- Quantization tooling source is cached, but the independent quantization environment and any quantized artifacts remain pending.
- llama.cpp, Ollama, and LiteRT-LM source archives are cached, but platform builds, binaries/installers, logits/audio checks, and release packaging remain pending.
- Android prerequisites are absent. iOS prerequisites exist locally, but that only proves the local build host has Xcode and an iPhoneOS SDK.
- GPU checks remain pending: `nvidia-smi`, `torch.cuda.is_available()`, `vllm.collect_env`, vLLM startup, model load, selected-token logprob parity, FP8/W4A16 kernels, and any latency claim.

Future LoRA adapters, calibration tables, distilled E2B checkpoints, GGUF files, quantized weights, mobile packages, and release images are generated outputs. They should be listed in release manifests after training and conversion, not downloaded as prerequisites now.

