#!/usr/bin/env bash
# Only execute after the user has enabled the GPU. This script never controls rental power.
set -euo pipefail
cd /root/autodl-tmp/lite-omnijev
source infra/bootstrap/autodl-env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SWANLAB_API_KEY
SWANLAB_API_KEY="$(cat /root/.config/lite-omnijev/swanlab.key)"
export LITE_OMNIJEV_SWANLAB_MODE=online
MODEL_DIR=artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2
RUN_DIR="artifacts/training/p0-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'Keep preparation in no-GPU mode; user must enable GPU first.'
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print({'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda})
PY
.venv/bin/python -m scripts.train_smoke preflight \
  --model-dir "$MODEL_DIR" --asset-manifest artifacts/manifests/e4b.json \
  --data artifacts/data/p0-train-100.jsonl --report reports/gpu-preflight.json
# Bound the first experiment. A timeout keeps completed-step checkpoints and returns nonzero.
timeout --signal=INT --kill-after=60 3600 .venv/bin/python -m scripts.train_smoke run \
  --model-dir "$MODEL_DIR" --asset-manifest artifacts/manifests/e4b.json \
  --data artifacts/data/p0-train-100.jsonl --output-dir "$RUN_DIR" \
  --report reports/gpu-train-smoke.json
printf 'Training completed at %s; HF/vLLM parity and API acceptance remain.\n' "$RUN_DIR"
