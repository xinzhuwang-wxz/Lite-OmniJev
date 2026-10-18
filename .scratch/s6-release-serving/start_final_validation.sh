#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/s6-release-serving-001
B=/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2
cd "$R"
# Fresh view preserves all previous run artifacts. No broad process matching.
SERVE_ENV="$R/envs/serve" MODEL_DIR=/root/autodl-tmp/s6-hf-repo BASE_DIR="$B" VIEW="$R/serving-view-bound-final" CONFIG_DIR="$R/config-bound-final" \
  nohup bash "$R/repo/serve/deploy/launch-vllm.sh" > "$R/logs/vllm-final-flat.log" 2>&1 < /dev/null &
echo $! > "$R/final-vllm.pid"
GATEWAY_PYTHON="$R/envs/serve/bin/python" REPO="$R/repo" LITE_MODEL_DIR="$B" LITE_MEDIA_ROOT=/root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001 \
  nohup bash "$R/repo/serve/deploy/launch-gateway.sh" > "$R/logs/gateway-final-flat.log" 2>&1 < /dev/null &
echo $! > "$R/final-gateway.pid"
echo FINAL_VALIDATION_LAUNCHED
