#!/usr/bin/env bash
# Canonical-chat gateway in front of the pinned vLLM backend (see README.md).
# GATEWAY_PYTHON: env with fastapi/httpx/transformers matching the frozen input-prep stack.
# REPO: Lite-OmniJev checkout the gateway code is imported from.
# LITE_MODEL_DIR: official base snapshot metadata (tokenizer/processor/chat template).
# LITE_MEDIA_ROOT: trusted root holding only verified development media.
set -euo pipefail
: "${GATEWAY_PYTHON:?set GATEWAY_PYTHON}"
: "${REPO:?set REPO}"
: "${LITE_MODEL_DIR:?set LITE_MODEL_DIR to the official base metadata snapshot}"
: "${LITE_MEDIA_ROOT:?set LITE_MEDIA_ROOT to the trusted media root}"
export LITE_UPSTREAM_URL="${LITE_UPSTREAM_URL:-http://127.0.0.1:8001}"
export LITE_UPSTREAM_MODEL="${LITE_UPSTREAM_MODEL:-lite-omnijev-e4b-d3p600}"
export LITE_MODEL_VERSION="${LITE_MODEL_VERSION:-767be09b0919235d}"
export LITE_RAW_LOGPROBS_CONFIRMED="${LITE_RAW_LOGPROBS_CONFIRMED:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PORT="${PORT:-8000}"
cd "$REPO"
exec "$GATEWAY_PYTHON" -m uvicorn serve.gateway.app:app \
  --host 127.0.0.1 --port "$PORT" --no-access-log --log-level warning
