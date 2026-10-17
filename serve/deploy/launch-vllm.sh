#!/usr/bin/env bash
# lite-omnijev-release-002 merged weights on the pinned vLLM serving stack.
#
# MODEL_DIR must be the FLAT-KEY merged artifact (base checkpoint key layout,
# 2130 keys). The plain merge_and_unload save is a compact tied-storage layout
# (2076 keys; text layers 24-41 k/v collapsed) that vLLM 0.26 refuses to load,
# and the bf16 merge additionally failed the <1e-3 parity budget (1.46e-2
# rounding; probes s6-merge-rounding-probe-004 / s6-fp32merge-probe-001).
# Native fp32 LoRA serving is not an alternative: vLLM 0.26 LoRA kernels
# assert fp16/bf16 only.
#
# Numerical recipe = the admitted S0 reference
# (reports/s0-full100-gpu-acceptance-001): float32, TF32 off, Triton IEEE,
# eager, prefix caching disabled, single sequence.
#
# SERVE_ENV: python env with vllm==0.26.0 / transformers==5.13.0 (infra/serve lock).
# MODEL_DIR: flat-key merged weights directory (config.json + model.safetensors).
# BASE_DIR: official base snapshot; provides tokenizer/processor metadata.
set -euo pipefail
: "${SERVE_ENV:?set SERVE_ENV to the pinned serving env}"
: "${MODEL_DIR:?set MODEL_DIR to the flat-key merged weights directory}"
: "${BASE_DIR:?set BASE_DIR to the official base metadata snapshot}"
DTYPE="${DTYPE:-float32}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
MODEL_LEN="${MODEL_LEN:-8192}"
SERVED_NAME="${SERVED_NAME:-lite-omnijev-e4b-d3p600}"
KV_BYTES="${KV_BYTES:-4294967296}"

# Serving view: hardlink the weights (same inodes -> same identity, no extra
# disk), keep the merged config, and copy the official metadata from the base
# snapshot (the merged dir carries weights only).
VIEW="${VIEW:-$PWD/serving-view}"
if [ ! -f "$VIEW/config.json" ]; then
  rm -rf "$VIEW"
  mkdir -p "$VIEW"
  cp "$MODEL_DIR"/config.json "$MODEL_DIR"/generation_config.json "$VIEW/" 2>/dev/null || true
  ln "$MODEL_DIR"/model*.safetensors* "$VIEW/" 2>/dev/null || true
  find "$BASE_DIR" -maxdepth 1 -type f ! -name "*.safetensors*" ! -name "config.json" \
    -exec cp {} "$VIEW/" \;
fi

# The dtype copy is a derived serving artifact; keep it OUT of the asset dirs.
# vLLM --hf-config-path expects a DIRECTORY containing config.json.
CONFIG_DIR="${CONFIG_DIR:-$PWD/config-fp32-dir}"
CONFIG_COPY="$CONFIG_DIR/config.json"
if [ ! -f "$CONFIG_COPY" ]; then
  mkdir -p "$CONFIG_DIR"
  "$SERVE_ENV/bin/python" - "$BASE_DIR/config.json" "$CONFIG_COPY" <<'PY'
import json, sys
source, target = sys.argv[1], sys.argv[2]
config = json.load(open(source))
changed = 0
for section in ("", "audio_config", "text_config", "vision_config"):
    node = config if section == "" else config.get(section)
    if isinstance(node, dict) and node.get("dtype") == "bfloat16":
        node["dtype"] = "float32"
        changed += 1
assert changed == 4, f"expected exactly 4 dtype fields, changed {changed}"
json.dump(config, open(target, "w"), indent=2)
PY
fi

export NVIDIA_TF32_OVERRIDE=0 TRITON_F32_DEFAULT=ieee
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$SERVE_ENV/bin:$PATH" # FlashInfer JIT needs the env's ninja
# The console script's shebang may point at another env (hardlinked clones);
# invoke through this env's interpreter explicitly.
exec "$SERVE_ENV/bin/python" "$SERVE_ENV/bin/vllm" serve "$VIEW" \
  --hf-config-path "$CONFIG_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" --port "$PORT" \
  --dtype "$DTYPE" --tensor-parallel-size 1 \
  --max-model-len "$MODEL_LEN" --max-num-seqs 1 \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --gpu-memory-utilization 0.75 --seed 42 \
  --enforce-eager --generation-config vllm \
  --no-enable-prefix-caching \
  --logprobs-mode raw_logprobs --max-logprobs 32
