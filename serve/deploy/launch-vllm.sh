#!/usr/bin/env bash
# lite-omnijev-release-002 merged weights on the pinned vLLM serving stack.
#
# Default recipe = the admitted S0 numerical-reference configuration
# (reports/s0-full100-gpu-acceptance-001): float32, TF32 off, Triton IEEE,
# eager, prefix caching disabled, single sequence. BF16 serving parity was
# measured failing the <1e-3 gate on H20 and remains pending separate evidence.
#
# SERVE_ENV: python env with vllm==0.26.0 / transformers==5.13.0 (infra/serve lock).
# MERGED_DIR: <release-002>/model/merged (identity recorded in MERGED_FROM.json).
# BASE_DIR: official base snapshot; provides tokenizer metadata (merged dir has
#   none) and the source config for the FP32 config copy.
set -euo pipefail
: "${SERVE_ENV:?set SERVE_ENV to the pinned serving env}"
: "${MERGED_DIR:?set MERGED_DIR to the release merged-weights directory}"
: "${BASE_DIR:?set BASE_DIR to the official base metadata snapshot}"
DTYPE="${DTYPE:-float32}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
MODEL_LEN="${MODEL_LEN:-8192}"
SERVED_NAME="${SERVED_NAME:-lite-omnijev-e4b-d3p600}"
KV_BYTES="${KV_BYTES:-4294967296}"

# The dtype copy is a derived serving artifact; keep it OUT of the release dir.
CONFIG_COPY="${CONFIG_COPY:-$PWD/config-fp32-copy.json}"

# Serving view: the merged dir carries weights only; vLLM needs the official
# metadata (tokenizer/processor/chat template) beside them. Hardlink the
# weights (same inodes -> same identity, no extra disk) and copy base metadata.
VIEW="${VIEW:-$PWD/serving-view}"
if [ ! -f "$VIEW/config.json" ]; then
  rm -rf "$VIEW"
  mkdir -p "$VIEW"
  cp "$MERGED_DIR"/config.json "$MERGED_DIR"/generation_config.json "$VIEW/" 2>/dev/null || true
  ln "$MERGED_DIR"/model*.safetensors* "$VIEW/" 2>/dev/null || true
  find "$BASE_DIR" -maxdepth 1 -type f ! -name "*.safetensors*" ! -name "config.json" \
    -exec cp {} "$VIEW/" \;
fi

if [ ! -f "$CONFIG_COPY" ]; then
  python3 - "$BASE_DIR/config.json" "$CONFIG_COPY" <<'PY'
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
exec "$SERVE_ENV/bin/vllm" serve "$VIEW" \
  --hf-config-path "$CONFIG_COPY" \
  --tokenizer "$BASE_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" --port "$PORT" \
  --dtype "$DTYPE" --tensor-parallel-size 1 \
  --max-model-len "$MODEL_LEN" --max-num-seqs 1 \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --gpu-memory-utilization 0.75 --seed 42 \
  --enforce-eager --generation-config vllm \
  --logprobs-mode raw_logprobs --max-logprobs 32
