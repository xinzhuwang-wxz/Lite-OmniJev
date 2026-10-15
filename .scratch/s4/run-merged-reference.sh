#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/s1-text-lora-001
export PATH="/root/autodl-tmp/envs/serve-vllm026-offline-check/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVIDIA_TF32_OVERRIDE=0 TRITON_F32_DEFAULT=ieee CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT=/root/autodl-tmp/cache/vllm TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/cache/torchinductor TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
exec timeout --signal=INT --kill-after=30s 900 /root/autodl-tmp/envs/serve-vllm026-offline-check/bin/vllm serve /root/autodl-tmp/s1-text-lora-001/artifacts/s3-merged-fp32-reference-001 --tokenizer /root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 \
 --served-model-name s3-decision --host 127.0.0.1 --port 18001 \
 --dtype float32 --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 8192 \
 --gpu-memory-utilization 0.55 --seed 42 --enforce-eager --generation-config vllm --logprobs-mode raw_logprobs --max-logprobs 32 \
 --attention-backend TRITON_ATTN --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":1,"audio":1,"video":1}' --skip-mm-profiling \
 --no-enable-log-requests --no-enable-prefix-caching --no-async-scheduling --no-enable-chunked-prefill --kv-cache-memory-bytes 4294967296 \
 --hf-config-path /root/autodl-tmp/gpu-transition-ba4b22a/artifacts/fp32-hf-config-001
