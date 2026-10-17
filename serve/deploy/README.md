# release serving 部署配方(release-002 实测通过形态)

目标:把 lite-omnijev-release-002(merged 权重)按 S0 固定协议提供 `/v1/systemone` 决策服务。
两个进程、两个环境,不要混用(依据 `docs/runbooks/service-environment.md`):

1. **vLLM 0.26.0 后端**(serving 固定栈,transformers 5.13.0):serve `model/merged/`,
   `--logprobs-mode raw_logprobs`,供网关取原始答案 logprobs。数值配方 = S0 已准入参考
   (float32 + TF32 off + Triton IEEE + eager + 禁前缀缓存;`reports/s0-full100-gpu-acceptance-001`);
   BF16 serving 已实测不过 <1e-3 parity 门,质量证据 pending。
2. **网关**(FastAPI,canonical-chat):输入准备用**官方基座 snapshot 的 metadata**
   (tokenizer/processor/chat template 与训练评测同一冻结版本;merged 目录不含 metadata)。
   权重身份由 vLLM 启动记录核验,`LITE_MODEL_VERSION` 记 adapter SHA 短码 `767be09b0919235d`。

## 启动

```bash
# 终端 A:后端(15G bf16,H20 实测 ~3 分钟 ready)
SERVE_ENV=/path/to/serve-env MERGED_DIR=<release>/model/merged \
  bash launch-vllm.sh          # 127.0.0.1:8001

# 启动记录(核对 /proc/<pid>/cmdline 与上方命令一致后):
python - <<'PY'  # 生成 server-record 供 s0_parity compare-vllm / 网关 operator 记录
...

# 终端 B:网关
GATEWAY_PYTHON=/path/to/serve-env/bin/python REPO=/path/to/repo \
LITE_MODEL_DIR=<base snapshot ee0ef60> LITE_MEDIA_ROOT=<受信媒体根> \
  bash launch-gateway.sh       # 127.0.0.1:8000
```

验证链(全部通过才可称 serving 复验完成,报告不覆盖、失败留证):

```bash
python -m scripts.s0_parity hf-reference  --model-dir <base> --adapter-dir <d3p600> ...   # 评测栈,GPU
python -m scripts.s0_parity compare-vllm  --reference <hf报告> --server-record <记录> ... # vs merged serving
python -m scripts.s0_gateway_parity compare-gateway --hf-vllm-report <compare报告> ...    # 网关 API
python -m scripts.browser_s0_cpu --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --base-url http://127.0.0.1:8000 --model-version 767be09b0919235d --output-dir <新目录>  # 真实浏览器
```

## 诚实边界(2026-10-18 实测记录)

- 网关返回 `calibration: unfitted`、`model_quality_verified: false`:温度校准绑定
  Transformers bf16 receipt,当前 Binding 合同拒绝 vLLM/merged 形态激活;校准 artifact 在
  `calibration/` 保留,激活待 native_lora recipe 的真实 serving 派生温度。
- `--max-model-len 2048 --max-num-seqs 1 --enforce-eager` 为 S0 可复现配置,非生产延迟/并发声明。
- 媒体边界同 `docs/runbooks/gateway-media.md`:单 audio/单 video、≤8 引用、64MiB/40MP 限制,组合外显式 422。
