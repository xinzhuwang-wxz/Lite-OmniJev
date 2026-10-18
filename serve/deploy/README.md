# release serving 部署配方(release-002 实测形态)

目标:把 lite-omnijev-release-002(D3P-600 767be09b)按 S0 固定协议提供 `/v1/systemone` 决策服务。
两个进程、两个环境,不要混用(依据 `docs/runbooks/service-environment.md`):

1. **vLLM 0.26.0 后端**(serving 固定栈,transformers 5.13.0):**D3P-600 的
   merged-fp32-flat 权重**,恢复官方基座平面布局的 **2130 键**,并使用官方基座 metadata
   创建独立 serving view。`--logprobs-mode raw_logprobs` 供网关取原始答案 logprobs。
   数值配方 = S0 已准入参考(float32 + TF32 off + Triton IEEE + eager + 禁前缀缓存;
   `reports/s0-full100-gpu-acceptance-001`)。

   HF 普通 `save_pretrained` 会折叠 layers 24–41 的共享 k_norm/k_proj/v_proj,形成
   2076 键 compact 工件,vLLM 0.26 拒载;必须使用补齐平面键的派生工件。
   bf16 合并舍入实测 1.46e-2,未过 <1e-3 parity 门
   (`reports/s6-merge-rounding-probe-004.json`/`reports/s6-fp32merge-probe-001.json`)。
   native fp32 LoRA 也不可用:vLLM 0.26 的 LoRA kernel 只接受 fp16/bf16。
   最终 merged-fp32-flat 配方已在100案例双遍验证中通过,HF/vLLM 最大原始概率差
   5.752e-6 (`reports/s6-serving-compare-vllm-mergedfp32-002.json`)。
2. **网关**(FastAPI,canonical-chat):输入准备用**官方基座 snapshot 的 metadata**
   (tokenizer/processor/chat template 与训练评测同一冻结版本;merged 目录不含 metadata)。
   权重身份由 vLLM 启动记录核验,`LITE_MODEL_VERSION` 记 adapter SHA 短码 `767be09b0919235d`。

## 启动

从仓库根目录执行;路径不含空格时可直接使用以下赋值形式。

```bash
# 终端 A:后端(fp32 平面键工件;以下路径替换为实际绝对路径)
SERVE_ENV=/path/to/serve-env MODEL_DIR=/path/to/release/model/merged-fp32-flat \
BASE_DIR=/path/to/base-snapshot \
  bash serve/deploy/launch-vllm.sh          # 127.0.0.1:8001

# 启动记录:按 docs/runbooks/s0-parity.md 核对实际进程参数并保存 server-record,
# 供 compare-vllm 验证;不得把示例参数当作真实启动记录。

# 终端 B:网关
GATEWAY_PYTHON=/path/to/serve-env/bin/python REPO=/path/to/repo \
LITE_MODEL_DIR=/path/to/base-snapshot LITE_MEDIA_ROOT=/path/to/trusted-media \
  bash serve/deploy/launch-gateway.sh       # 127.0.0.1:8000
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
  `calibration/` 保留。当前服务使用 raw 概率;激活温度需要与实际 vLLM/float32/merged
  recipe 匹配的校准及来源绑定证据,不能直接套用 Transformers 温度。
- `--max-model-len 8192 --max-num-seqs 1 --enforce-eager`、4GiB KV cache 为默认
  serving parity 配置,非生产延迟/并发声明。H20 引擎显存实测39,056MiB;计时边界见
  `reports/s6-perf-report-002.md`。
- 媒体边界同 `docs/runbooks/gateway-media.md`:单 audio/单 video、≤8 引用、64MiB/40MP 限制,组合外显式 422。

## 产品输出与内部诊断

按ADR0024，默认API不返回`p_insufficient`，置信度语义标为`normalized_peak`。
复跑历史内部gateway parity脚本时，仅在隔离的本地诊断服务显式设置
`LITE_INTERNAL_READOUT_DIAGNOSTICS=1`，才返回原来的内部读出字段；不要将此配置用于产品演示。
根路径`/`提供Playground，媒体按受信服务器URI及SHA256引用；当前没有公网上传/鉴权。

切换版本或回滚时给该版本选择新的`VIEW`与`CONFIG_DIR`，例如`/runs/release-001/view`及
`/runs/release-001/config`。脚本校验来源与派生文件身份，目录不匹配会拒绝启动；
不会删除用户指定目录，也不会把上一个版本的权重悄悄用于新配置。
