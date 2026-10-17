# 02 · release-002 产品化:serving 复验 + 网关入包 + 真实浏览器验收(交付链③)

状态:done · 所有者:主会话 · 2026-10-18 · 终态见文末

## 闭环定义

模型已定版(release-002 = D3P-600 767be09b,merged 权重,用户 18 日确认不再训练)。本票把
release-002 从"权重+证据包"推进到"可服务、可浏览器验收的产品形态",并产出 ⑥ 所需 serving 性能数据。

## 验收(冻结)

1. **serving 复验**:`scripts/s0_parity.py hf-reference`(基座 ee0ef60 + 767be09b adapter,评测栈
   before-gpu-teacher-job-final-001,CUDA,公开 100 案例)产出 reference-ready;vLLM 0.26.0 按
   `docs/runbooks/service-environment.md` 固定配方 serve **release-002/model/merged**;
   `compare-vllm` 两遍 repeat exact + parity maxΔ<1e-3 → 同时证明
   merged≡adapter 与 serving≡HF,不修改冻结 harness(其 verified_assets 仅认官方基座目录)。
   **数值配方(修正)**:沿用 S0 已准入参考配置(`reports/s0-full100-gpu-acceptance-001`)
   ——**float32**、`NVIDIA_TF32_OVERRIDE=0`、`TRITON_F32_DEFAULT=ieee`、eager、禁前缀缓存、
   kv 4GiB、hf-config-path 副本仅改 root/audio/text/vision dtype;H20 上 BF16 已实测不过
   <1e-3 门,BF16 serving 质量照旧标 pending,不擅改阈值。
2. **API 复验**:网关 canonical-chat 上线(serve env,官方基座 metadata 作 LITE_MODEL_DIR,受信
   媒体根=s0-prepare-public-001),`scripts/s0_gateway_parity.py compare-gateway` 100 案例两遍
   一致且 vs HF/vLLM 报告 maxΔ<1e-3;`/health` inference_configured=true。
3. **真实浏览器验收**:`scripts/browser_s0_cpu.py --base-url`(live 分支,首次真实服务运行,
   Mac Chromium over SSH 隧道)6 案例全 200 + 422 隐私负例,报告 passed_live_browser_contract_only。
4. **gateway 入包**:release-002/gateway/ 含 launch-vllm.sh、launch-gateway.sh、README(配方、
   身份核验步骤、诚实限制)、验证回执;SHA256SUMS 重生成。
5. **⑥ 数据**:serving 显存实测、网关端到端 Server-Timing(p50/p90),补 reports/s4-perf-report-001.md。

## 范围外(不因本票扩散)

- 校准激活:vLLM recipe(bf16/native_lora receipt)合同下 merged 形态被拒,如实标 serving 校准
  pending;网关返回 calibration: unfitted,不静默降级。
- 量化端侧(④)、HF 发布(⑤,须用户 token)、生产并发/延迟门槛、Playground 前端。
- 长上下文:max-model-len 2048 沿用 P0 可复现配置;生产长度扩容另票。

## 依赖与资产

- 盒 GPU(H20,空闲);S0 契约资产在盒(/root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001);
- 工作树 /root/autodl-tmp/s6-release-serving-001/repo(= main a62a3ca,git archive 同步);
- serve env 硬链接克隆 + av 15.1.0/soundfile 0.13.1(锁内版本);Mac .venv 有 playwright+chromium。

## 记录

- 2026-10-18 开票,验收冻结;GPU 计划串行:hf-reference ×1 → vLLM up → compare-vllm → 网关 → 浏览器。
- **发现①(权重工件)**:现 release `model/merged`(bf16 merge_and_unload)相对
  fp32(base+adapter) 前向偏差 **1.46e-2**(6 案例探针,`s6-merge-rounding-probe-004.json`),
  构造上过不了 <1e-3 parity 预算;fp32 内存合并偏差 **1.53e-05**(`s6-fp32merge-probe-001.json`)。
  处置:重建 `model/merged-fp32/`(31.76G,`finalize_merged_fp32.py`,box scratch)。
  **后续升级(发现④)**:merged 工件(含 fp32 重合并)是 HF 共享存储 compact 布局
  (2076 键;layers 24–41 的 k_norm/k_proj/v_proj 被折叠,其中 36 个 k/v 张量在基座平面
  文件中独值),vLLM 0.26 按平面名加载拒载;HF 侧语义等价(探针 1.53e-05)。**serving
  身份改为官方基座平面布局 + d3p600 native LoRA**(`--enable-lora --lora-modules
  decision=<adapter>`,P0 配方):S0 已证基座平面文件可服务、adapter 目标不含 24–41
  k/v(无共享歧义)、且 native_lora 是校准合同唯一接受的 adapter_mode。merged 工件保留
  用于 HF 分发/量化,如实标注。
- **发现②(存量回归)**:`scripts/assets.py` METADATA 白名单缺 `.gitattributes`,而
  configs/project-model-assets.json 的 ee0ef60 文件列表含它 → 该 manifest 路径下
  verified_assets 必炸。已修(白名单补名),tests/test_assets.py 22/22。
- **发现③(文档错误)**:`docs/runbooks/s0-parity.md` 的 `--asset-manifest configs/...` 示例
  不可用(configs 是整项目清单,`bytes` schema);实际通过的验收用 per-model
  `/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json`(sha 207ce3a8…,与
  s0-full100-gpu-acceptance-001 记录一致)。runbook 已改。
- 环境:serve env 硬链接克隆至 s6-release-serving-001/envs/serve + av 15.1.0/soundfile 0.13.1
  (清华镜像;PyPI 直连龟速);vLLM 0.26.0+transformers 5.13.0+torch 2.11.0+cu130 实测导入 OK;
  网关 import 链在该 env 通过(peft 惰性,无需安装)。

## 终态(2026-10-18 凌晨,三项验收全过)

1. **serving parity ✅**:`s6-serving-compare-vllm-mergedfp32-002.json`——vLLM 两遍**精确可重复
   (Δ=0.0)**;vs HF 参考 raw 5.75e-6 / joint 5.67e-6 / pU 3.6e-10,全部 << 1e-3 门。
   同时证明:merged-fp32 ≡ base+adapter(评测权威栈)、vLLM serving ≡ HF、平面键物化无损。
2. **网关 parity ✅**:`s6-serving-gateway-parity-001.json`——100 案例两遍精确一致;vs HF/vLLM
   报告 conditional Δ=0.0、p_insufficient Δ=0.0(逐位一致);`/health` canonical-chat +
   inference_configured=true + model_version=767be09b0919235d。
3. **真实浏览器 ✅**:`passed_live_browser_contract_only`(首次 live 分支实战;Chromium 153 →
   SSH 隧道 → 真网关 → 真权重)6 案例 200 + 422 隐私负例;时延入 ⑥ `s6-perf-report-002.md`。
4. **入包 ✅**:release-002/gateway/{launch-vllm.sh,launch-gateway.sh,make_parity_records.py,
   README.md,receipts/(9 份回执含浏览器报告)};model/merged-fp32-flat/(serving 身份,硬链接);
   SHA256SUMS 重生成(47 文件)并全量校验。
5. **serving 形态定版**:merged-fp32-flat(平面键,2130 键,共享层死名以运行时真源值填充)。
   校准激活仍 pending(Binding 合同要求 native_lora receipt;网关如实返回 calibration: unfitted)。

过程烧坑(全部留 log):盒 console-script shebang 指旧 env(改显式解释器);vLLM 0.26 默认开
前缀缓存破坏两遍精确性(1.8e-6,禁用后 0.0);compare 报告拒覆盖(harness 正确,路径要轮换);
`pkill -f` 匹配自身 ssh 命令行两次(教训:远端清理不用宽 pattern)。
