# P0 原始契约审计

结论：需补证据。原始方案不支持把 S0 的 `<1e-3` 概率一致性失败改写成通过，也不支持启动 S1；但原始 P0 出口本身是“100 条样本端到端冒烟流水线”，严格 HF/vLLM 概率门槛主要写在阶段 0/S0。若按原始 P0 工程闭环判定，现有证据已覆盖训练、恢复、持久化和真实 HTTP/浏览器返回结果；还需要把最新 FP16 网关/确认批证据正式落到 `p0:04`，并明确 P0 报告里把 S0 parity 标为未通过。

## 必须证明的事

本审计只回答三个问题：

1. 原始文本是否把 `<1e-3` HF/vLLM 概率一致性设为 P0 集成票的必要通过条件。
2. 现有 GPU 训练、恢复、备份、浏览器/API 与确认批证据是否足以支持“P0 工程冒烟闭环”。
3. 即使 P0 工程闭环可关，是否仍必须保持 S0 parity 未过、S1 不得开始。

## 原始契约证据

原始 HTML 明确把 P0 定义为基建与规范出口：`P0 基建与规范 ... 用 100 条样本跑通「构建数据 → 训练 10 步 → 评测 → 推理服务返回结果」的端到端冒烟流水线，并在 CI 中每次提交自动运行`。这句话没有写 `<1e-3` 数值门槛，重点是 100 样本、10 步、评测、推理服务返回结果。

原始 HTML 在阶段 0 写明严格概率门槛：`阶段 0：基线与评测框架（不训练...）` 的通过标准包括 `vLLM 与 HF 读出的概率差 < 1e-3（100 题抽样；确认 vLLM 返回的是未经采样参数处理的原始 logprobs）`，且失败处理为 `读出不一致 → 先修模板和分词，不进阶段 1`。这支持把 `<1e-3` 作为 S0/S1 前置门槛，不支持 S1 启动。

同一原始 HTML 在通用提交验收里还写了 `读出一致性测试（HF 与 vLLM 概率差 < 1e-3）`。这意味着如果把 P0 解释成完整 CI/规范落地，读出一致性仍是一个未满足的全局工程缺口；但项目本地拆票已经把真实 P0 集成票降到“差异如实记录”，把严格跨引擎概率闭环移到 S0。

`docs/02-project.md` 与此一致：P0 行是 `真实基座 100 样本→训练10步→评测→服务；配置/环境/权重/数据可追溯`；S0 行才包含 `100题概率一致性`。同文件还明确说 `P0十步GPU训练和恢复已有证据，跨引擎概率门槛尚未通过，不能宣称模型质量达标`。

`configs/gates.json` 把 `hf_vllm_max_probability_delta.max_exclusive: 0.001` 放在 `S0` 下，没有放在 P0 下。

## 本地票据证据

`.scratch/p0/spec.md` 的 P0 用户故事要求 `评测者通过相同模板从HF/vLLM读出完整合法token概率，不能用mock代替`，但没有写 `<1e-3`。`Further Notes` 写了 `科学门槛未过不扩大数据`。

`.scratch/p0/issues/04-real-integration.md` 的验收项要求真实 Gemma E4B LoRA 十步、重载 checkpoint、经 HF 与 vLLM/网关返回可观察结果，并要求 `GPU/HF/vLLM差异如实记录`、`P0报告明确后续S0尚未通过`。该票没有把 `<1e-3` 写成关闭条件，但目前状态仍是 blocked/open，需由主会话基于最新证据更新。

`.scratch/s0/spec.md` 与 `.scratch/s0/issues/03-multimodal-parity.md` 明确保留 S0 严格门槛：`100题概率差严格<1e-3`、raw logprobs 覆盖、相同 100 题重跑一致、真实媒体路径等。`.scratch/s0/issues/08-s0-integration.md` 也要求全部 S0 门槛通过才进入 S1。

## 已满足的 P0 工程证据

训练闭环：`reports/gpu-train-smoke.json` 为 `status: completed`，`steps_completed: 10`，`trainable_parameters: 34881536`，`adapter_sha256: 97db02cd...`，`peak_cuda_bytes: 17296830976`。这满足 100 样本十步真实训练和仅 LoRA 训练的核心证据。

恢复闭环：`reports/p0-resume-equivalence.json` 为 `status: passed`，`losses_identical: true`，`adapter_sha256_identical: true`。这满足完整 checkpoint 恢复不是 warm start 的证据。

持久化/备份：`reports/p0-gpu-backup.json` 记录 `checkpoint_step: 10` 与同一 adapter sha。若主线程保留远端/本地资源清单，这一项可支撑 P0 产物可追溯；但 `p0:04` 的“资源清单与关卡通知”仍需正式更新为“用户要求继续开卡，因此仅记录资源清单，不通知关卡”。

HF 参考：`reports/p0-base-hf-001.json` 与 `reports/p0-adapter-hf-001.json` 均为 `status: reference-ready`，带 manifest、data、template、inputs 与 adapter hashes。说明训练与读出参考侧可追溯。

真实浏览器/API：`reports/p0-browser-fp16-001.json` 记录 Chromium `browser_version: 153.0.8010.12`，`health_status: 200`，请求状态含 `200` 与负向 `422`，返回体中明确 `validation_status: GPU parity and calibration pending`。这可证明候选网关真实返回并如实暴露未完成状态；它不是 parity 或模型质量通过证据。

## 未通过或不得误称通过的证据

严格 P0/S0 数值对照仍失败：`reports/p0-base-vllm-triton-001.json` 为 `status: failed`，`max_conditional_and_U_difference: 0.061476704094846624`，`max_raw_probability_difference: 0.05749319682582993`。`reports/p0-adapter-vllm-triton-001.json` 为 `status: failed`，`max_conditional_and_U_difference: 0.04401873806938439`，`max_raw_probability_difference: 0.003427832903781569`。这些不能被写成 `<1e-3` 通过。

FP32 只证明基座诊断：`reports/p0-fp32-base-full-comparison.json` 为 `status: passed`，最大差约 `7.9e-6`，但没有证明独立 LoRA 正式服务路径，也不能替代低精度产品完成。

影子确认批不放行：主工作树的 `reports/confirmation-base-vllm-001.json` 与 `reports/confirmation-adapter-vllm-001.json` 均为 `status: failed`，其顶层 failed 来自原始严格数值对照；独立 shadow_assessment 为 review_required，来自尾部复核：base `joint_TV.max: 0.011289845561305085`，adapter `joint_TV.max: 0.022916160439071978`，聚合预算内但触发 `joint_TV_review_cases`。`configs/candidates/parity-shadow-v1.json` 写明 `release_or_stage_acceptance: false`，所以确认批最多支持后续 ADR/正式 gate 讨论，不能作为 P0 工程验收的替代，也不支持 S0 放行。

BF16 低精度风险是真实的：`reports/calibration-bf16-summary-001.json` 中 adapter `joint_TV.max: 0.24488756403077794`，并有 joint top 变化。FP16 定标更好，但 `reports/calibration-fp16-adapter-impact-001.json` 仍有 `joint_TV.max: 0.014920934950270377`；这说明阈值调整必须作为正式 gate 变更，而不是把现有失败说成通过。

## 判定

P0 原始工程冒烟闭环：需补证据。按原始 P0 文本，`<1e-3` 不是 P0 端到端冒烟流水线的必需通过条件；现有训练、恢复、备份、HF 参考、真实浏览器/API 证据足以支持“工程闭环基本成立”。但主状态和 `p0:04` 仍未正式更新，且 P0 报告必须明确 S0 parity 未过、候选网关只是如实返回 pending。

S0 parity：不通过。严格 `<1e-3` 在现有 BF16/TRITON base 与 adapter 上失败；影子策略仍不是正式 gate，且确认批触发复核。

S1 启动：不支持。原始 HTML 对阶段 0 失败处理写明 `读出不一致 → 先修模板和分词，不进阶段 1`；本地 S0 票据也全部依赖 `p0:04` 与 S0 集成门槛。

## 建议的收口方式

1. 不改 `configs/gates.json`，不声称 `<1e-3` 通过。
2. 在主线程把 `p0:04` 改为“P0 工程闭环关闭 / S0 parity blocked”前，补一份 P0 工程报告，引用训练、恢复、备份、浏览器/API、候选返回 pending 的证据。
3. 若要把 P0 从“blocked by strict parity”改为“done with S0 parity blocked”，需要同步更新 `plan/STATUS.md`、`p0:04`、`docs/02-project.md` 的状态措辞，说明依据是原始 P0/S0 分界，而不是阈值放宽。
4. S0 继续保留 `<1e-3` 或通过 ADR/用户批准定义新的正式 gate；确认批的 `review_required` 不能当通过。

## 本审计执行证据

- 读取并比对：`docs/Lite-OmniJev设计方案-最终版.html`、`docs/02-project.md`、`configs/gates.json`、`.scratch/p0/spec.md`、`.scratch/p0/issues/04-real-integration.md`、`.scratch/s0/spec.md`、`.scratch/s0/issues/03-multimodal-parity.md`、`.scratch/s0/issues/08-s0-integration.md`。
- 读取报告：`reports/gpu-train-smoke.json`、`reports/p0-resume-equivalence.json`、`reports/p0-gpu-backup.json`、`reports/p0-browser-fp16-001.json`、`reports/p0-base-vllm-triton-001.json`、`reports/p0-adapter-vllm-triton-001.json`、`reports/p0-fp32-base-full-comparison.json`、`reports/calibration-*.json`，以及主工作树 `reports/confirmation-*.json`。
- 未运行 GPU，未读取 holdout，未修改代码、门槛、票据或 STATUS。
