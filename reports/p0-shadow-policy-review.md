# P0 影子概率策略审查

审查对象：`configs/candidates/parity-shadow-v1.json`、`plan/proposals/p0-parity-policy-review.md`、`data/parity_fixture.py`、`reports/calibration-fp16-*-{hf,vllm,impact}-001.json`。本报告只审查能否进入影子确认批；不构成 P0、发布、生产或正式 gate 放行。

## 结论

可以继续执行 `parity-shadow-v1` 的影子确认批，前提是确认批必须按候选文件冻结执行，完整记录 `reject/review_required/shadow_supported`，且不得把 `review_required` 解释为通过。当前证据支持“FP16 LoRA 候选值得在未见确认集上验证迁移性”，不支持修改原严格 gate，也不支持生产风险声明。

我没有发现候选文件把已知失败静默抹平：它保留了 `release_or_stage_acceptance=false`、确认批不可事后筛选、确认后改规则必须换独立 cohort，并将 `joint_TV>0.01`、`normalized_expected_score_delta>0.01`、条件 top 集变化设为人工复核触发。这个设计足够作为下一步影子实验，不够作为正式验收契约。

## 候选指标复核

FP16 adapter calibration：

- 100/100 行身份匹配；conditional top 集变化 0；joint top 集变化 0；tie 0。
- mean joint TV = 0.000556，小于候选 0.002；p95 joint TV = 0.002742，小于候选 0.01。
- joint accuracy degradation = 0.0；joint Brier degradation = +0.000272，小于候选 0.005。
- 已知真值 ECE15 劣化：Choice -0.000000011、Noul -0.0000000002、Score +0.000304，均小于候选 0.005。
- Score normalized expected-score delta 最大值未超过 0.01；没有触发 Score 尾部复核。
- 仍有 1 个 `joint_TV>0.01` 尾部，必须进入 `review_required`：`parity-calibration-score-2-084`，target=U，reference p(U)=0.381238，candidate p(U)=0.366318，joint TV=0.014921。这个尾部主要是拒答概率漂移，不是答案 top 翻转。

FP16 base calibration 不是 LoRA 候选的放行对象，但作为基座风险背景必须保留：

- mean joint TV = 0.001738，小于 0.002；p95 joint TV = 0.007901，小于 0.01；joint top 集变化 0；tie 0。
- 有 4 个 `joint_TV>0.01` 尾部：`score-5-090` TV=0.022394、`score-10-099` TV=0.013363、`choice-3-017` TV=0.011715、`noul-074` TV=0.010673。
- base 的尾部覆盖 Score、Choice、Noul，说明正式 gate 不能只用 adapter 当前结果证明系统性安全。

BF16 calibration 已经证明低精度选择有真实决策风险：adapter 出现 1 个 joint top 变化，p(U) 0.616981 -> 0.372094，joint TV 最大 0.244888。这个结果支持选择 FP16 继续影子确认，但也说明“低精度可接受”必须由 dtype/backend 绑定，不能泛化到 BF16。

## 主要风险

`p(U)` 风险没有被正式独立成候选 requirement。当前 FP16 adapter 唯一尾部就是 p(U) 差，能被 `joint_TV>0.01` 捕获；但正式 gate 仍应保留显式的拒答指标，例如 p(U) delta 分位数、p(U) MSE 按 sufficient/insufficient 分层、拒答 top 决策变化和 calibration/confirmation 两批对比。只靠 joint TV 与 Brier，容易让“拒答边界漂移但标签未翻”的问题在平均指标里变钝。

Score 风险目前被人工触发器覆盖，但没有聚合 requirement。FP16 adapter 的 normalized Score delta 未超过 0.01，不能证明未来正式 gate 不需要 Score 预算；正式变更应保留 Score MAE/期望分 delta 分层，尤其是 2/3/5/10 级分别报告。

tie 风险在现有 FP16 calibration 中未出现，但候选描述要求含并列集合的 argmax 不变。正式 gate 必须在报告里直接输出 pairwise joint top set/tie 变化，而不只依赖 `truth_metrics.joint_unique_top1_accuracy`；准确率相同不能证明 top 集未变。

fixture 独立性有限。`data/parity_fixture.py` 生成 calibration=100、confirmation=100，二者 pool/generator family 分离，且每行 family_id 唯一；这足以防止把同一文本改名复用。但 manifest 已说明它们是 100 个内容隔离 case family，不是 100 个独立算法或生产样本。确认批最多验证候选规则在同类程序真值任务上的迁移性，不能声明真实多模态业务或生产校准。

## 对外表述边界

允许的表述：

- “FP16 LoRA 影子候选在 calibration 上满足 mean/p95 TV、accuracy、Brier、ECE 聚合预算，但触发 1 个 tail review，需要在未见 confirmation cohort 上检验。”
- “当前确认批若通过，也只是支持下一步 ADR/正式 gate 讨论，不替代原 P0 gate。”

禁止的表述：

- “P0 概率一致性已通过。”
- “<0.001 已证明过严，可以直接放宽。”
- “100 例零 top 翻转证明生产风险很低。”
- “review_required 等同于 shadow_supported。”

## 正式 gate 变更前必须保留

- 原模板、token 映射、U 概率公式、完整 raw logprob、非有限概率、身份/provenance 检查仍是零容忍。
- 正式 ADR 必须绑定 dtype/backend/model/adapter 范围，不能把 FP16 结果外推到 BF16 或其他 vLLM 后端。
- calibration 用于定标，confirmation 只用于一次性验证；若确认失败，保存 v1 失败记录，换 v2 时必须用新的独立确认 cohort。
- 正式报告必须包含 p(U) 分层、joint top/tie pairwise 变化、Score 期望分分层、Brier/ECE/accuracy degradation、所有人工复核尾部逐题证据。
- 不得读取产品 holdout 内容；阶段 4/5/6 仍只能接收隔离 gate 的汇总门槛报告。

## 复核命令

本审查使用本地 CPU 读取已有 JSON 并用 `uv run python` 重新生成 calibration fixture 元数据；未运行 GPU、未生成 confirmation 输出、未读取 holdout。

