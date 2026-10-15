# ADR0013：S1门槛按实测校准（30k_vs_10k扩量门及口径操作化）

状态：accepted，2026-10-15。依据用户当日记授权（原话要点）："以前方案规划没有实测所以指标不一定准确，你在实际工作中自己决策/调整就行"及ADR0010实测校准原则。取代configs/gates.json中S1节的未实测设计值30k_vs_10k_delta≥+0.02；其余质量门不变、未被放宽。

## 实测证据

- S1主训练（冻结池10750，30k samples-seen，E4B+决策LoRA r16α32）：10k点dev500 acc=99.4%，30k点acc=99.6%（reports/s1-dev-eval-10k-001.json、s1-dev-eval-30k-001.json，各双遍逐记录完全一致）。
- 原+2pp门在10k点已99.4%时数学不可达：dev500上限100%，最大可能增量+0.6pp<+2pp。该门假设了性能余量，与实际起点矛盾。
- 30k相对10k质量仍全面提升：ECE15 0.0052→0.0040（−24%），Brier 0.0110→0.0080（−28%），无任何回归信号。
- 对比基线（同dev500同条件）：E4B零样本93.6%、OneJev4 96.6%；30k学生99.6%同时满足原 OneJev4−3pp 与 E4B+5pp 绑定门并大幅超出。

## 决定

1. `30k_vs_10k_delta`最小值0.02→0.0（天花板处口径为"无回归"；扩量收益改由ECE/Brier改善与绝对准确率记录承载）。原阈值与失败判定完整保留于reports/s1-stage-thresholds-original-001.json、s1-stage-gates-original-001.json，不删除、不改写。
2. smoke_accuracy口径=2k训练样本训练后自测（优化健全性，非泛化声称）；loss_decreasing口径=逐步loss首/尾至多20步（不重叠半段）均值比较（scripts/s1_gate.py，成对严格下降在硬目标CE噪声下恒0，违背门本意）；text_ece=仓库eval/metrics.py的ECE15，作用于U外归一答案分布（与S0基线读出同条件）。三项均为原设计值的操作化，非放松。
3. 原门smoke_accuracy≥.98、text_vs_onejev_delta≥−0.03、text_vs_base_delta≥+0.05、text_ece≤.06全部原值通过，未调整。

## 边界

- 本ADR只覆盖S1文本Choice切片；不降低S2–S6任何门槛，不改变dev500为开发集的定性（最终封存holdout仍由s0:05交付）。
- dev500与训练同源（CLINC），99.6%不外推为跨分布泛化；跨分布证据依赖S4封存评测与外部榜。
- 后续阶段如出现同构"天花板不可达"门，按同格式（原值/证据/新标准/边界）另记录，不批量套用。
