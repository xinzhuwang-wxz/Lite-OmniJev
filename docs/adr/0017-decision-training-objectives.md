# ADR0017：从目标出发安排监督与奖励优化，避免臆称复现RLCD

状态：accepted-for-experiment-design，2026-10-15。用户先建议微调后尝试RL，随后明确撤回固定顺序，授权负责人在项目各方面自主创新、按实测决定方法和时点。产品达标为主线，创新收益经对照与独立gate确认后采用。

## 已核对的公开证据

- TypeSafe官方称Jev使用RLCD（Reinforcement Learning for Calibrated Decisions），面向校准决策，包含单步分类/路由等：https://typesafe.ai/blog/introducing-system-one-models-and-jev 、https://docs.typesafe.ai/introduction/machine-learning-primer 。公开材料未给出奖励公式、优化器、采样方式或可复现训练配方；不将PPO/GRPO/DPO或任何社区算法臆称为官方RLCD。
- 社区OneJev公开为冻结vision tower、对答案概率用cross-entropy+Brier微调：https://github.com/OmniJev/OneJev/blob/main/README.md 。与官方Jev和受禁用模型分开。
- 原HTML已安排R1监督/GRPO对照及R2–R5，并非只允许多步agent用RL。单步/多步不是是否采用RL的判据；目标、监督信息、梯度方差、校准、算力才是。

## 实验顺序与有效性

1. 当前主线先提供真实同数据、同初始化的监督对照。c725485加入可选CE+Brier（所有合法答案+U联合概率，FP32、Brier为平方误差和）；零权重保持原CE/续训身份，非零权重进入checkpoint配置。当前E07新gold实验先CE，有收益即扩大验收，失败才接CE+Brier，避免为比较而拖慢达标。
2. 可在有具体瓶颈时提前做小规模奖励优化先导，不强制等所有微调结束；也不为使用RL增加无信息的采样。候选包括离线偏好与有限答案精确期望奖励；社区配方作对照，新设计不要求已有公开复现。
3. 本产品一次前向可得到2–20选项及U的完整分布。若所有动作奖励已知，能直接求 `sum(q(a)*r(a))` 并与固定reference KL及proper scoring结合；这是有限动作数学推导，不声称新颖、SOTA或官方RLCD。只基于gold正确性的奖励没有新增监督信息，纯奖励与KL不自动保证校准。
4. 原R1需先操作化：同一分布的`−Brier(q,y)`若作为所有动作相同、停止梯度的奖励，不能给出有效动作优势；若直接保留Brier梯度则是可导评分损失。不同提示变体的奖励不应未经定义就当同条件GRPO组。先明确动作相关奖励、参考策略、梯度路径与公平计算预算，再启动对照。
5. 新目标先数学/梯度测试，再固定小数据与卡时；只扩大有收益候选。比较accuracy、NLL/Brier/ECE、coverage、p(U)、置换一致性、旧模态遗忘与实际开机成本。封存只由独立gate报告汇总；明显有收益再做多种子与完整阶段门。

当前尚未实现/运行RL，不把监督CE+Brier冠以RL。原始方案保留；后续候选必须有独立提案与实测，不能以“颠覆性”作为放宽验收的理由。
