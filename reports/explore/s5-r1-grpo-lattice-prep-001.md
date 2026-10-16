# s5-r1-grpo-lattice-prep-001 — GRPO-on-lattice 规格、CPU 推导与降级结论

状态:**CPU 完成reward/闭式推导/自检;结论=α/γ 项可分解回 CE、β 项不可分解但闭式可微,
按"不硬做 RL"规则降级为 B1 耦合辅助损失;采样式 GRPO 不排 GPU。**

## 规格原样(协调指令)

group=同视频问题格;reward=α·逐题正确性+β·格级结构一致性(极性孪生翻转=预测排序翻转)
+γ·p(U) 可答性正确;240s GPU 试点、固定种子、预注册门同 B1;与 CE-only 对照。

## CPU 实现(全部实跑,零 GPU)

- `.scratch/s4/r1_lattice.py`(stdlib):
  - `lattice_reward(rollout, golds, twin_groups, triple_groups, α,β,γ)` — 采样式 reward;
  - `expected_reward_closed_form(distributions, …)` — 独立单 token 分布下的期望 reward 闭式:
    α/γ 逐项线性;β孪生 `E[1[a≠b]]=1−⟨p_L,p_R⟩`;β传递(三元组)=8 个标签组合的多项式;
  - `coupled_loss(…)` = −闭式(精确可微代理);
  - `montecarlo_expected_reward(...)` — 对照估计器;
  - `load_grid_lattice(...)` — 从 B1 grid 工件抽 golds/43 孪生对/6 传递三元组/24 U 题。

## 自检结果(实跑输出)

- 闭式 vs Monte-Carlo(200k rollouts,固定种子):|gap|=0.0012 < 0.01 ✓
- **β=0 时** reward 对 q1 分布的数值梯度在孪生 q2 分布替换(yes 0.2→0.7)下不变 ✓(α/γ 可分解)
- **β=1 时** 同一梯度随 q2 分布显著改变(数值见 JSON)✓(β 不可分解为逐题 CE)
- 传递三元组一致性函数枚举 8 组合,含 U 破坏一致性的定义 ✓

## 推导结论(诚实降级)

1. 单 token 读出 + 独立采样下,GRPO/REINFORCE 的期望梯度=∇E[R],而 E[R] 有精确闭式 →
   采样只加方差不加信号;**纯 gold reward(α only)严格等价 advantage 加权 CE(已停路线,不重做)**。
2. β 孪生项是真信号(梯度耦合两个不同 prompt 的分布,CE 表达不了),但它同样是闭式可微的
   `<p_L,p_R>` 惩罚+CE 组合 → **正确形态是 B1 的耦合辅助损失,不是 RL**。
3. 仅当未来出现"不可闭式的格级指标"(如全局排 consistency 的组合计数不可多项式表达时)才值得
   采样式 GRPO;当前 126 题格全部可闭式,不满足该条件 → 本切片不排 RL 的 GPU 试点。

## 降级后的试点(与 B1 臂2 合并,见 s4-b1-grid-prep-001)

- 损失:`α·CE(gold 含 U)+β·⟨p_L,p_R⟩(43 孪生对)+β3·(1−P_consistent)(6 三元组,可选)`;
  默认 α=1.0、β=0.3、β3=0.1(小系数,结构项只作正则)。
- 需要 trainer 小扩展:同批内两前向共享 LoRA,损失在分布层耦合;数学已单测,接线代码待
  main review 后上 GPU;若接线不及窗口,臂1(CE-only)独立成立,门同 B1。
- 对照判定:两臂同数据/种子/步数,唯一差异=β 项;`Δ=臂2−臂1` 在 video150 与 DEV50 上
  同向为正才采纳 β 项;任一臂过 B1 门即按门处理。
