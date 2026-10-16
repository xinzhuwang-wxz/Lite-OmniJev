# s5-r1b-permutation-grpo-prep-001 — R1-B 置换组 GRPO 有界试点:CPU 实现与不可分解性证明

状态:**CPU 完成(reward/group/KL 实现+闭式检查+变体数据已上传远端);GPU 试点待 main;全部预注册。**
按用户确认保留 RL 线并对标"Calibrated Typed Decisions via Reward Learning"范式:奖励含两个变体
——**R1-B-gold**(−Brier 对 one-hot gold)与 **R1-B-wsft**(−NLL 对谓词因子化 typed 目标,
复用 b1-wsft-grid 的 SoftTarget/hard/Insufficient 行,typed_target 映射到含 U 的完整标签分布);
两者共享 group(同题 8 个语义惰性变体)/GRPO 组优势/KL 锚框架。"世界状态因子化目标 × typed 奖励学习"
的组合按前沿检索与本次证明记录为零先例(诚实边界:检索覆盖同前沿报告 I1 口径)。
试点优先级:**R1-B-wsft > R1-B-gold**(前者与 B1+WSFT 数据零额外成本)。

## 规格落地

- **组**=同题 K=8 个语义惰性变体(包装模板 pv0..pv7:原句/"Answer yes or no:"/"Is the following true?"/尾缀/"Based on the video:"/"Question:"/"Watch carefully."/"Decide:"——绝不触碰比较词/对象/极性,gold 不变性按构造成立且逐行 unwrap 校验)。
- **奖励**(逐变体采样 rollout):`r_v = −Brier(p_v, gold) + λ·consistency_v`,λ=0.5;
  `consistency_v = #(a_v == a_{v'})/(K−1)`(组内其余变体)。
- **GRPO 组优势**:`A_v=(r_v−mean)/ (std+ε)`;**KL 锚**到起点 adapter(e465)分布,β=0.05。
- 数据:WSFT 292 行 ×8 = **2336 行**(target 层原样保留)+ legacy4 变体 1168 行,均已生成、
  pydantic 抽验通过、上传 `/root/autodl-tmp/s1-text-lora-001/artifacts/s4-ensemble-oracle-001/`
  (r1b-variants-k8 7f905a15…、legacy4 b62ef1c4…,ledger 用 {item_id,pool})。

## 闭式检查与不可分解性证明(①,实跑输出在 JSON)

模块 `.scratch/s4/r1b_permutation_grpo.py`(纯 CPU stdlib),K=8 示例分布上,**两个奖励变体都跑**:

1. 闭式 `E[R]=Σ_v[−Brier(p_v,g)]/K + λ·Σ_v⟨p_v, mean_{v'≠v}p_{v'}⟩/K`(gold)与
   `E[R]=Σ_v[−NLL(p_v,t_wsft)]/K + λ·Σ_v⟨…⟩/K`(wsft)均与 Monte-Carlo 200k rollouts 一致
   (|gap|≈0.0005,seed 13)✓。
2. **λ=0 时**两个变体的精度项(Brier/NLL)梯度在其它变体分布被替换后逐位不变 →
   精度项可按变体分解(=软gold/typed蒸馏,无需 RL)✓。
3. **λ>0 时**同一梯度显著改变 → consistency 项跨变体耦合,**不可分解为任何逐题 gold 损失**
   ——这是与已停"纯gold奖励RL"的本质区别,合法 RL 形态成立(两奖励变体同结论)✓。
4. 诚实附注:E[R] 仍有闭式 → 精确梯度形态("变体一致性正则 typed 蒸馏",8 前向可微)方差
   严格优于采样 GRPO;采样 GRPO 的不可还原成分只剩 KL 锚与 clipping(步长控制)。

## 240s 试点脚本(②,复用 B1 格+变体,固定种子 42)

- **形态 R1-B0(零 trainer 改动,今天可跑)**:legacy4 变体 1168 行当普通 WSFT 数据走
  `scripts.train_s1 run`(命令=launch-b1-wsft-pilot 臂1w 同款,仅换 data/ledger/steps=60→40、
  sample_count=1168;config 其余不变)。它测"变体增广"分量(无一致性/KL)。
- **形态 R1-B-exact(精确梯度,需 trainer 小扩展,差异说明)**:损失=
  `Σ_items Σ_v [Brier(p_v, t_wsft) + λ·(1−⟨p_v, p̄_{−v}⟩)]/N + β·KL(p_θ||p_init)`;
  批按"项×8变体"成组采样;K=8 全量 2336 行 240s 放不下,**按组下采样 40 项×8=320 前向**
  (~0.73s/前向 ≈ 234s,贴上限;实际以首 5 步耗时早停)。
- **形态 R1-B-grpo(采样式,用户要的 RL 标签形态,差异说明不实现)**:同 R1-B-exact 的组
  采样,但梯度来自 `E[Σ_v A_v·∇log π(a_v)]`(REINFORCE+组基线+KL 锚+clip 0.2);
  与 exact 的差异仅为估计器方差与步长控制,期望目标相同(证明见上)。
- trainer 扩展差异(两种形态共用):批构造按组、双份前向(record p_init 一次缓存)、
  损失项接线 ~60 行;未实现(遵守"只写差异说明")。

## 预注册门(③)

与 B1 臂同门:video150 ≥121/150 **或**关系/存在子项 ≥+3pp,且 DEV50 不回退(>1 题掉即停);
R1-B 附加判据:B0(纯增广)若 ≈臂1w(≤0.5pt 差)→ 增广分量无贡献,一致性信号只看 exact/grpo 臂。

## 优先级建议(④,窗口只够一个时)

**B1 臂2(孪生耦合损失)优先于 R1-B**,理由:
1. 命中面:孪生耦合直接打关系/存在错误簇(50 个已定位);R1-B 的 consistency 打的是
   表面扰动鲁棒性——前沿报告 I10 明确"置换一致性先例充分且**与实测错误结构不重叠**"。
2. 信号纯度:R1-B 奖励里可分解的精度项(−Brier/−NLL 对 gold/typed 目标)占主导(λ=0.5),
   新增信号只有一致性项;B1 臂2 的 β 项全部是新结构信号。
3. 成本:臂2 每对 2 前向、43 对;R1-B-exact 每项 8 前向,同预算样本量少 4 倍。
若窗口有第三格:**R1-B-wsft**(零数据成本、承载 typed 校准范式)> R1-B-gold;
R1-B0(纯增广)只在臂2/臂1w 都排不上时作为零改动替补。
