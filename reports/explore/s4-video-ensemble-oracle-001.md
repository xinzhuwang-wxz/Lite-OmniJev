# s4-video-ensemble-oracle-001 — 离线融合 oracle(零 GPU)

输入:两份已完成 GPU 作业的逐题 records(远端只读拉取,SHA 全部钉住,见 JSON inputs)。
候选=adapter e46550143b28(step0100),父=adapter f2dbb83b(课程初始化来源);
纯基座=冻结 E4B 无 adapter。所有输入同 data_sha256、逐题 id/target 对齐核验通过。

## video500(候选 74.4% vs 父 70.8%)

| 指标 | 值 |
|---|---|
| 平均概率集成 | 0.7480 (+0.40pp) |
| 几何平均集成 | 0.7480 |
| 最优加权集成 | 0.7540 @ w_left=0.6 |
| 任一对即正确 oracle | 0.8220 (+7.80pp) |
| 分歧率 | 0.1920 |
| 分桶 both/cand-only/parent-only/neither | 315/57/39/89 |
| 对比解码(精确 logits) | lambda=0.25:0.7260; lambda=0.5:0.7280; lambda=1:0.7020; lambda=1.5:0.5840; lambda=2:0.5200; lambda=3:0.4380 |

## 判定

- oracle 增益 **+7.80pp** ≥ +5pp 门槛 → 按预注册规则**不终止**集成路线;
  但全部可实现规则(均值/几何/加权≤0.754、任意 λ 对比解码全部为负)都不能兑现 oracle 余量。
- 对比解码在**全量 video500 精确逐题 logits** 上 λ=0.5/1.0/2.0 全部负增益
  (λ=0.5:0.7280、λ=1:0.7020、λ=2:0.5200);
  计划中的 video150+DEV50 GPU 探针只会复算同一批确定性 logits,**无需再跑**。
- video500 候选 vs 纯基座(55.6%):oracle 0.8480(+10.40pp)、
  均值集成 0.6000(-14.40pp)、对比解码全负 → 纯基座做 amateur 同样不可用。
- JevBench231:均值集成 0.7489(-1.73pp)为负,
  oracle 0.8182(+5.19pp);互补结构 97% 集中于 ambiguous n=7(见 /tmp/innov-p0b)。
- 帧采样边缘化侧信号(旧 CE0 adapter,DEV50 uniform vs motion):marginal 0.6200 vs 单采样 0.6000,
  分歧率仅 4.00%、oracle 上限 0.6200(≤+2 题)→ 帧维互补远小于模型对互补。

## 对颠覆轨的指向

oracle 余量真实存在(96 个分歧题、89 题双错)但概率级融合取不出来 → 增益须来自**输入/结构维**
(B1 程序格一致性、B2 感知-决策解耦、ADR 目标中心输入 token),而不是决策后融合。
