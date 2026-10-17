# SOTA 差距总表 001(ADR0025 + ADR0026 + ADR0027 口径)

生成:2026-10-17 深夜,SOTA冲刺3。我方数全部同题同协议实测(单遍,两次 MMAU 已证 repeat_identical);对照按 ADR0027 引用其公开发布数,口径差异逐格标注。

## 判定榜四榜 × 我方候选(D3P-600,767be09b)

| 榜 | D3P-600(候选) | 基座 E4B | 最强对照公开数 | 差距 | 备注 |
|---|---|---|---|---|---|
| JevBench 231 | 76.62%(macro 74.93) | 76.19% | **Jev-Omni-12B 86.15(micro 87.45,官方模卡)** | **−9.5** | 弱族集中:temporal_numeric 6.7% / ambiguous 28.6% / tradeoff 33.3% / long_policy 52.6%;简单族 100% |
| MMAU test-mini 1000 | 60.6% | **65.9%** | Jev-Omni-12B 63.10(micro,官方模卡) | 候选 −2.5;**基座 +2.8** | 基座部分受益榜的 A-偏置(见机制注);候选 −5.3 已解剖=位置先验漂移+2~3pt 真实退化;U 通道塌缩(校准层覆盖,模型卡披露) |
| MVBench 九任务 1800 | (实测中) | (实测中) | Jev-Omni-12B 53.09(**14 任务/2786 题,口径不同,仅参照**) | 待填 | 同协议 9 任务数为本表口径 |
| Decision Index 0.3(公共部分) | 未测 | 未测 | 未调研 | — | 27/44 子集已准入,输入已 staged;下窗口打分 |

排除项:DecisionBench Medium(教师卡引 87.57)= 两轮检索无公开数据集,私有评测,不可验证;OneJev test set / Valen Sokoban 集 = 自有私有题。对照无公开数:OneJev-4B(仅对比图)、OneJev-27B(仅对比图)、Valen(Sokoban 专用)。

## 结论(截至本表)

1. **SOTA 未达成**:JevBench −9.5 是主要缺口,根因=训练规模(对照全参 3 万-9.9 万题 vs 我方 600 样本 LoRA)。
2. **缺口集中可攻**:弱族合计 ~100 题,若结构化执行(CCD-for-decisions,实验进行中)拉到 80% → 总分 ~89-93,可越过教师发布数。预注册门:弱族 ≥80% 且全榜 ≥84。
3. **MMAU 已有可声明位**:基座(路由形态)65.9 > 教师公开 63.10;候选全模态挂载则回退,发布形态建议音频/文本路由基座(ADR0024 解禁),模型卡双形态如实并列。
4. U 通道塌缩、MMAU 位置先验、JevBench 长文本弱族均如实入模型卡。

证据:s4-public-bench-{mmau,mvbench}-*.json、s4-jevbench-d3p600-001.json、s4-mmau-regression-001/、teacher-mmau two-pass complete、Jev-Omni 官方模卡(fetched 2026-10-17)。
