# SOTA 冲刺预注册决策树 001(2026-10-17,SOTA冲刺2)

前提更新:RL 形式化器主训练已完成(F1 .36→.46,三连 continue),不再有"RL 训练 gate";待出的是 **RL 级联终判**(`s4-ns-cascade-002/final/gate.json`,autogate 自动产出)。SOTA 口径已改为 CLEVRER 官方四类题全部超过榜单所有条目(含教师),dev150(仅 exist)只作诊断。

## 输入信号(三个,均自动产出)
- S1 RL 级联终判:cascade_correct /150 与 executor resolved/correct。
- S2 PTR 探针:enter/exit 族上 PTR 准确率 vs D3P vs 教师(同题)。
- S3 D3P 四类题基线:`clevrer-fourtype-dev2-001/d3p600-debug40-report.json`(按类型/子类型)。

## 分支(门预注册,不移动)
| 分支 | 触发 | 动作 | 门 | 估时 / 卡时 |
|---|---|---|---|---|
| A 置信度仲裁(零 GPU) | S1 级联 < 123 且执行器-直答并集 ≥ 直答 +8 | CPU:以 D3P 置信度/K3 执行一致性为特征,在 dev150 前 75 题定阈值、后 75 题验证 | 验证半 ≥ 直答 +3 题,否则弃 | 1 h / 0 |
| B PTR→CCD 化归 | S2:PTR 在 enter 族 ≥ 教师同题 −2 | 原子银行 + 执行器(fable 两模块)接入;生成式逐帧列物体 → 校准原子读出 | dev2-judge enter/exit 子集 ≥ 教师;再扩到全 descriptive | 1 天 / 4–6 GPU h |
| B' 原子训练 | S2:PTR 接近但不足(差 3–10 题) | 原子银行训练 LoRA(presence/count/position),先 2k 原子小批,AUC 门 .97 | 小批后原子 AUC ≥ .97 再扩量 | 0.5 天 / 3 GPU h |
| C 物理推演 | S3 显示 predictive/counterfactual 远低于榜首 | 执行器 + ccd_physics(fable)用原子轨迹推演 | 真值世界 oracle:predictive per-option ≥ 90、counterfactual ≥ 85;不过则改进物理,不上模型 | 1–2 天 / 2 GPU h |
| D 定向蒸馏 | B/B' 均不过 | D3P 扩到 enter 族富集的 1500 题(切片/runner 已备) | dev2-judge enter 族 +5pt | 1 天 / ~25 GPU h(教师缓存) |
| E 目标中心 token | 以上全不过 | 需架构变更 ADR(4002 轨道) | 先写 ADR + 小探针 | 2–4 天 |

## 不做(已证伪或被取代)
- RL 继续训练至"七谓词 +0.2":级联整体被"执行器覆盖直答"的结构性损失主导(部分结果胜 4 负 10),提高 F1 不解决仲裁问题;除非 A 证明仲裁有效再评估。
- 执行器 v5 单独修 EXIT_REL:被 fable 新执行器(四类题全算子)取代。
- 同骨干集成、COMBO 混合、自适应取帧、温度转移、纯 gold RL、K3 自一致:见既有证伪证据。

## GPU 排期
采样结束 → q4type(D3P 四类题基线 ~17 min)与 autogate(CPU)同时 → PTR 收尾 → 按分支表排下一 GPU 作业;任何时刻 GPU 空闲 >10 min 视为排期失败。
