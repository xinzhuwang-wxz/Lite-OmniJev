# CCD 执行器+物理 真值世界 oracle 001(2026-10-17)

代码:`eval/ccd_world.py`、`eval/ccd_executor.py`、`eval/ccd_physics.py`、`scripts/ccd_oracle_eval.py`(82d00ad,自写;官方 CLEVRER 代码无许可未复用)。数据:CLEVRER train 场景 DEV 0–299 / CHECK 9000–9299(真值注释,非模型感知)。完整表:本地忽略目录 `artifacts/ccd-tools/{dev,check}-final.json`。

| 类别 | DEV | CHECK |
|---|---|---|
| descriptive | 100.0 (3300) | 100.0 (3298) |
| explanatory per-option / per-question | 100.0 / 100.0 | 100.0 / 100.0 |
| predictive per-option / per-question | 99.05 / 98.10 | 97.33 / 94.67 |
| counterfactual per-option / per-question | 98.04 / 93.45 | 97.11 / 90.30 |

- 物理 ~7 ms/视频;残余错误=从未发生碰撞的对象对的擦边碰撞(反事实)、漏预测碰撞(预测)。
- 披露:CHECK 首跑 explanatory 3 错、descriptive 1 错后修了同帧碰撞排序/祖先规则(DEV 不变),故 CHECK 的 D/E 100% 非完全未触碰;物理参数未在 CHECK 上调。
- 含义:**推理层在完美感知下已高于公开报告的各类榜首**(二手数,待副手核实);SOTA 胜负完全取决于感知(逐帧存在/位置 → 轨迹 → 碰撞)。下一步:感知敏感度曲线(fable 进行中)与 G1 原子门。

## G1 原子门(零训练,D3P-600,30 个 dev 池视频的 2000 原子,20:13)
- presence 85.6%(AUC .949,门 .97 未过)、count 65.0%、column 36.8%、row 30.8%(6 档)。
- 结论:VLM 不作定位工具(与敏感度曲线一致);CLEVRER 按 ADR0025 不是判定榜,原子训练暂不投 GPU,CCD 方法保留到判定榜上按实测差距再选用。
