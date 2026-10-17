# 交接文档:SOTA冲刺3 夜班(2026-10-17 晚 → 18 凌晨)

> 本文件是 2026-10-17 夜间窗口的完整快照。**接续入口仍是 `plan/STATUS.md`**;本文用于不误导新 session 的全量对账。所有数字均为盒上核验产物,无一条转述。

## 一、本窗口干了什么(时序)

1. **盘点除垢**:21 个旧 handoff 归档 `plan/archive/`;STATUS 三次重写;盒上过时脚本/残片入 `attic-*` 目录;AGENTS 榜单口径更新;转换器规范版入仓(替换 191 行旧版);9+ commits。
2. **定选模型**:时序核验——**D3P-600(767be09b)= 最后一个完成训练+评测的产品模型**(1500 只建了 teacher-cache 未开训;combo3 是已证伪实验)。`lite-omnijev-release-002/` 已按它拼装(候选 + 绑定校准 + 封存证据,SHA256SUMS 齐)。
3. **榜单实测**(ADR0025/0026/0027 口径):
   - MMAU 1000:基座 65.9 / D3P-600 60.6 / 教师同协议 64.6(修复加载后;教师自家协议旧数 62.0 仅参考)
   - JevBench 231:D3P-600 76.62(macro 74.93;与父适配器 e465 完全一致,ADR0025 附录数字经实测确认)
   - MVBench 1800:实测收尾中(见"在跑")
   - 教师官方模卡发布数已抓取:JevBench 86.15/87.45、MMAU 63.10、MVBench 53.09(14任务/2786)、DecisionBench Medium 87.57
4. **口径修订**:ADR0026(自建榜废除,只用四公开榜)、ADR0027(对照引用公开数,不为对照烧卡;我方只测自己)
5. **创新实验(CCCD-for-decisions)按预注册门证伪**:探针全榜 74.03 < 直答 76.62,弱族 40.2% << 门 80%;**oracle 路由上界 80.5% < 门 84%**,数学封死修订。机制:同模型分解带同样盲区(与 CLEVRER RL 线同构)。代码已并入 main(scripts/ccd_decisions_probe.py),证据 `s4-ccd-dec-probe-001`。
6. **MMAU 回归解剖**(-5.3):位置先验漂移为主(榜偏 A,基座吃偏置 .777 vs 候选 .562)+ 2-3pt 真实排序退化;U 通道塌缩(pU→e-6,发布校准层覆盖,模型卡须披露);音频路由→基座为最优发布形态。证据 `s4-mmau-regression-001`。
7. **Bet B 规模化蒸馏已发射**(用户批准):`joint-train-v3.jsonl` 16,177 全 gold 决策样本(text 10,750 + image 4,167 + screenshot 1,260),init=D3P-600,rank64/α128,1800 步/accum 10/lr 1e-4,checkpoint 每 100;**预注册门:终点自动跑 JevBench ≥86;约束 MMAU ≥63.10 不回退**。用户令:时刻监控,基本收敛即停。

## 二、在跑(接手时先查)

- `pbm-d3p`/`pbm-base`:MVBench 双分(报告 `reports/s4-public-bench-mvbench-{d3p600,base}-001.json`)
- `joint-v3`:Bet B 训练(log `/root/autodl-tmp/rl-logs/s2joint-v3.log`;末行 `JOINTV3_JEVBENCH:` 即门判定;`S2JOINT_V3_DONE` 完成)
- 监控规则:**基本收敛即停**(checkpoint eval 连续无提升);JevBench 过 86 → 候选升级为 joint-v3 adapter,重拼 release-002;未过 → D3P-600 维持,数字照实

## 三、差距总表(截至夜间,证据 `reports/s4-gap-table-001.md`)

| 榜 | D3P-600 | 基座 | 最强对照公开数 | 缺口 |
|---|---|---|---|---|
| JevBench 231 | 76.62 | 76.19 | 教师 86.15(micro 87.45) | **−9.5(Bet B 主攻)** |
| MMAU 1000 | 60.6 | **65.9** | 教师 63.10 | 基座形态 +2.8 ✓ |
| MVBench 1800 | 实测中 | 实测中 | 教师 53.09(14任务,口径不同仅参照) | 落地后填 |
| Decision Index | 未测 | 未测 | 未调研 | 下窗口打分(输入已 staged) |

## 四、证伪清单(累计 9 条,勿重试)

同骨干集成 / COMBO·SAS 混合 / 自适应取帧(CLEVRER)/ 温度转移 / 纯 gold RL / K3 自一致 / v3 并集 / 置信度仲裁 / **CCD-for-decisions(判定榜)**。另:RL 级联(CLEVRER 主场 118<123 已闭)。

## 五、资产地图

**盒(ssh seeta,凭据不入仓)**:
- 模型:`/root/autodl-tmp/lite-omnijev/artifacts/models/`(基座 ee0ef60、OneJev-4B/27B、Jev-Omni、Qwen3.5-2B、Valen-adapter)
- 候选:`s1-text-lora-001/artifacts/s6-d3p-student-600-001/run/adapter`(767be09b);release-002:`…/lite-omnijev-release-002/`
- 评测树:`/root/autodl-tmp/s5-calibration-code-fa83acc`(含本轮 s1_dev_eval 架构感知补丁——Unified checkpoint 必须走 Gemma4UnifiedForConditionalGeneration,否则静默随机初始化双塔)
- 样本:`…/public-bench-samples-001/`(MMAU/MVBench 契约+media,VERIFY-OK;过时中间产物在 attic-intermediate 子目录)
- Bet B:训练数据 `s2-joint-pool-v3-001/joint-train-v3.jsonl`;发射器 `/root/autodl-tmp/launch-s2joint-v3.sh`
- 对照已发布数:Jev-Omni 官方模卡(HF fetched 2026-10-17,数字见差距表)

**本地仓(main 分支)**:scripts/{public_bench_samples,comparator_eval,ccd_decisions_probe}.py、comparator_families/、train/template.py(payload_dict 提取)、tests/test_comparator_eval.py、ADR0026/0027、reports/s4-gap-table-001.md、issues/01

## 六、已知债务/风险

- issues/01:clevrer_fourtype_eval mypy 8 错(存量)
- MVBench 口径:我方 9 任务/1800 vs 教师卡 14 任务/2786,引用须双列
- 教师 MVBench 同协议数未跑(按 ADR0027 省卡,引用其发布数+口径标注)
- Valen:权重在盒(1.16GB RLCD checkpoint + Qwen3.5-2B 底座),加载码未移植,矩阵记 conversion pending
- release-002 的 gateway/ 为空,serving 复验 pending
- U 通道塌缩:候选在音频输入上 pU≈0,发布校准层覆盖封存池,公开输入上的 p(U) 可信度未验

## 七、下一窗口队列

1. joint-v3 收敛监控 + JevBench 验门 → 过 86 则换候选重拼 release-003 + 模型卡
2. MVBench 差距形态 → 决定主动感知探针是否值得(CLEVRER 上已证伪,榜未验)
3. Decision Index 27 子集打分(官方 runner 在 `s4-index-work-001/source`)
4. JevBench 对照格补全(引公开数;缺格标"无公开数")
5. 训练侧:答案顺序均衡/位置增广(MMAU 残余 2-3pt 的修法)
