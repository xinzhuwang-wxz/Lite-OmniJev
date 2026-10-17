# SOTA 宏观冲刺方案 001(接手唯一依赖;2026-10-17 19:25 CST,SOTA冲刺2)

目标(用户定):在本产品相关的公开榜单上超过**所有**条目(含教师 OneJev-27B),产品输出保持 Jev 式(Choice/Score/yes-no,无 U);内部管线全面解禁(ADR0024 修订二)。主战场 = CLEVRER 官方四类题(descriptive / explanatory / predictive / counterfactual)。其他模态榜单(MVBench、MMAU 等)见副手的 `reports/explore/s4-leaderboard-sota-map-001.md`(编写中)。

## 1. 证据全景
- 交付候选 D3P-600(adapter 767be09b)封存校准完成;发布目录 lite-omnijev-release-001。→ plan/STATUS.md、reports/ 下 s5 封存汇总。
- dev150(全是 CLEVRER descriptive exist yes/no)同题:教师 141、OneJev-4B 133、D3P 123;三者并集 149。→ `s4-sota-takeover-diagnosis-001.md`
- 执行器 v4 在完美场景 oracle 94.67%;真实抽取级联 114(旧)/ 117(RL 部分 57 题);执行器覆盖直答胜 4 负 10。→ `s4-ns-quality-gate-001.json`、takeover 诊断
- RL 形式化器 60 视频训练完成,探针 F1 .362→.438→.459(ckpt20/40/60),final 在 `artifacts/s6-formalizer-rl-main/final`。
- 置信度仲裁(分支 A)在部分数据上选择"永不覆盖",无增益(待全 150 复核)。→ 盒上 `.scratch/s4/arbitrate.py`
- 证伪路线:同骨干集成、COMBO/SAS-full 混合、自适应取帧、温度跨候选转移、纯 gold RL、K3 自一致、v3 三信号并集。→ reports/explore/ 各报告
- 公开榜(二手,待副手核实):VRDP 约 D94/E96/P87/C75;后续工作在 P 93、C 78 超过它;PhysMind 报告整体 per-option 87.22 / per-question 72.55。度量口径混杂(per-option vs per-question)。

## 2. 差距分解
- dev150 对教师 18 题:enter 族 15(n=106,教师 101 / D3P 86)、stationary 2(15/13)、moving 0(17/17)、exit 0(7/7)、static 1。
- 谓词接地(RL 前 150 场景):颜色 .97、形状 .99、材质 .85(召回 .74)、enterers .64(精度 .50)、exiters .09、moving .49(召回 .36)、collisions .26。
- 四类题差距:待 q4type 基线(D3P debug40)出数;explanatory/predictive/counterfactual 目前没有任何实测。

## 3. 路线图
**主线 CCD(组合校准决策)**,设计 `s4-ccd-paradigm-001.md`:逐帧静态原子(存在/计数/行列位置)由产品模型校准回答 → 轨迹/事件 → 自写执行器 + 物理推演 → Jev 式输出;组合分布回蒸馏直答。
1. G1 原子质量(零训练):`queue-g1-atoms.sh` 已排队;门 = presence AUC ≥ .97、行列准确率 ≥ .90。≈15 min GPU。
2. PTR(生成式逐帧列物体)探针:进行中,`artifacts/s4-ptr-probe-001/`;门 = enter 族 ≥ 教师同题 −2。
3. 执行器 + 物理:fable 编写 `eval/ccd_world.py`、`eval/ccd_executor.py`、`eval/ccd_physics.py`、`scripts/ccd_oracle_eval.py`;门 = 真值世界 D ≥ 99、E ≥ 98、P ≥ 90、C ≥ 85(per-option)。CPU。
4. 原子训练(G1 不过时):`scripts/ccd_atom_bank.py --pool train` 生成 → LoRA 小批 2k 原子 → AUC 门 → 扩量。≈3 GPU h。
5. 端到端 CCD:原子 → World → 执行器,在 dev2-debug 调试,dev2-judge 只看汇总;过门后在 CLEVRER validation 私有聚合作业出正式数。
6. 回蒸馏:CCD 组合分布当软目标训直答快路径(D3P 管线复用)。

**备用线**:A 置信度仲裁(零 GPU);D D3P-1500 定向 enter 族(≈25 GPU h 教师缓存);E 目标中心 token(需 ADR)。详见 `s4-contingency-tree-001.md`。

## 4. 弹药盘点(已核实 ✓ / 未核实 ?)
| item | 状态 | 位置 |
|---|---|---|
| 执行器 v4(仅 descriptive exist 子集) | ✓ 已就绪 | 盒 `.scratch/s4/ns_quality_gate.py`(execute) |
| RL 形式化器 final + ckpt20/40/60 | ✓ | `artifacts/s6-formalizer-rl-main/` |
| formalizer-ds | ✓ 存在(规模未复核) | `artifacts/s4-ensemble-oracle-001/formalizer-ds` |
| K3 场景:便宜版 150 + RL 版 41+抢救中 | ✓ 进行中 | `s4-ns-cascade-001/scenes-k3-rl.jsonl`、`s4-ns-cascade-002/rescue/` |
| PTR 探针 | ✓ 运行中 | screen `ptr`,`artifacts/s4-ptr-probe-001/` |
| 四类题评测集 | ✓ | `artifacts/clevrer-fourtype-dev2-001/`(debug 300 / judge 900 / debug40) |
| 原子银行 + G1 样本 | ✓ | `artifacts/ccd-atoms-dev-001/`;代码 `scripts/ccd_atom_bank.py` |
| 训练注释(逐帧真值) | ✓ | `/root/autodl-tmp/clevrer-train-ann-001` |
| validation(封存来源) | ✓ 已解压,只供私有聚合 | `/root/autodl-tmp/clevrer-val-001` |
| 评测器 | ✓ | `s5-calibration-code-fa83acc` 的 `scripts.s1_dev_eval` |
| 新执行器 + 物理 | 需准备(fable 进行中) | `eval/ccd_*.py` |
| 原子训练配置 | 需准备 | 复用 D3P/LoRA 训练入口 |

## 5. 决策树
| 信号 | 读数 | 动作 | 下一脚本 |
|---|---|---|---|
| RL 级联终判 `final/gate.json` | ≥ 141 | 罕见;复核 dev2-judge 后转 CCD 整合 | — |
| | 124–140 | 跑分支 A 仲裁,记录;主线不变 | `arbitrate.py QUESTIONS final/merged-rl-150.jsonl D3P_RECORDS` |
| | ≤ 123 | 记为"形式化器级联不及直答",停 RL 线 | — |
| PTR | enter 族 ≥ 教师 −2 | 化归成立 → 主线 5 优先 enter/exit | CCD 端到端 |
| | 差 3–10 | 原子训练(主线 4) | `ccd_atom_bank.py --pool train` |
| | 更差 | 查感知(帧分辨率/提示);G1 定夺 | G1 报告 |
| q4type 基线 | 每类与榜首差距 | 差距最大的类优先;P/C 依赖物理 | `ccd_oracle_eval.py` |
| G1 | 过门 | 跳过原子训练 | 主线 5 |
| | 不过 | 原子训练 | 主线 4 |

## 6. 限制与风险
- 卡:用户要求不空转、省卡时;开关卡/续费由用户决定。
- ssh 抖动:长作业一律 `screen -dmS`,短命令轮询;启动脚本禁止 `rm` 输出(本窗口事故);`timeout` 按实测速率设(sampler ≈2.5 min/视频/分片)。
- 封存:CLEVRER validation 内容不得打印给 agent(本窗口已有一次披露,见 takeover 诊断);dev2-judge 只看汇总;dev150 已被反复使用,不再作判定。
- 许可:官方 CLEVRER 代码无许可,不复用;数据 CC0。
- 显存:H20 97 GB;每个 E4B bf16 进程 ≈16 GB。

## 7. 接手须知
- 接续入口:plan/STATUS.md 顶部"19:40 更新";本文件;`s4-contingency-tree-001.md`;`s4-ccd-paradigm-001.md`。
- SSH:`~/.ssh/config` 的 `seeta` 别名(密钥认证,凭据不入仓)。Python env:`/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`;模型:`/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef60…`;媒体:`/root/autodl-tmp/s3-media-candidates-001/video/media`。
- 盒上 screen:`rescue0-2`(fd 抢救,勿杀)、`autogate`(RL 终判)、`ptr`、`q4type`、`qg1`、`unzipval`。
- 家族-池映射:`/root/autodl-tmp/clevrer-family-pools.txt`(train 8299、dev 500;四类评测用其外的 1201 个 train 视频)。
- 关键结论:差距集中在 enter 族;静态感知强、事件弱;覆盖式级联输;同源集成无红利。
