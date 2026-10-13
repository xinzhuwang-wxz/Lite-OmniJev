# ADR-0006：以可复现公开视频集建立教师基线

状态：用户2026-10-14明确选择“采用固定9任务真实基线（推荐）”。依据 `docs/research/teacher-mvbench-decision-options.md` 及固定源审计报告；原设计HTML保留。

## 决定与边界

仅替换S0教师MVBench“发布53.10±2分复现”要求：固定官方MVBench revision `230a2d4fac8900333c61754641c7a13e069ac9c6` 的9任务全部1800原行（STAR3、Perception4、FunQA1、STA1），建立真实、独立、两遍一致的教师基线。不按结果重挑集、不猜14任务排除规则。发布53.10/53.09只标未复现引用，不能与新集合当同集比较。

官方Jev-Omni模型/head、官方template、概率映射和已允许研究来源不变。MMAU1000题63.10±2分、S0其余门槛、S3同题教师比≥90%、S4同题最强允许教师比≥97%全部不变。公开九任务不能替代各模态开发集或封存集；教师真值优先规则不变。

## 可验证契约

- 原任务/row顺序、JSON及原媒体hash、原start/end、16帧RGB内容/索引、解码器和代码版本冻结；预注册后不得因成绩或失败删行。
- STAR/STA有界题按官方16-bin中点公式，start=max(0,round(start*fps))，end=min(round(end*fps),N-1)；3条越界保留原题和分母、夹到最后帧。无界题取完整视频；teacher/student使用相同RGB输入资产，各用官方processor。
- 九任务分别correct/200，macro等权、micro总correct/1800；任何缺媒体、解码失败、非有限概率、hash变化、漏题令批次不通过。两遍逐题预测/规定概率一致，不仅总分一致。
- 开卡前全部1800行做真实CPU clip/16帧有效性及RGB hash验收。缓存实现受30GB派生预算约束，可采用完整无损PNG或原视频+逐帧hash manifest/有界缓存；后者GPU运行时须预取重解码、逐帧hash匹配后才送两模型，不能暗缩帧/分辨率。
- 开卡仍由用户操作。先≤32预注册smoke，再完整两遍；估算超批准卡时、OOM、无效输出或不一致则可恢复停止留证，不自动扩预算。

机器门槛从 `teacher_mvbench_delta_abs` 换为 `teacher_mvbench_baseline_verified=1`；该数值必须由完整协议/hash/行数/真实forward/重复报告支撑，手填1或仅CPU准备不算通过。旧±2不再是本项要求，不允许其他门槛跟随豁免。

## 后果

可按现有合法公开媒体推进，无须等待作者未知的14/2786清单。外部覆盖收窄且不可直接比较发布值，明确记录该限制。Moments/SSv2/TVQA/Matterport/NTU等额外授权只在后续明确扩范围时触发，非本方案首发前置；没有给出这些数据的商用训练许可。
