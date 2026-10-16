# ADR4001(提案,worktree 暂名,main 整合时重编号):感知-决策解耦的 generate-then-decide 服务输入模式

状态:**proposed,2026-10-16,S4 创新切片产出;未实现、未采用。采用需用户批准。**
编号 4001 为 worktree 暂名,主树整合时按 ADR 序列重编。

## 背景与问题

video500 候选 74.4% vs 教师 94.2%(19.8pt 师生差),残余错误集中在同视频关系/事件存在;
oracle-001(reports/explore/s4-video-ensemble-oracle-001)证明该余量在概率级融合中取不出,
必须从输入维取。前沿调研(reports/explore/s4-frontier-research-001)将感知-决策解耦列为
可试点方向:冻结基座先生成结构化感知摘要,决策 LoRA 在 [问题+自生成摘要] 上做单 token 直答。

## 决策提案

服务与训练的决策输入契约允许 **模型自生成的感知摘要** 作为 prefix 文本段:

1. 感知 pass:冻结基座(无 LoRA)贪心生成 JSON 结构化摘要(对象/入出场时相/顺序/碰撞),
   官方 chat template、thinking 关闭;每视频一次、题目无关、可缓存(前沿报告 I7 同构收益)。
2. 决策 pass:决策 LoRA 在 [摘要(文本)+问题(+可选原始媒体)] 上单 token 读出;
   训练用同一基座产生的摘要(训推一体,无 exposure mismatch)。
3. summary-only 变体:决策 pass 不带视频 token——端侧/低成本服务形态,是本 ADR 的
   主要成本论据(决策 pass 输入 token 大幅缩短)。

## 与现行约束的冲突(须用户逐项决策)

- "训练/服务共用官方 chat template":摘要作为用户内容注入不违反消息结构,但**用户内容
  不再是纯原始输入**——需用户确认官方模板约束是否及于用户内容纯度(前沿报告 I7 同款问题)。
- 单步/延迟产品契约:感知 pass 引入一次额外生成(题目无关、按视频摊销);summary-only
  决策则省媒体 token。净延迟按实测,不预设结论。
- thinking-off:摘要 pass 是显式生成,与"决策单 token、thinking 关闭"并行不悖,但形式上
  多了生成步骤——是否视为"变相 thinking"须用户判定。

## 试点证据与门槛(已 CPU 准备,GPU 待 main)

- CPU 管线已实跑自测:`.scratch/s4/caption_probe.py`(提示+严格 JSON 校验,--self-test 过)、
  `.scratch/s4/build_b2_conditioned.py`(两变体条件化行,pydantic 过,fixture 测试过)。
- GPU 步骤与预注册门见 reports/explore/s4-b2-generate-then-decide-prep-001:
  100 TRAIN 视频 caption 抽样(valid≥80%,<50% 停)→ 240s 条件化切片 →
  video150≥121/150 或关系/存在子项≥+3pp 且 DEV50 不回退;采用前须本 ADR 获批。

## 预期增益依据(诚实)

- 上限锚:教师 94.2% 证明该信息量在更强感知下可得;摘要若能携带谓词级事实,关系/存在
  错误(50 个已定位)是直接受益面。增益未实测,不预设数字;前沿报告类比量级 +3–8pt(事件切片)。
- 主要风险:摘要幻觉污染同视频全部问题(valid 率+双采样一致性过滤是闸门);摘要质量
  依赖基座感知能力(基座 video 55.6%,摘要 pass 能力上限未知,试点第一测)。

## 成本

- 感知 pass:每视频一次生成(~512 token 输出);按视频缓存后摊销至全格问题。
- 决策 pass:summary-only 变体输入 token 显著缩短(移除视频 token);
  summary+video 变体增加摘要 token。训练侧增量:条件化数据重训决策 LoRA。
- 无基座/引擎/head 改动;LoRA 仍唯一可训练件。

## 需用户决策点

1. 是否接受"用户内容含自生成摘要"的输入契约解释。
2. 是否接受感知 pass 的额外生成步骤(及端侧档是否只跑 summary-only)。
3. 试点过门后是否进入正式采用流程(完整重训+全模态复核)。
