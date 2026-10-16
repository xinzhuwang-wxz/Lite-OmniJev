# S4 前沿深度调研 001:单步多模态决策的范式级空白

- 日期:2026-10-16 | 类型:纯研究(未跑任何 GPU/CPU 实验;除文献检索)
- 范围:2024–2026 单步多模态决策/视频因果推理/RL-for-VLM/测试时计算/表征注入文献 → 10 个候选范式,按"前人没用过 × 命中实测失败模式 × 期望收益"排序(用户澄清:不做 RL 偏好,全维度平等)
- 数据来源:WebSearch(2024–2026 一手链接,二手摘要标注);所有 SOTA 数字均注明出处与置信级别;**内部实测数字引自任务书,未复测**

## 0. TL;DR

- 榜单格局:CLEVRER 类已被神经符号/仿真路线打到 ~96%(CRCG),前沿 VLM 92–95%;我们的差距(74.4%)集中在**谓词组合层**(in/out 22 错、motion 15、start/end 13),而 SOTA 恰好用"程序分解+世界状态执行"解决这一层——但**从未有人把世界状态执行结果作为训练信号回灌给单 token 直答的冻结 VLM**(他们都在推理时执行,我们是训练时蒸馏)。
- 最大已知天花板:video500 教师-学生差距 19.8pt(94.2 vs 74.4)——纯粹的蒸馏效率问题,不需要新范式,需要" deliberation 后分布"蒸馏。
- **建议前 3**:I1 世界状态因子化目标(数据+目标构造)、I5 deliberation 蒸馏先验(标签构造)、I2 孪生分歧 U 监督(校准/答案性)。三者全部契约合规、CPU 可完成大部分验证、≤45min GPU 试点。
- **明确不做**:纯 outcome-reward RLVR(数学等价 CE,内部已证;文献亦证熵塌缩)、slot/视觉 tokenizer 替换(违冻结契约且有先例 SlotTok)、注意力操控(违"不改推理引擎")、新增 PRM/verifier 网络(VisualPRM 先例+违契约)。

## 1. 榜单格局:单步多模态决策 SOTA 在做什么

**CLEVRER/视频因果推理**
- 原始基准设计就是"神经管描述性、符号管因果"(NS-DR):[CLEVRER, Yi et al., ICLR 2020](https://arxiv.org/abs/1910.01442)。
- 当前最强是**推理时仿真**路线:CRCG "Think before You Simulate"(2025-06)对反事实题选择性调用物理仿真,搜索摘要称 CLEVRER **96.1%**([arXiv 2506.10753](https://arxiv.org/abs/2506.10753);二手数字,外部引用前需核对原文);前沿通用 VLM(Claude/GPT/DeepSeek 级)约 92.7–95.3%(同搜索摘要),VideoThinker-R1 去偏后 79.1%。CLEVRER 本体趋于饱和,继任者 CLEVRER-Humans([arXiv 2310.03635](https://arxiv.org/abs/2310.03635))、Physion/PhyBlock 更难。
- 对我们的含义:**20pt 差距不是"模型不够大",是"谓词组合没有人管"**——符号路线的增益全部来自把问题分解为谓词并对着世界状态执行。我们实测错误(in/out/motion/start-end)正是未分解的谓词。

**MMAU/MMMU/MMBench 类**
- MMAU(音频,[ICLR 2025 Spotlight](https://sakshi113.github.io/mmau_homepage)):Gemini 1.5 仅 66.15%,Qwen2-Audio 55.4%;MMAU-Pro([AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39430/43391)):Gemini 2.5 Flash 59.2%。头部共性:**没有奇技,靠原生全模态预训练+长上下文**;分歧在"是否显式融合 vs 端到端"。我们 audio 79.8% 已在同代头部区间,crossmodal 71.0% 是短板——文献确认 MLLM 普遍存在**单模态捷径/模态偏差**([MLLMs are Deeply Affected by Modality Bias](https://arxiv.org/abs/2505.18657))。

## 2. RL for VLM 前沿:真实收益与失败模式

- [VLM-R1](https://arxiv.org/abs/2504.07615)(2025-04):规则奖励 R1 式 RL 在检测/grounding 上比 SFT 更稳、更可泛化。
- [Perception-R1](https://neurips.cc/virtual/2025/poster/119361)(NeurIPS 2025):论点=多模态推理瓶颈在**感知**而非推理,GRPO+规则奖励打感知任务;其 ICLR 2026 后续用 judging LLM 做标注一致性。**注意:Perception-R1 解决的是 grounding/检测类感知,不是我们的关系谓词/存在性谓词**。
- 失败模式明确:RLVR/GRPO **熵塌缩**([Understanding and Preventing Entropy Collapse in RLVR](https://arxiv.org/html/2605.11491v1);verl 文档同载),且有研究指出根因是**二元 outcome reward 本身**而非算法。FAST-GRPO 等在做效率修复。
- 对我们的含义:决策 token 上的 outcome-RL 我们已内部证明数学等价 CE(已停);**RL 只有在奖励携带 outcome 之外的信息(谓词级/过程级)时才有增量**——这直接催生 I1 的奖励变体,但 I1 主体是数据构造,不依赖 RL。

## 3. 测试时计算:单步决策上的收益边界

- [Perception-Time Scaling, PTS](https://openreview.net/forum?id=WGIcXH9rk9)(ICLR 2026):token-rich 感知+把复杂视觉操作分解为简单操作采样——**感知侧 TTS 是新方向但都走 CoT/多 token 路线**。
- [Resolving the Grounding Paradox in Thinking with Images](https://arxiv.org/html/2604.11025v1)(2026-04):结论=感知不确定性下 test-time 感知扩展是有前途方向。
- TTS 综述(ACL Findings 2026)确认:**感知任务上的 TTS 是空白区**(现有 TTS 集中在生成与推理)。
- MindJourney(NeurIPS 2025,VLM+可控世界模型做 3D 空间 TTS)、UniT(Meta,统一多模态 CoT-TTS)均为重路线。
- **检索不到先例**:"多视角/多裁剪分布平均+学习视角权重→单 token 读出"(见 §5-I3 检索记录)。PTS 分解的是操作并生成 token;我们把边缘化做成**纯分布算术,零生成 token**,与 thinking-off 契约兼容。

## 4. 架构侧:冻结骨干上的合规注入

- **对象中心 token**:[Slot-MLLM/SlotTok](https://arxiv.org/abs/2505.17726)(对象中心 tokenizer 注入冻结 MLLM,图像)、Slot-VLM(NeurIPS 2024,object-event slots 对齐冻结 LLM,视频;检索摘要提及,未给直接链接,引用前需检索确认)。**有先例且要换视觉 tokenizer→违我们冻结契约,不建议**(列入 ADR 清单仅作记录)。
- **视觉 prompt/overlay**:[Set-of-Mark](https://arxiv.org/abs/2310.11441)(2023,图像分割编号 overlay,zero-shot 超 fully-tuned SOTA)、Graph-of-Mark(AAAI,像素级场景图 overlay)、Motion-as-Prompt([2026-08](https://arxiv.org/html/2608.11655v1),视频运动 overlay,training-free)、Learning-to-Localize(CVPR 2024,检索摘要提及,无直接链接)。**结论:overlay 系先例充分,单独用不算创新**;但"决策 LoRA 训练时以世界状态生成的结构化 overlay 作为输入侧 scaffold"仍是空白(并入 I7/I9 讨论)。
- **工具增强感知探针**:检测/分割器在推理链里调用是 agent 路线,与我们单步/单卡/端侧契约冲突,不展开。
- **选项顺序偏差**(校准侧):[Pezeshkpour & Hruschka](https://arxiv.org/abs/2308.11483)(顺序可致 13–75% 差)、Zheng et al. "LLMs Are Not Robust Multiple Choice Selectors"(ICLR,检索摘要提及,未见直接链接)、[EMBER/Permutation-Equivariant MCQA](https://ojs.aaai.org/index.php/AAAI/article/view/40401/44362)(AAAI 2026)、Liusie et al. "General Permutation Debiasing"(ACL Findings 2024,teacher-student 置换一致性,检索摘要提及)。**一致性训练有先例→相关 idea 降级为增量改进**。
- **弃权/答案性**:[R-Tuning](https://arxiv.org/abs/2311.09677)(NAACL 2024 杰出论文,自对齐式"I don't know")、[AbstentionBench](https://arxiv.org/abs/2506.09038)(NeurIPS 2025,推理模型弃权更差)、"Know Your Limits" 弃权综述(TACL 2025,检索摘要提及,未见直接链接)、CVPR embodied 弃权("When Robots Should Say I Don't Know")。**均未用"同权重未适配基座"作为答案性 oracle**(见 I2)。
- **过程奖励**:[VisualPRM](https://internvl.github.io/blog/2025-03-13-VisualPRM)(InternVL 8B,步骤级验证器)、[StructReward](https://arxiv.org/html/2608.08326v1)(结构化过程奖励,免训练验证器,但结构来自 LLM 而非仿真器)、多模态数学 PRM(NeurIPS 2025)。**全部需要 CoT 步骤可验证;单 token 直答没有步骤可验——没人做过"世界状态执行谓词"版**(见 I1)。

## 5. 候选范式(10 个,按最终排序)

> 通用约定:试点 GPU 预算 ≤45min(单卡);种子固定;dev 切片+JevBench231 mini 双门槛;所有训练均冻结 E4B 主干+决策 LoRA+官方模板+thinking off,除非单列说明。

---

### I1. 世界状态因子化目标(World-State Factorized Targets, WSFT)— 排名 1

- **一句话范式**:用仿真器世界状态(CLEVRER 类物体轨迹/碰撞标注)执行 B1 程序格的谓词 DAG,把 one-hot 训练目标替换为"谓词因子化分布"——gold 选项 1−ε,共享 k/n 谓词的近失选项按谓词重合度获得软质量,谓词执行不确定时给 U 质量。
- **Novelty claim**:PRM 系(VisualPRM/StructReward)验证的是 CoT 步骤,我们是**无 CoT、训练时、目标分布级**的世界状态回灌;神经符号系(NS-DR/VRDP/CRCG)在**推理时**执行程序得答案,我们只借其执行器造标签,模型本体保持单 token 直答。**检索词**:`"process reward model" multimodal step-level verification video QA 2025 scene graph simulator`(零直接命中)、`CLEVRER scene graph annotations program execution partial credit reward training neural video QA`(零直接命中,仅 VeriGraph 机器人规划等无关项)。差异最近邻=[StructReward](https://arxiv.org/html/2608.08326v1)(LLM 结构奖励,非仿真器执行,验证 CoT)、[CRCG](https://arxiv.org/abs/2506.10753)(推理时仿真)。
- **命中失败模式**:空间 in/out 22 错、事件运动 15、start/end 存在性 13(全部=谓词组合失败);附带改善 U(谓词不可判定→U 有监督来源)。
- **增益机制与量级**:one-hot"悬崖"迫使模型在近失选项间硬分类,软目标教谓词级可分性;软标签蒸馏文献量级 2–5pt,关系切片估 **+4–8pt**(依据:同尺寸前沿 VLM 在 CLEVRER 92–95 vs 我们 74,差距主体在谓词层;符号路线靠谓词执行拿满该层)。诚实边界:仅覆盖有世界状态标注的仿真族,自然视频需迁移。
- **契约关系**:完全合规(数据/目标构造,不改模型/模板/读出)。
- **有界试点**:CPU=对 dev 仿真族执行谓词 DAG 生成因子化目标(程序格 B1 已有,场景标注为公开数据集附带);GPU≤45min=r16 LoRA SFT 2–4k 样本,评 dev 关系切片+JevBench231 mini。
- **预注册门**:成功=关系切片(in/out+motion+start/end 合并)≥+3pt abs 且其余切片回退≤1pt 且 ECE 不变差;放弃=因子化目标与 one-hot 差异≤0.5pt(2 seeds 后)。
- **风险**:软目标模糊决断力(缓解:软质量上限 0.3+退火);谓词解析覆盖不足(先只做 3 类失败谓词);变体=作为 GRPO-on-lattice 的谓词级奖励(奖励携带 outcome 外信息,规避熵塌缩根因;但**主体不依赖 RL**)。

### I5. Deliberation 蒸馏先验(D3P)— 排名 2

- **一句话范式**:教师(OneJev27B 同输入)不蒸馏 argmax 而蒸馏"短 deliberation 后/金答案反事实提示探针后"的**完整选项分布**,单 token 学生在零生成 token 条件下吸收该分布。
- **Novelty claim**:System-2 蒸馏(Nye et al. 2021,检索摘要提及,未见直接链接)、Distilling step-by-step、implicit CoT 均为文本侧/生成式;**反事实提示探针(given-answer 提示下的选项分布)作为分布蒸馏目标**未检索到先例。最近邻=System 2 Distillation、B2(推理时 generate-then-decide;D3P 把全部 deliberation 成本移到训练时)。
- **命中失败模式**:video500 师生差 19.8pt(94.2 vs 74.4)=最大已知天花板;顺带各模态。
- **增益机制与量级**:rationale/hint 条件下的教师分布严格多携带信息;蒸馏文献单 token 可吸收其 2–5pt。**先做免费 CPU 检查**:若教师分布已近 one-hot(熵<0.1)则收益封顶,直接放弃。
- **契约**:合规(标签侧;学生仍是官方模板单 token)。
- **试点**:GPU 30min 教师 hint-probe 重查询(cached 输入)+SFT 30min 内 2k;成功=video dev +2pt;放弃=与普通教师标签 SFT 打平。
- **风险**:教师 API/权重可用性;探针偏置(用两种提示取平均)。

### I2. 孪生分歧 U 监督(Twin-Divergence U)— 排名 3

- **一句话范式**:用**未挂 LoRA 的同权重冻结基座**作为答案性 oracle:U 目标 =f(基座熵, 基座-教师分歧),训练决策 LoRA 在"证据不支持"时精确输出 U;推理时只跑 LoRA,零额外成本。
- **Novelty claim**:[R-Tuning](https://arxiv.org/abs/2311.09677) 用**同一模型自采样一致性**造弃权标签;AbstentionBench 表明推理模型弃权更差;**"同权重减适配器的孪生分歧"作为 U oracle 未检索到先例**。检索词:`base model self-knowledge abstention distillation adapter "frozen backbone" entropy unanswerable supervision`(零命中)、`R-Tuning unknown-aware tuning...`(命中=自对齐,非孪生)。
- **命中失败模式**:ambiguous 家族 2/7(决策先验在故意含糊题上劣于基座——恰说明基座保留了答案性信息而 LoRA 训练抹掉了它);crossmodal ECE .252;p(U) 产品指标。
- **增益机制与量级**:基座携带预训练"认知上限",答案 SFT 造成过度作答;孪生分歧隔离"适配器带来的"vs"证据支持的"。估计:ambiguous 家族 2/7→5–7/7;主答案精度 +0–1pt(诚实:主要收益在校准与 U,不在精度)。
- **契约**:合规(训练时标签;基座输出可缓存)。
- **试点**:CPU=用已缓存基座输出拟合 f(校准集);GPU≤45min=2k U 增强目标 SFT;成功=ambiguous U-正确≥5/7 且可答题集掉点<0.5pt;放弃=U 频率漂移>2× 目标基率。
- **风险**:基座在 crossmodal 上自身答案性差(先只对 video/image 族启用);f 过拟合校准集(预注册简单函数族:熵阈值+分歧指示)。

### I6. 模态条件混合蒸馏(MCMD)— 排名 4

- **一句话范式**:教师在模态子集变体(video-only/audio-only/text+video)上的分布组成可靠性门控混合目标 p\*=g(p_full, p_v, p_a),g 在校准集按题型拟合;学生单次前向直命中 p\*。
- **Novelty claim**:C2KD/跨模态蒸馏是**教师模态→学生模态**的迁移;[模态偏差](https://arxiv.org/abs/2505.18657)研究只诊断不构造目标;"可靠性门控模态混合目标蒸馏进融合输入单前向决策"未检索到。检索词:`modality gap unimodal bias multimodal LLM distillation cross-modal calibration 2025`(命中=诊断/迁移类,零混合目标类)。
- **命中**:crossmodal500 71.0%(最差精度)+ECE .252(最差校准)。
- **机制/量级**:直接惩罚单模态捷径;混合目标天然带软结构≈校准。估 **+3–6pt crossmodal、ECE 显著降**(模态偏差修正文献量级)。
- **契约**:合规(标签侧)。
- **试点**:GPU 30min 模态子集教师查询(cached)+SFT;门:crossmodal +2pt 或 ECE≤0.18;放弃=g 不优于均匀平均。
- **风险**:子集查询成本只在数据构建期;g 过拟合(预注册 2–3 个 g 候选)。

### I7. 时间锚索引 scaffold — 排名 5(轻 ADR)

- **一句话范式**:基座自生成**媒体级时间索引**(帧时间戳事件摘要,题目无关、可按视频缓存)注入用户内容,把 start/end 存在性变成文本可核对事实;非每题 CoT。
- **Novelty**:dense captioning 与 dense video QA 存在;"自生成媒体级时间索引服务整个问题格(与 GRPO-on-lattice 同构按视频分组)的单 token 决策"未检索到先例;与 B2 差异=索引 amortize 到视频级、问题无关。最近邻=B2、Motion-as-Prompt(overlay 而非文本索引)。
- **命中**:start/end 13 错、motion 15。
- **机制/量级**:时间边界检测→文本匹配;一次索引服务全格(格内问题共享)。估事件切片 **+3–8pt**(caption 条件化 QA 类比)。
- **契约**:**轻 ADR**——输入侧含自生成内容,但非 thinking、非每题生成;需确认"官方 chat template"仅约束消息结构还是也约束用户内容纯度。
- **试点**:GPU 20min 索引生成(2 采样一致性过滤)+零样本决策评估→若零样本 +1pt 再 SFT-lite;门:事件切片 +2pt。
- **风险**:caption 幻觉污染同视频全部问题(双采样一致性过滤;索引错误率单列报告)。

### I3. 感知边缘化决策(PMD)— 排名 6(测试时,服务契约需决策)

- **一句话范式**:推理时对同一媒体跑 K 个廉价视角(时间段/空间裁剪/模态子集),各得单 token 读出,输出混合 p(a)=Σw_k p_k(a),w_k 由各视角熵/U 特征在校准集学习;零生成 token。
- **Novelty**:PTS/grounding paradox 走 token-rich 感知;TTS 综述称感知侧 TTS 空白;**"学习权重视角混合→单 token 读出"检索零命中**。检索词:`multi-crop ensemble video language model distribution averaging learned view weights single forward decision`(仅无关 ensemble 论文)、`test-time scaling self-consistency multimodal perception marginalization single answer token`(综述+PTS,无本范式)。
- **命中**:in/out(裁剪视角)、start/end(时间段视角)、crossmodal(模态子集视角暴露捷径)。
- **机制/量级**:对固定帧下的感知注意方差做 Monte-Carlo 边缘化;自一致性类收益 2–5pt,最差切片估 **+3–6pt**;代价 K× 推理(仅 GPU 档)。
- **契约**:模型零改动;但多次前向+读出聚合,**需产品级决策**(单步/延迟契约)。vLLM 实现零侵入(多次标准请求,外部聚合)。
- **试点**:零训练:cached dev,K≤5,熵加权 vs 均匀 vs 学习权重三臂;GPU 45min 内含 ablation;门=最差两切片各 +2pt;放弃=均匀≈学习权重(退化为普通 TTA)。
- **风险**:混合冲掉高置信正确答案(备选几何平均臂);延迟与端侧档不兼容。

### I9. 干预孪生等变监督 — 排名 7

- **一句话范式**:仿真器最小编辑孪生对(同视频改一个谓词),监督**结构化变化**:被编辑谓词牵涉的问题格题目答案必须按编辑方向变化,格内其余题目分布必须不变(等变+不变双约束)。
- **Novelty**:CF-VQA/反事实增强与标签不变增强理论([Yang, ICML 2023](https://arxiv.org/abs/2202.12230))是标签不变式;**"答案恰好随被查谓词编辑而变"的等变式约束覆盖问题格**未检索到(检索词 `counterfactual data augmentation consistency regularization vision question answering 2024 2025` 零直接组合命中)。
- **命中**:关系簇全部+U(编辑致含糊的孪生教答案性边界)。
- **机制/量级**:强制因果依赖正确视觉特征、杀相关性捷径;估 +2–5pt 关系切片。
- **契约**:合规(数据+损失)。
- **试点**:CPU 仿真重渲染管线(若族内支持);GPU 30min SFT;门=+2pt 且 held-out 孪生等变违反率<10%。
- **风险**:重渲染成本(不可行则退化用现有反事实池的自然对);约束过强伤其它题(不变项系数退火)。

### I4. 世界状态硬负例选项合成 — 排名 8

- **一句话范式**:用世界状态生成"谓词编辑邻居"干扰项(与 gold 恰差一个谓词:碰撞对象换色/事件提前退出),教决策绑定精确谓词组合。
- **Novelty**:干扰项生成在 NLP MCQ 成熟;场景图接地干扰项合成用于视频单 token 决策训练未检索到(同 I1 检索)。增量型创新,与 I1 共享执行器。
- **命中**:关系簇;机制=最大化部分感知下的混淆度;估 +2–4pt。
- **契约**:合规。试点=CPU 生成+30min SFT,门同 I1。
- **风险**:过难干扰项教文本捷径(掺混比 30%)。

### I10. 置换一致性正则 — 排名 9(先例充分,降级为增量)

- 一句话:选项重排下的内容级分布一致性损失。**先例**:General Permutation Debiasing(teacher-student 置换一致性,ACL Findings 2024,检索摘要提及)、[EMBER](https://ojs.aaai.org/index.php/AAAI/article/view/40401/44362)(AAAI 2026 置换等变架构)→ 不构成范式级创新;且实测错误结构与顺序偏差不重叠。仅作为 I1/I5 训练的免费附加正则(强度 0.1),不单独立项。

### I8. Verbalizer 几何读出 — 排名 10(需 ADR,不推荐先行)

- 一句话:Choice 读出用各选项自有内容 token 而非位置字母,softmax 落在语义组织区。**先例**:PET verbalizer 选择、contextual calibration(2021)→ NLP 侧成熟;VLM 单 token 决策侧未检索到但属迁移。**契约冲突**:改答案 token 映射(契约冻结项)→ 必须用户+ADR。预期 +1–2pt 量级,收益/契约成本比差,**不建议先做**。

## 6. 排序表(期望收益 × 可行性/成本)

| 排名 | Idea | 维度 | 命中失败模式 | 预期主收益(诚实) | 契约 | GPU 试点 | Novelty |
|---|---|---|---|---|---|---|---|
| 1 | I1 WSFT | 数据/目标构造 | 关系簇 50 错 | +4–8pt 关系切片 | 合规 | ≤45min | 高(零先例) |
| 2 | I5 D3P | 标签构造 | video 师生差 19.8pt | +2–5pt video/全模态 | 合规 | ≤45min | 中高(探针目标零先例) |
| 3 | I2 孪生 U | 校准/监督 | ambiguous 2/7、ECE | U-正确 2/7→5+/7 | 合规 | ≤45min | 高(零先例) |
| 4 | I6 MCMD | 标签构造/校准 | crossmodal 71/ECE.252 | +3–6pt、ECE 降 | 合规 | ≤45min | 中高 |
| 5 | I7 时间锚 | 输入构造 | start/end、motion | +3–8pt 事件切片 | 轻 ADR | ≤45min | 中高 |
| 6 | I3 PMD | 测试时计算 | in/out、crossmodal | +3–6pt 最差切片 | 服务决策 | ≤45min(零训练) | 中高 |
| 7 | I9 等变孪生 | 数据/损失 | 关系簇+U | +2–5pt | 合规 | ≤45min | 中高 |
| 8 | I4 硬负例 | 数据构造 | 关系簇 | +2–4pt | 合规 | ≤45min | 中 |
| 9 | I10 置换正则 | 损失(附加) | (顺序偏差,非实测簇) | +0–1pt | 合规 | 并入 | 低(有先例) |
| 10 | I8 verbalizer | 读出/校准 | (顺序偏差) | +1–2pt | 需 ADR | — | 低(有先例) |

## 7. 前 3 建议 + 明确不做

**前 3 = I1 + I5 + I2**(可并行,共享程序格/缓存基础设施):
1. **I1**:唯一直接打击 50 个已定位关系/事件错误且零契约成本的范式级空白;符号路线已在 SOTA 端证明该层可拿满。
2. **I5**:唯一攻击最大量化天花板(19.8pt 师生差)的纯标签侧手段;先花 0 成本查教师分布熵决定去留。
3. **I2**:修复唯一"劣于基座"的实测回归(ambiguous),且 U/p(U) 是产品级指标;基座输出大概率已在缓存里,近乎免费。

**明确不建议做**:
- **纯 outcome-RLVR/GRPO on 决策 token**:内部已证数学等价 CE(已停路线);文献根因=二元奖励→熵塌缩([Entropy Collapse in RLVR](https://arxiv.org/html/2605.11491v1))。RL 仅在 I1 谓词奖励变体下有增量,且非主体。
- **Slot/视觉 tokenizer 替换**(SlotTok/Slot-VLM):违"冻结主干"硬约束,且已有先例、非范式空白。
- **注意力操控/推理引擎钩子**(OPERA 系):违"不改推理引擎"。
- **新增 PRM/verifier/分类头**(VisualPRM 路线):违"不新增分类头";且 8B PRM 服务成本与端侧档矛盾。
- **EMBER 式置换等变架构**:改架构+有先例。
- **推理时显式仿真/CoT**(CRCG/P TS token 路线):违 thinking-off+单步产品契约(PTS 是最强邻居,我们借其"感知侧扩展"论点但不走 token 路线)。

## 8. 需用户+ADR 决策清单(突破冻结契约项)

| Idea | 冲突点 | 预期收益 | 建议 |
|---|---|---|---|
| I7 时间锚索引 | 输入含自生成媒体级内容(非 thinking、非每题生成);"官方模板"是否约束用户内容纯度 | +3–8pt 事件切片 | 轻量 ADR 后试点 |
| I8 verbalizer 读出 | 契约冻结"答案 token 映射" | +1–2pt | 收益/成本比差,不建议 |
| I3 PMD 多次前向 | 单步/延迟产品契约(仅 GPU 档启用?) | +3–6pt 最差切片 | 若试点过门再提请产品决策 |
| (弃)Slot 注入 | 冻结主干+视觉 tokenizer | 未知 | 已有先例,不值违约 |
| (弃)注意力操控 | 不改推理引擎 | 未知 | 违约,不做 |

## 9. 检索记录(novelty 断言依据)

零直接命中的检索词(经 Z.ai web_search_prime,2026-10-16):
1. `process reward model multimodal step-level verification video question answering 2025 scene graph simulator` → 仅 VisualPRM/StructReward/数学 PRM,无仿真器执行+单 token 直答组合。
2. `CLEVRER scene graph annotations program execution partial credit reward training neural video QA` → 仅 VeriGraph(机器人规划)等无关项。
3. `base model self-knowledge abstention distillation adapter "frozen backbone" entropy unanswerable supervision` → 零命中。
4. `multi-crop ensemble video language model distribution averaging learned view weights single forward decision` → 零命中(仅无关 ensemble)。
5. `test-time scaling self-consistency multimodal perception tasks 2025 marginalization single answer token` → PTS/综述,确认感知侧 TTS 空白但无本范式。
6. `counterfactual data augmentation consistency regularization vision question answering 2024 2025` → 组合零命中。
7. `modality gap unimodal bias multimodal LLM distillation cross-modal calibration 2025` → 诊断/迁移类,无混合目标蒸馏。

有先例(用于降级/差异化):置换一致性(EMBER、Liusie 2024)、slot 注入(SlotTok、Slot-VLM)、overlay(SoM、Graph-of-Mark、Motion-as-Prompt)、弃权自对齐(R-Tuning)、System-2 蒸馏(Nye 2021)。

**数字置信声明**:CRCG 96.1%、前沿 VLM 92.7–95.3%、MMAU 66.15%/55.4%、MMAU-Pro 59.2% 均来自搜索摘要二手转述,已给链接;对外引用或写 ADR 前应回原文核对。内部实测数字(74.4/94.2/82.8/79.8/71.0/99.6/97.0/76.62)引自任务书,未复测。

## 参考(一手链接)

CLEVRER https://arxiv.org/abs/1910.01442 | CRCG https://arxiv.org/abs/2506.10753 | CLEVRER-Humans https://arxiv.org/abs/2310.03635 | MMAU https://sakshi113.github.io/mmau_homepage | MMAU-Pro https://ojs.aaai.org/index.php/AAAI/article/view/39430/43391 | VLM-R1 https://arxiv.org/abs/2504.07615 | Perception-R1 https://neurips.cc/virtual/2025/poster/119361 | Entropy Collapse RLVR https://arxiv.org/html/2605.11491v1 | PTS https://openreview.net/forum?id=WGIcXH9rk9 | Grounding Paradox https://arxiv.org/html/2604.11025v1 | SoM https://arxiv.org/abs/2310.11441 | Motion-as-Prompt https://arxiv.org/html/2608.11655v1 | Localize→Spatial(CVPR 2024,无直接链接) | 选项顺序 https://arxiv.org/abs/2308.11483 | EMBER https://ojs.aaai.org/index.php/AAAI/article/view/40401/44362 | 置换去偏(ACL Findings 2024,无直接链接) | R-Tuning https://arxiv.org/abs/2311.09677 | AbstentionBench https://arxiv.org/abs/2506.09038 | Know-Your-Limits 综述(TACL 2025,无直接链接) | VisualPRM https://internvl.github.io/blog/2025-03-13-VisualPRM | StructReward https://arxiv.org/html/2608.08326v1 | 模态偏差 https://arxiv.org/abs/2505.18657 | SlotTok https://arxiv.org/abs/2505.17726 | Slot-VLM(NeurIPS 2024,无直接链接) | System-2 Distillation(2021,无直接链接) | 增强一致性理论 https://arxiv.org/abs/2202.12230
