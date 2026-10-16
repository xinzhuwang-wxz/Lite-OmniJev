# S4 前沿深度调研 002:第二轮新维度(数据/检索/路由/组合/训练动力学/音频/校准)

- 日期:2026-10-16 | 类型:纯研究(零 GPU;仅文献检索)| 前置:见 001(视频因果 SOTA/RLVR/测试时扩展/对象中心/蒸馏/置换一致性)
- 排序标准(用户澄清沿用):前人没用过 × 命中实测失败模式 × 期望收益 × 可行性/成本;本轮 8 个 idea,给出 top3 与不建议项
- 内部实测数字引自任务书/协调者提示,未复测;外部数字均为搜索摘要二手转述并给链接

## 0. TL;DR

- 七个维度中,**数据维度(模拟器在环)与组合维度(子决策辅助监督)检索不到先例组合**;检索/路由/校准/训练动力学四个维度的核心机制均有扎实先例(kNN-Prompt、TRIAGE/UEC 级联、AdaTS/ATS、Model Soups),只能做"组合式增量",已如实降级。
- **建议 top3**:II-2 双轨事件脚本合成(单脚本渲染同步 video+audio,AND 组合真值精确、按模态分量做反事实编辑)、II-5 子决策辅助监督(AND 题分解为模态子题做训练期辅助目标,推理仍单前向)、II-4 程序键 kNN 决策插值(零 GPU、cached logits 即可验证)。
- 与 001 的协同:II-1/II-2 产出的合成数据天然携带 001-I1(世界状态因子化目标)所需谓词真值;II-3 路由与 001-I2(孪生分歧 U)共享同一分歧信号基础设施。

## 1. 各维度调研结论

**1.1 数据维度(模拟器在环/程序驱动扩增)**
- 程序重组问题生成**有先例**:[AGQA](https://ai.stanford.edu/blog/agqa)([arXiv 2103.16002](https://arxiv.org/abs/2103.16002),192M 程序生成 QA 对,Action Genome 视频)、CLOSURE"Measuring Compositional Consistency"(CVPR 2022,Gandhi et al.,程序重组测组合一致性)、AGQA 2.0(平衡划分去语言偏置)、AGQA-Decomp(问题→子问题 DAG 分解引擎,组合维度的最近邻)、GQA(2019 图像版)。文本侧:[Puri et al. 2020] 合成 QA 训练超过真实标注(EMNLP,SQuAD)。
- **"模拟器在环的决策校准训练"检索零直接命中**(检索词见 §9-1):仿真合成数据(NVIDIA robotics/物理 AI 管线)、物理引导视频生成、合成 QA 各自碎片存在,无一统一到"闭环误差驱动→决策 token 训练"。
- 对我们的机会:CLEVRER 视频本身由物理引擎渲染、场景文件公开,可重渲染 + 重编程问题;池隔离合规做法=只在 TRAIN family 视频上生成新问题/新渲染(变体不跨池)。

**1.2 检索维度(kNN 式插值)**
- 核心机制先例充分:[kNN-LM](https://openreview.net/pdf?id=HklBjCEKvH)(分布插值)、[kNN-Prompt](https://arxiv.org/abs/2205.13792)(**零样本 MCQ/分类:邻居标签分布作软目标**——与我们最接近的先例)、[Drozdov et al. EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.218.pdf)(**λ 按检索距离条件化**)、[Xu et al. ICML 2023](https://proceedings.mlr.press/v202/xu23a/xu23a.pdf)(机制分析)。
- 空白:**程序结构作为检索键**、**typed decision(Choice/Score/Noul/U)的插值校准**、λ 用孪生分歧特征条件化——组合未检索到。

**1.3 路由维度(高熵分派)**
- 先例扎实:TRIAGE token-level cascade(**UEC:用 SLM-LLM 联合预测结果校准 SLM 熵→deferral 信号**——与我们设想几乎同构)、UCCI(arXiv 2026-05,cost-optimal 级联的校准不确定性)、FrugalGPT 谱系、"Don't Build a Router. Train the Small Model to Know When"(Distill Labs,**= 001-I2 的主张**)。
- 结论:机制非新;我们的独有资产是**孪生结构(同权重±adapter)+ U 作为一等路由端口 + 实测互补性(oracle 81.8%)**。诚实定位=工程应用,不做范式宣称。

**1.4 组合维度(AND 题分解)**
- 推理时分解先例:零样本 VQA 分解(NeurIPS 2023,多步管线)、Successive Prompting(EMNLP 2022)、无监督多跳分解(Perez et al.)、AGQA-Decomp(评测期分解)。
- **训练期辅助分解监督(推理单前向、无 CoT)检索零直接命中**(§9-4);搜索摘要明确提示该组合"可能是新的"。

**1.5 训练动力学(adapter soup/EMA/检查点平均)**
- [Model Soups](https://arxiv.org/abs/2203.05482)(ICML 2022,共享初始化同盆地平均有效)、[Model Stock](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06044.pdf)(ECCV 2024,少检查点智能合并)、[Branch Soup](https://www.sciencedirect.com/science/article/pii/S0925231226028043)(与"LoRA soup"对照:晚分支共享轨迹再分叉)。
- 已知失败模式:跨盆地/跨初始化平均退化(宽/平坦极小值讨论);adapter 合并的任务干扰(TIES/DARE 谱系,检索提示未见 LoRA-soup 失败模式专文)。对同种子不同步长/数据的决策 LoRA:同盆地,安全;收益典型 +0.5–1pt,零推理成本。**定位=免费快赢,非范式**。

**1.6 音频维度**
- [MUSIC-AVQA](https://arxiv.org/html/2203.14072v2)(真实音视频 QA);**Tsinghua AVQA(ACM MM 2022)含合成视频 QA+模板问题生成**——音频合成先例存在但为数据集构建,无反事实分量控制、无决策 token 训练;SoundSpaces(声学仿真)未在本轮检索直接命中(仅背景知识,引用前需核对)。
- 空白:程序化音频事件场景(计数/顺序/时间窗/声源身份谓词)+ 精确真值的决策训练;尤其**单事件脚本双轨渲染(同一时间线出 video+audio)** 未检索到(§9-3)。

**1.7 校准即服务**
- 直接先例:[AdaTS](https://arxiv.org/abs/2207.06211)(AAAI 2023,per-sample 温度)、[ATS for LLMs](https://aclanthology.org/2024.emnlp-main.1007.pdf)(EMNLP 2024,per-token 温度,PEFT 式实现,代码开源)。输入自适应温度=已知技术;我们仅剩"类型化温度专家(Choice/Score/Noul×模态)+conformal 联合"的工程组合。**定位=C4 的老实扩展,非创新主体**。

## 2. 候选范式(8 个,按最终排序)

---

### II-2. 双轨事件脚本合成(Twin-Track Scripted Synthesis, TSS)— 排名 1

- **一句话范式**:用一个事件脚本(对象、碰撞、时间线、声源)同时渲染 video 与 audio 两轨,使每个 crossmodal AND 题的组合真值**由构造精确**,并可对单一模态分量做反事实编辑(只改视频不改音频→答案应翻转/不变),以此训练决策 LoRA。
- **Novelty claim**:合成 AVQA 数据集存在(Tsinghua AVQA 2022,模板题),但都是**开环数据集构建**;**"单脚本双轨渲染 + 分模态反事实编辑 + 组合真值精确 + 决策 token 训练"未检索到先例**(§9-3)。最近邻=Tsinghua AVQA(合成但无反事实分量控制、无决策训练)、MUSIC-AVQA(真实数据)、AGQA(单模态视频)。
- **命中失败模式**:crossmodal500 71.0%(最差精度)+ ECE .252(最差校准)——AND 组合错误正是"两个模态分量各自对、组合错"或"单模态捷径";反事实分量编辑直接教"哪个分量决定答案"。
- **增益机制与量级**:组合真值精确→可造 001-I1 因子化目标与 II-5 子决策监督;分模态编辑→等变约束(001-I9 的可执行版)。估 crossmodal **+3–6pt、ECE 显著降**(模态偏差修正文献量级;诚实:合成→真实迁移是主要不确定性)。
- **契约**:合规(数据侧;渲染用仿真器与 DSP,非生成式模型,不引入新依赖栈或许可风险;音频合成用简单事件合成器,非 AudioCraft 类生成模型)。
- **有界试点**:CPU=1–2k 双轨样本(脚本→渲染+DSP),谓词覆盖 AND 组合与反事实编辑各半;GPU≤45min=混合 30% 合成 SFT,评 crossmodal dev+真实分布切片(检验 sim2real 迁移);零 GPU 备选=先只做"反事实编辑一致性"评测集,量测现有候选模型的分量依赖偏差(纯评测,不训练)。
- **预注册门**:成功=crossmodal dev +2pt 或 ECE≤0.18,且 text/screenshot 不回退;放弃=合成混合训练在真实分布切片 0 增益(说明 sim2real 断裂),或反事实评测显示模型已分量正确(问题不在组合)。
- **风险**:合成域与真实域差距(缓解:域随机化渲染参数+真实数据主导混合比);音频 DSP 保真度(先用事件类声音,避开语音);渲染算力为 CPU 时间(预算内)。

### II-5. 子决策辅助监督(Sub-Decision Auxiliary Supervision, SAS)— 排名 2

- **一句话范式**:把 crossmodal AND 题程序化分解为两个模态子题("视频中 X 是否发生""音频中 Y 是否发生"),作为**训练期辅助任务**(额外样本/辅助 prompt 形式)监督同一决策 LoRA;推理仍是单前向单 token,分解只在训练时存在。
- **Novelty claim**:分解先例全部是**推理时多步管线**(零样本 VQA 分解 NeurIPS 2023、Successive Prompting、AGQA-Decomp 评测期);"训练期辅助分解监督、推理单前向无 CoT"检索零直接命中(§9-4)。最近邻=AGQA-Decomp(分解引擎,评测用)、001-I6 MCMD(模态**输入**子集蒸馏,非问题分解)。
- **命中失败模式**:crossmodal 71.0(组合错误);间接支持关系簇(谓词分解同构)。
- **增益机制与量级**:多任务辅助任务教内部因子化,单 token 读出吸收子结构;多任务 VQA 增益典型 +2–4pt;估 crossmodal **+2–4pt**。与 II-2 天然配套(脚本自带子题真值);没有 II-2 时可用程序格对现有数据造子题。
- **契约**:合规(训练侧辅助样本;推理不变)。
- **试点**:CPU=从程序格/脚本生成子题对;GPU≤45min=主任务+辅助任务混合 SFT,评 crossmodal dev;门:+2pt 或 ECE 改善≥30%;放弃=辅助任务稀释主任务(主任务掉点>1pt 且无增益)。
- **风险**:辅助任务与主任务梯度竞争(调度退火;辅助权重≤0.3);子题措辞偏置(模板随机化)。

### II-4. 程序键 kNN 决策插值(Program-Key kNN Decision Interpolation, PKI)— 排名 3

- **一句话范式**:TRAIN 池建立 datastore,键=(问题文本嵌入 ⊕ 程序结构编码 ⊕ 模态/题型类型),值=gold/教师选项分布;推理时检索 K 近邻,与单 token 读出插值 p'=(1−λ)p+λ·Σw_i p_i,λ 由(邻居距离、邻居一致性、候选熵)在校准集拟合——决策分布的检索校准。
- **Novelty claim**:kNN-Prompt 已做"邻居标签分布辅助零样本 MCQ"、Drozdov 已做距离条件 λ;**"typed decision × 程序结构键 × 校准集拟合 λ(含分歧特征)"组合未检索到**(§9-2)。诚实定位=已知组件的新组合,非从零范式。
- **命中失败模式**:ECE .252(插值天然收缩过置信)、长尾题(检索邻居提供先验);oracle 互补 81.8% 暗示外部信号可用。
- **增益机制与量级**:非参数平滑决策分布;估 ECE **降至 ≤0.15 量级、acc +0.5–1.5pt**(kNN 插值文献量级,保守)。
- **契约**:推理侧新增检索依赖——**服务架构决策**(vLLM 旁挂 ANN 索引;延迟 ~ms 级但改变服务形态);训练零改动。
- **试点**:**零 GPU**——全部在 cached dev logits + TRAIN 标签上做:建索引、拟合 λ、三臂对比(无插值/固定 λ/条件 λ);门:ECE 降≥25% 且 acc 不掉;放弃=邻居召回@10 内无金分布(检索键失效)。
- **风险**:检索键在短问题上区分度低(程序结构键正是为此);泄漏防护(仅 TRAIN 池,样本族隔离沿用);服务工程量。

### II-1. 谓词靶向闭环问题合成(Closed-Loop Question Synthesis, CLQS)— 排名 4

- **一句话范式**:程序文法在 TRAIN family 视频+世界状态上合成新问题,**闭环**瞄准当前误差聚类(每轮评 dev 误差→按谓词生成率反比于正确率),并叠加 nuisance 重渲染(纹理/光照/颜色域随机化,标签不变对)。
- **Novelty claim**:AGQA/CLOSURE 生成机制是**开环、无误差反馈、非决策 token**;"误差聚类驱动的闭环生成 + 域随机化重渲染 + 决策 LoRA"未检索到(§9-1 检索"模拟器在环决策校准训练"零命中)。最近邻=AGQA(开环生成)、CLOSURE(组合一致性诊断,非训练数据)、域随机化(sim2real 机器人,Tobin 2017 谱系,概念先例)。
- **命中**:关系簇 50 错(in/out 22/motion 15/start-end 13);nuissance 对顺带提升鲁棒性与 U 边界。
- **机制/量级**:误差反比采样把训练质量集中在失败谓词;估关系切片 **+3–6pt**;与 001-I1 因子化目标乘性配合(合成数据自带谓词真值)。
- **契约**:合规(TRAIN family 内生成,变体不跨池;holdout 不触碰)。
- **试点**:CPU=生成管线+首轮误差画像;GPU≤45min=SFT 2–4k;门:关系切片 +3pt;放弃:闭环 vs 均匀合成无差异(闭环价值证伪)。
- **风险**:生成题分布偏移(限制模板贴近原族);过拟合合成模板(混 50% 原始数据)。

### II-3. 孪生级联分派 + U 端口(Twin Cascade with U port, TCU)— 排名 5(工程快赢,非范式)

- **一句话范式**:候选模型先行,其单 token 读出的熵经**联合结果校准**(在 calibration split 上用候选×基座×gold 联合拟合,UEC 同构)决定三路:留候选/升级冻结基座/输出 U;升级仅在候选高不确定时发生(平均成本低)。
- **Novelty claim**:**机制非新**(TRIAGE/UEC、UCCI、FrugalGPT 谱系完备);独有组合=孪生同权重结构(升级后无分布风格跳变)+ U 一等端口 + 实测互补锚点(ambiguous 基座 5/7 vs 候选 2/7;oracle 81.8%)。诚实:这是把已知级联校准搬到我们的孪生设置,**不宣称范式级新**。
- **命中**:ambiguous 家族(无需任何训练直达 5/7 水平)、整体精度上界(81.8% oracle 说明路由价值)。
- **机制/量级**:路由无学习误差时按 oracle 互补上界;现实熵信号不完美,估总体 **+1–2pt + ambiguous 修复**。
- **契约**:推理侧双前向(仅高分位触发)——**服务契约需决策**(与 001-I3 同列);零训练。
- **试点**:**零 GPU**=cached 候选/基座/gold logits 上拟合路由阈值+仿真路由收益;门:ambiguous ≥5/7 且总体 +1pt 且升级率<20%;放弃:校准集上熵 AUC<0.7。
- **风险**:升级率失控(阈值按分位预注册);两模型输出风格不一致的产品观感(同为官方模板,风险低)。

### II-8. 音频谓词程序合成(Audio Predicate Synthesis)— 排名 6

- **一句话范式**:程序化构造音频事件场景(事件计数/顺序/时间窗/声源身份/时长响度音高谓词)带精确真值,专攻 Score/Noul 音频题与 audio 79.8% 的谓词结构;是 II-2 的音频单轨版,亦是 001-I1 的音频数据源。
- **Novelty**:Tsinghua AVQA 合成先例为模板题+合成视频;"谓词程序化音频+决策 token 训练"未检索到(§9-3 同源)。最近邻=AVQA 2022、DCASE 合成数据传统。
- **命中**:audio500 79.8%(第二差模态);谓词结构与关系簇同构。
- **机制/量级**:同 II-2;估 audio +2–4pt。
- **契约**:合规;试点 CPU 生成+GPU 30min;门 +2pt。
- **风险**:事件声音真实度(域随机化音色);语音类题不适用(明确排除)。

### II-7. 类型化温度专家 + conformal 联合(T²ME)— 排名 7(C4 扩展,非新范式)

- **一句话范式**:温度按 (decision type × 模态 × 特征桶) 分专家拟合,校准集选专家;再与 C1 conformal 叠加(conformal 集大小作为温度反馈信号)。
- **Novelty**:AdaTS(per-sample)与 ATS(per-token,EMNLP 2024)已覆盖输入自适应温度;我们仅剩"类型化专家+conformal 联合"工程组合。**不宣称创新**。
- **命中**:ECE .252(crossmodal 尤甚)。
- **试点**:**零 GPU**=cached logits 拟合;门:ECE≤0.18;放弃:与全局温度差<10%。
- **风险**:分桶过拟合校准集(桶数预注册≤12)。

### II-6. 决策 LoRA 汤(Decision Soup)— 排名 8(免费快赢)

- **一句话范式**:同配置多种子/多数据序的决策 LoRA 权重平均(同盆地安全),外加 logit 空间汤对照;零推理成本。
- **Novelty**:Model Soups/Model Stock/Branch Soup 先例完备;零新意,**纯运维收益**。
- **命中**:方差型错误(多采样读出方差的权重版);典型 +0.5–1pt。
- **试点**:**零 GPU**(若多检查点已存)——纯权重平均+重评;门:+0.5pt 且无切片回退;放弃:汤掉点(盆地外,直接弃)。
- **风险**:跨数据子集的 LoRA 干扰(Branch Soup 提示共享轨迹越长越安全;只汤同任务同数据种子)。

## 3. 排序表

| 排名 | Idea | 维度 | 命中 | 预期主收益 | 契约 | 试点 | Novelty |
|---|---|---|---|---|---|---|---|
| 1 | II-2 TSS 双轨合成 | 数据/构造 | crossmodal 71/ECE.252 | +3–6pt、ECE 降 | 合规 | CPU 重+GPU≤45 | 中高(组合零先例) |
| 2 | II-5 SAS 子决策辅助 | 组合/训练 | crossmodal 组合错 | +2–4pt | 合规 | CPU+GPU≤45 | 中高(零直接先例) |
| 3 | II-4 PKI kNN 插值 | 检索/校准 | ECE.252、长尾 | ECE≤0.15、+0.5–1.5pt | 服务决策 | **零 GPU** | 中(组件有先例) |
| 4 | II-1 CLQS 闭环合成 | 数据 | 关系簇 50 错 | +3–6pt | 合规 | CPU+GPU≤45 | 中高(闭环零先例) |
| 5 | II-3 TCU 孪生级联 | 路由 | ambiguous 2/7、oracle 81.8 | +1–2pt+ambiguous 修复 | 服务决策 | **零 GPU** | 低(机制已知) |
| 6 | II-8 音频谓词合成 | 数据/音频 | audio 79.8 | +2–4pt | 合规 | CPU+GPU≤30 | 中 |
| 7 | II-7 T²ME 温度专家 | 校准 | ECE.252 | ECE≤0.18 | 合规 | **零 GPU** | 低(AdaTS/ATS 先例) |
| 8 | II-6 Decision Soup | 训练动力学 | 方差型错误 | +0.5–1pt | 合规 | **零 GPU** | 低(先例完备) |

## 4. Top3 理由 + 不建议

- **II-2**:唯一直接构造"AND 组合精确真值 + 分模态反事实"数据的路线,同时为 001-I1/001-I9/II-5 供弹药;检索确认无人做过决策训练用途的双轨反事实合成。
- **II-5**:检索明确空白的组合(训练期分解监督×推理单前向);单独可做、与 II-2 乘性增强。
- **II-4**:唯一零 GPU 即可完整验证的校准路线,可本周出结论;组件先例已如实标注。
- **不建议**:II-6/II-7 作为"创新"立项(先例完备,仅作运维/校准维护项执行);纯路由故事对外宣称范式创新(II-3 定位为工程快赢);任何把检索/路由机制写成 novelty claim 的表述——审稿与内部诚实度都会受损。

## 5. 需用户/服务架构决策清单

| Idea | 冲突点 | 预期收益 | 建议 |
|---|---|---|---|
| II-4 PKI | 推理侧新增 ANN 检索依赖(改变服务形态) | ECE 大幅改善 | 零 GPU 试点过门后再提请 |
| II-3 TCU | 双前向升级路径(单步/延迟契约) | +1–2pt+ambiguous | 同上 |
| II-2/II-8 | 无模型契约冲突;新增渲染/DSP 数据管线依赖(需写明用途与许可,按 AGENTS 新依赖规范) | +3–6pt | 依赖清单随票提交 |

## 6. 与 001 的整合视图

- 优先组合拳:**II-2(数据)→ 001-I1(因子化目标)+ II-5(子决策辅助)→ 001-I6(模态混合蒸馏)** 形成对 crossmodal 的三面夹击;**II-1(闭环合成)→ 001-I1** 对关系簇。
- 001-I2(孪生分歧 U)与 II-3(孪生级联)共享"候选×基座分歧"缓存基础设施,先建缓存,两者零边际成本。
- 001-I3(PMD 测试时边缘化)与 II-4(PKI 插值)同属读出后处理,可共用评测脚手架。

## 7. 检索记录(novelty 断言依据)

1. `simulator in the loop synthetic training data video question answering counterfactual video generation training 2025` → 碎片化(Puri 合成 QA/NVIDIA 仿真管线/物理引导视频生成),**零"闭环决策校准训练"命中**。
2. `kNN-LM retrieval interpolation classification multiple choice calibration nearest neighbor label distribution` → kNN-Prompt/Drozdov/kNN-LM 机制先例;**程序键×typed decision×校准 λ 组合零命中**。
3. `synthetic audio-visual question answering generation simulation SoundSpaces paired audio video training data` → MUSIC-AVQA/AVQA(含合成模板题);**双轨反事实决策训练零命中**(SoundSpaces 未直接命中,引用前需补查)。
4. `question decomposition auxiliary supervision training single forward no chain-of-thought compositional generalization VLM` → 全部为推理时多步分解;**训练期辅助监督×单前向零直接命中**(搜索摘要亦提示可能为新)。
5. 路由/温度/soup 维度:TRIAGE-UEC、UCCI、AdaTS、ATS、Model Soups 谱系命中充分,相关 idea 已降级。

## 8. 参考(一手链接)

AGQA https://ai.stanford.edu/blog/agqa | AGQA 论文 https://arxiv.org/abs/2103.16002 | CLOSURE(Measuring Compositional Consistency,CVPR 2022,无直接链接) | kNN-LM https://openreview.net/pdf?id=HklBjCEKvH | kNN-Prompt https://arxiv.org/abs/2205.13792 | Drozdov λ 条件化 https://aclanthology.org/2022.findings-emnlp.218.pdf | kNN-LM 机制分析 https://proceedings.mlr.press/v202/xu23a/xu23a.pdf | Model Soups https://arxiv.org/abs/2203.05482 | Model Stock https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06044.pdf | Branch Soup https://www.sciencedirect.com/science/article/pii/S0925231226028043 | AdaTS https://arxiv.org/abs/2207.06211 | ATS for LLMs https://aclanthology.org/2024.emnlp-main.1007.pdf | MUSIC-AVQA https://arxiv.org/html/2203.14072v2 | AVQA Tsinghua(ACM MM 2022,无直接链接) | TRIAGE/UCCI/FrugalGPT/域随机化/零样本 VQA 分解/Successive Prompting/Perez 无监督分解/AGQA-Decomp/Puri 合成 QA(检索摘要提及,无直接链接,引用前核对)
