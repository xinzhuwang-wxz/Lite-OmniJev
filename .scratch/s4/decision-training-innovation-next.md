# 下一训练创新判定：先补原任务可辨性诊断，暂不启动新目标 GPU 小试

2026-10-16；所有者 decision_rl_evidence；基于代码 34eea16。范围仅设计与 CPU 数学核验，无新训练、下载、校准/封存读取。机器证据见 reports/explore/decision-training-innovation-next-001.json。

## 决定与证据

本轮提出 **0 个可立即执行的新目标**；拒绝为了创新名义再运行 expected-reward、DPO、Brier/KL 扫权重或极性 flip。不是认定单步不适合 RL，也不是证明一切新目标无效，而是尚未识别能解决当前瓶颈且可验证的新监督或业务效用。

- train/decision_reward.py 已正确注释：one-hot 奖励产生 -q_gold，是 CE 梯度的 q_gold 缩放；helper 未接训练。其 KL 实际方向是 KL(reference || q)，不得误称另一方向。
- reports/s4-claim-pair-original100-comparison-001.json：匹配 CE 与 pair 均 73/100、预测完全相同，pair Brier .379226 > .373116。补充来源概率方向不同，不能称稳定收益。
- reports/s4-temporal-brier-vs-matched-ce-001.json：两者110/150；Brier .374914→.370032，但 temporal 70→69/97。历史 sidecar 没有 input/tensor hashes，不能夸大严格张量配对。
- reports/explore/s4-temporal-coverage-full500-diagnostic-001.json：第二段356/500，temporal224/329；在同dev拟合的乐观阈值也仅356/500、temporal226/329。它不是新阈值或无偏上限，只说明这批输出不能靠单阈值修好大部分错误。首段当前工作候选完整video为354/500，不能混用两模型指标。
- reports/explore/s4-temporal-coverage-segment003-results-001.md：第三段DEV50 36→37，但更广video150116→108，说明反复小切片筛选不足以支持推广。
- reports/explore/s4-temporal-order-contrast-results-001.md：collision顺序程序gold试验DEV50仍33/50，Brier .447486→.478265；已见TRAIN12仅8→9，证明局部学习而非原任务泛化。不要重做相同语义的变体实验。
- reports/explore/s4-temporal-order-contrast-feasibility-001.md：固定24 TRAIN题全部exist subtype，其中10含filter_before/after、15含filter_in、5含filter_out、0含filter_collision（重叠计数）。事件顺序builder产生的是额外collision关系，不能代表原时序判断。采样bucket分离仅是机会，不证明16帧观察可判真值。
- 固定TRAIN12记忆成功而DEV50退化，不能据此推断冻结主干容量不足；当前更直接的问题是覆盖、任务语义匹配及输入证据可辨性。

这些是不同候选与对照的证据，不拼接成单一模型成绩；本轮不宣称新增模型质量证据。

## 梯度与信息：为什么不再用 one-hot reward 充当新目标

令 q=softmax(z)，动作是全部合法答案及 U，gold 为 y。L_CE=-log(q_y)，L_reward=-q_y。

∂L_CE/∂z_j=q_j-1[j=y]；∂L_reward/∂z_j=q_y(q_j-1[j=y])。

所以方向完全相同、只依当前样本q_y缩放，参数空间链式梯度也如此；所有非gold（包括U）得同类信号，没有事件顺序/可观察性新信息。恒定奖励梯度为零。任意固定奖励时 ∂(-E_q[r])/∂z_j=q_j(E_q[r]-r_j)，若错误选项的效用不同，方向可以非共线；但其区别必须来自可审计的真实效用/结构gold，不能凭“RL”设定。CPU反例仅证明数学可能性，不授权采用其合成奖励。

CE、Brier、KL、group proportion约束可能改变优化几何；“不是CE标量倍数”不等于“新增监督”。若群组比例只是同一批one-hot标签求和，也未增加真值信息；在此处没有证据支持其能学会原时序谓词。成本敏感奖励更不能在没有业务代价定义时自拟错答/U代价。U是真实证据不足事件，不得把模型错题、低置信或未采中碰撞帧自动标U。

## 唯一先决诊断：原 TRAIN 时序谓词 × 16帧证据可辨性

复用已批准CLEVRER TRAIN的pool/ledger、官方postfix programs和annotation，不引入数据源。此前collision诊断的固定24family可作为审计入口；先按规范family过滤读取，不能扫描或解码cal/holdout载荷。原始标签不改变、原问题不改写、没有重新极性flip。

1. 对这24题逐题以原始program区分入场、出场、before/after过滤和被查询状态；记录TRAIN pool、source/member、program、原gold、media与实际processor的frame indices/tensor哈希。已有统计直接复用，不重复做collision统计。
2. 只对明确实现并用官方原gold交叉核对的程序算子做CPU求值；未知算子、缺标注、重名对象、重复事件保持unresolved并计数，绝不猜标签。若本地没有已验证executor，则首先交付算子覆盖表，不能以函数名猜程序语义。
3. 对同一个程序与已准备的真实16帧，区分“能给出标注层充分证据证书”“已证明有歧义”“未证明”；把未被采样的帧信息遮蔽之后，若两个都满足可见约束的合法事件完成具有不同答案，可以证明观测不足；仅事件离最近采样帧很远不能构成这种证明。标注坐标也不等于模型真的看清，证书与视觉可辨性要分别报告。
4. 输出聚合表和哈希，不导出原题、原始媒体或内部时间到模型prompt；允许的额外监督只可能来自经过核验的官方TRAIN程序执行/约束，而不是教师猜测、dev错误或封存反馈。

CPU预算：24 family、180秒、一次审计，不下载模型、无GPU。无法取得已绑定输入或程序语义时输出unresolved并停止；不为了凑齐另取family。当前本轮只完成数学验证与现有证据复核，尚未执行这个需要现有TRAIN资产的诊断，不能标为ready。

## 何时重新提出一个新目标及严格对照边界

只在诊断识别出可验证、与原任务相关的结构约束/真实效用，且可写出不等价于CE重权的梯度后重新提案；目前没有可启动的B arm，不创建空壳GPU作业。

届时先在CPU证明标签/约束正确并测试梯度与非有限拒绝，再冻结唯一目标；A=现有CE，B=CE+该唯一目标。两臂必须同初始化adapter SHA、相同120draw TRAIN family/顺序/回放、seed、模板、合法答案+U映射、实际输入tensor、16帧、优化器初态及学习率。额外结构配对输入如存在，A也须相同曝光并计入120draw，记录前反向次数/显存/墙钟，不能把更多数据收益当目标收益。

每臂最多10 optimizer steps和240秒，step5保存checkpoint；任一臂超时/NaN/Inf/OOM/身份或隔离不符立即停止，保留证据，不自动重启或扫参。先完成A再B，有限对照不是已经授权的GPU计划。已有CE结果只有上述身份全部一致才能复用。

预注册筛选：固定已有DEV50/原video150，不按候选表现重新选题；video150要求相对严格匹配CE净增至少3题，且NLL、Brier不恶化、ECE恶化不超过.01、coverage@5%不下降；原cross及各旧模态50题不得净退超过1题，U/yes分布变化须给配对解释。此为未来试验筛选，不改阶段gate，亦非显著性声明。任一条件失败则停止该目标路线；通过才允许更广dev和独立gate，不能反复更换小切片。已有多轮dev曝光需显式报告选择偏差，不将cal/holdout用作筛选集。

## 交接

继续当前主线已授权的数据覆盖/保留优化器短段工作；本结论不改变其调度、候选或质量门。创新GPU当前状态rejected_pending_prerequisite，不是永久否定RL或等待用户许可。唯一后续是上述TRAIN原程序诊断；诊断失败也应收口，不硬凑新目标。
