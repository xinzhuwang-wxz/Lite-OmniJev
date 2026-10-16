# 跨视频条件排序：仅一次、容量对照后的候选

结论：**值得保留一个有条件短试验，当前不启动。** rank64容量对照先收口；若单纯CE已有稳定原任务收益，优先CE。若rank64在TRAIN仍存在同问题双视频排序失败、且没有明确数值/容量阻塞，才比较下述唯一目标。非RLCD复现、非SOTA、没有新标签；新增的是利用已验证配对关系的条件优化约束。

## 实际证据与契约

查阅 main 的 train/pair_objective.py、prepare-continuous-matched-pool.py 与固定课程报告（SHA见JSON）。实际池300对=600不同TRAIN family视频，加600既有replay；每步三相邻pair+六replay。pair成员canonical非媒体prefix/question、literal原文及完整official program相同，媒体SHA/family不同，官方gold分别yes/no。原视频与标签不变；raw数据未在本次重新读取，依据已绑定的builder与准备报告，不能把本次称重新验证全部300对。

root提供的最新实测：r16课程step10 DEV34/50vs33，但video150111vs116；step20 DEV30停止。首30对TRAIN12准确率7→9、pairboth2→3/6、方向2→4/6，仅部分可学。root最新rank64自身基线34/50、Brier约.4425，10步pending；这是会话证据，未把旧committed STATUS中的初始化失败当当前结论。r16和r64基线不同，严禁跨rank算增益。

旧claim_pair_loss要求同一个非空component family、相反语义claim；这里明确是不同family、相同问题不同视频。**不能传假family绕过guard，也不能直接放宽旧guard**。未来训练必须有独立跨视频契约：两TRAIN来源/ledger与媒体hash、完整文本/program/nonmedia输入一致、相反official gold、两个不同family、完整yes/no/U token映射；pair元数据绑定SHA并进入resume目标身份。主会话拥有实现；本交付不改旧接口。

## 唯一目标：条件于一正一负的配对似然

对同问题的正/负视频，d+=z_yes+−z_no+、d−=z_yes−−z_no−，Δ=d+−d−。定义 R=softplus(−Δ)，采用零margin、温度1，不加新超参搜索。若独立输出联合yes/no/U概率q，条件于“两视频恰一yes一no，且都非U”，正确gold assignment概率为

q+(yes)q−(no) / [q+(yes)q−(no)+q+(no)q−(yes)] = sigmoid(Δ)。

因此R是该条件事件的负对数似然，不把q替换为新的服务概率。训练total=(sum_all_12 CE_joint + λ sum_3 R)/12，固定λ=.1；CE含U，replay不加R。可选Brier/KL保持0。所谓反向“no视频优于yes视频”给出完全相同R，双向相加只会把权重翻倍，不另做bidirectional arm；两成员已经都收到梯度。

## 与平衡CE的差异、以及没有增加的信息

令s=sigmoid(−Δ)：∂R/∂d+=−s，∂R/∂d−=s；二阶混合偏导∂²R/(∂d+∂d−)=−s(1−s)，独立逐题CE则为0。对共享问题先验b，d±=b+v±，R对b梯度为0；它只推动视频间相对分离，消除的是这个**加性共同项**，不保证神经网络所有文本捷径被移除。同program/问题匹配不能控制其他视觉捷径。

one-hot reward对单例是q_gold×CE梯度；这里梯度强度由另一视频决定，联合方向通常不与两样本CE总梯度共线；且pair项对U-logit梯度为0，joint CE不为0。严格限定：忽略U时，每个二分类成员的方向仍能在当前点写成CE的自适应权重倍数，但权重跨样本依赖；不能声称引入新的视觉真值或彻底脱离重权。它与常数/独立单例重权不同，关键是共享先验抵消与跨视频条件化。所有监督来自现有官方gold及已验证精确配对，不创造U、不使用dev标签构造pair、不读cal/holdout。

排序不约束绝对概率：两样本都预测yes仍可能Δ很大；U也可很大而R不变。CE必须保留，pairboth、单题概率校准和U必须独立检查；不能把pair方向准确当产品正确率。softplus梯度在错误大margin时有界但会加大纠错，已正确大margin时迅速饱和。固定λ=.1且上述/12归一化，使每个pair成员额外yes/no logit梯度绝对值≤.1/12；不隐式平均两次或把3pair项放大为6项。

## 一次短试验的条件、对照和退出

- rank64容量对照终态后主会话判断：若有明确推理数值错、TRAIN梯度无效、OOM/时间阻塞，先修阻塞，不试排序。若CE通过更广筛查则优先CE，不因本提案插队。可尝试的窄情形是CE能学但匹配TRAIN方向/both仍不足，未看到可推广收益。
- A和B均从**同一个rank64初始化**独立启动，不能B从已训练A接续。相同optimizer初态、seed、固定课程前120draw、3pair+6replay、媒体tensor/16帧/模板/LR/精度；唯一差异λ0/.1。现有rank64 CE可复用仅当全部身份一致；否则重新匹配，不用r16对照。
- 每臂最多10步/240秒，step5 checkpoint，无续训/扫权重；逐step报告CE、R、Δ、pairboth、梯度范数、峰值显存/时间。pair同时保留两前向计算图可能增加峰值（旧pair实测约57GB），不能沿用串行CE显存估值；A即使不计算R也须相同输入曝光。费用未知填null。
- 提前停止：任一非有限、错pair/来源/张量/目标身份、OOM、超时。相同固定TRAIN12仅机制观察：至少多正确1个pair方向且pairboth不退，CE不升；这不是泛化验收，不能据此扩大数据。
- 固定DEV50按各自相同rank64起点与A比较；B相对A若正确数下降≥2或Brier增加>.02直接停止，其余先看配对错误与NLL/ECE/U，不因仅1题上涨称成功。只有DEV正确数不低、NLL/Brier均不高、U无异常才接原video150。video150相对A至少净增3、且不低于当前parent116/150，NLL/Brier不恶化；原cross/旧模态筛查不明显退化才获得更广验证资格。不达则停止该单权重路线；不以小样本显著性、重选切片或改质量门挽救。
- 最新曝光inventory需包含两臂实际draw；尚未跑的1200池不能称已训。筛选沿用现有dev，明确多轮选择偏差；不查询cal/holdout。推理完全保持单问题、原合法答案+U logits读出，无pair依赖、无新head/engine。

本文件只给数学可行性和一个条件性实验决策，不是训练实现、GPU授权或性能结论。CPU检查见同名JSON；既有R1/其他失败证据保留。
