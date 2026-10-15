# S4 单adapter参考保留候选（冻结试验）

触发：video pilot第二轮video99→113/150，但原crossmodal35→33/50；两融合未通过保留门。原生路由BF16在100题probability parity失败，当前路线不推广。保留首轮6bb86b09…。

假设：在与第二轮完全相同的300个TRAIN draw/30步和初始化上，添加正确旧任务回放参考约束，可减轻遗忘而保留新视频能力。仅改变损失，数据/抽样/学习率/预算/媒体处理保持第二轮配置。

目标：L=gold CE+0.5*mask*KL(qref||p)。qref为固定首轮adapter的同输入合法答案集合（含U）分布。mask仅非video且属于首轮TRAIN replay、参考唯一argmax正确时为1；video/旧参考错误/并列为0。gold原样保存，不以参考替代真值。这是监督保留约束，不称RL或原创/SOTA。

实施：scripts/train_reference.py离线生成300draw侧文件；train/trainer.py可选项默认关闭。侧文件绑定TRAIN数据、ledger、实际prepared张量、官方模板答案顺序、reference权重/config、输入资产和draw序列。首轮reference与训练不同时驻留GPU，保持单次训练forward。CPU已有91回归（代理）和主树26针对验证；尚未真实预计算/训练。

预算：仅一组lambda=0.5，不扫参；reference预计算≤240秒、训练≤480秒、第一轮筛选≤300秒。训练保持每步记录gold CE/KL/启用count，checkpoint15/30；NaN/Inf/契约失败立即停。先video固定150目标≥102且temporal≥60/motion≥7，再原cross固定50≥35、补充≥40，audio≥37，其余旧3不退。失败即止，不扩完整dev，不重用封存反馈。成功才完整dev/教师/独立holdout，不把小筛选当S4门。

本候选需先完成真实CPU preflight及sidecar生成核验；入口实现通过不等于已训练。所有新增实际更新报告必须追加曝光库存；原数据池保持不变。

## 实测结论：停止推广
reference预计算300draw通过200.16秒，纯text mapping预检bug修复后完整CPU预检111.01秒通过；训练30步439.06秒完成，139个正确TRAIN replay实际启用KL，gold CE均值.352264、KL均值.002654。video113/150通过，但原cross33/50不满足35门；与普通video CE逐题预测完全相同（首轮对照纠正2/退化4，no→yes从10增14）。停止此λ候选，不跑补充及后续，不以argmax/损失改善替代阶段门。完整报告和sidecar哈希保留。
