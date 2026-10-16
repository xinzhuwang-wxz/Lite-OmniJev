# 事件顺序pilot首试：训练有信号，未见原任务泛化收益

单次CE10步120draw从f2db完成，adapter831d78160b0a7e39413b53da5dd93c5c9bccef82972d511746e302d2ce426a00；原固定DEV50同33/50，但Brier .447486→.478265/ECE .104194→.155529变差、coverage .12持平。暂不扩video150或更多order数据，不推广该branch。具体训练/DEV/launch receipts同名前缀；原scope字符串继承fresh-family措辞，实际data绑定是独立程序gold关系题，不能据文案改变实验定义。

后续同12TRAIN只读诊断：固定hash12family、6yes6no，before f2db8/12→after9/12；legalCE .6237266262→.4497224937(-27.90%)，margin .1875→.8229166667，全部prepared tensor/input/gold/token映射相同。说明已见subset有学习信号但未饱和；不把TRAIN提升当推广或原DEV泛化证据。实际两arm各约24秒、90秒硬预算内；报告s4-order-train12-001。

CPU expected reward原型已经数学/梯度检查：纯正确1奖励只重加权legalCE，容易样本权重更大，不自动解决时序判断，也非proper校准；不为满足创新名义在GPU扫权重。可进一步实验的业务代价或结构奖励需有明确额外目标。

现在主线准备把小步验证与保留AdamW状态兼顾：原checkpoint有完整优化器/RNG，缺少独立invocation停点；固定大池/全局config的新CPU协议仍不得放宽resume身份。数据池1200是CPU备选/全局上限，不是已经训练1200或授权无人跑100步。质量门与24h最后6h发布缓冲保持。013实际隔离通过64已知更新/未知0。
