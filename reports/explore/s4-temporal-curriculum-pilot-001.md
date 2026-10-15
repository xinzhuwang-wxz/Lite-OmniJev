# S4 有界时序课程更新

2026-10-16，CPU准备通过后启动一次10步试验，未通过S4正式门。

依据：coverage完整video500的163错误有122落在既有词匹配时序组；时序AUC .7253，开发集oracle阈值仅能把207/329提高到218/329，不能将偏置修正当作主要解法。新工作候选CE0在固定video150为109（时序68/97）、cross10073/supp50 43、旧模态保持。以该CE0为唯一初始化4b8e422f…，只做CE，不使用已停止的pair目标。

固定120draw、10steps、accum12、seed42、LR2e-5、rank16/alpha32、bf16、官方模板、1120图像预算与uniform16视频帧；第5步存检查点。每step6video+2cross+4old。video60取自已有完整8299已准入TRAIN：50temporal（25yes25no）、5motion（2yes3no）、5existence（3yes2no），60唯一family；不宣称未见。cross10组正负claim共20条，audio/image/screenshot/text各10条。实际数据/ledger/配置/init与既有准入追溯在artifacts/s4-temporal-curriculum-pilot-001/report.json；完整库存isolated。

这是一个整体课程策略更新，视频份额、时序比例与样本覆盖共同变化，不把结果归因为单独一项。固定配额不扫比例，不修改old candidate或gold。原builder的training_admitted=false是通用builder不授准入标记，保留原字段；实际适用范围由ADR0005/S3全8299来源、21184联合TRAIN与ledger绑定证明，不靠本轮口头授权替代。

训练硬上限240秒，NaN/Inf/身份或池错误/显存错误立即停止；逐步查看耗时、损失、已见样本和显存，第5步有检查点。训练后优先固定video150单遍（180秒），比较整体及三启发式分组、概率损失；若无有用信号停止，不追加完整500。若有收益，再短查跨模态与旧模态保留；小样本一题差异按ADR0022综合判断，不改最终S4门。与独立motion16推理试验不混合。
