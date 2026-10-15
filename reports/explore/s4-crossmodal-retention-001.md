# S4 跨模态反例覆盖候选

状态：30步/360draw训练已完成（449.90秒），固定筛选运行中，尚无质量通过结论。源证据s4-reference-retention-diagnostic-001.json：两次视频训练video113/150，但原cross35→33/50；no→yes错误10→14，reference KL与CE预测完全相同。当前cross replay30条为17yes/13no。此证据支持检验覆盖不足/yes偏向假设，不证明全部错误由样本配比造成。

固定试验：初始化仍为保留首轮6bb86b09…，复用第二轮原300TRAIN draw；增加60条已准入crossmodal-train-001原题（20both_true、20image_false_audio_true、20image_true_audio_false），不改gold/组件/媒体。排除此前实际draw已见cross，按固定hash排序；任一quota不足即阻断，不放宽。源池/组件ledger/隔离审计通过后，按seed42反向布局保证每步原10draw+新增2cross，总360/30steps/accum12，其余配置同第二轮，CE，不加KL。增加跨模态权重是唯一主要变量；不声称新增样本独立于其组件。

上限：一个候选，训练600秒，不扫参数；video150≥102且temporal≥60/motion≥7→原cross50≥35→补充≥40→audio≥37→image40/screenshot50/text49，任一失败即停。相同固定dev，不查询封存。通过小筛选才允许完整dev与既定阶段门；不降质量门槛。新更新报告与runtime data必须重新绑定曝光库存。
