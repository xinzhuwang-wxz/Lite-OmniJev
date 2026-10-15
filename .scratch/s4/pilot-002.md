# S4 视频定向候选（第二轮，运行前冻结）

假设：当前候选的视频学习覆盖不足；首轮视频50次抽取全部来自S3 stable已见200条。开发150诊断显示temporal60/97不变、motion10→7/14、existence29→32/39。下一次改变视频训练课程，保留其它模态回放，检查新视频监督能否改善时序/运动判断。主干、LoRA、loss、学习率、16帧和1120预算不变。不把这项小实验包装为已证明机制或创新成果。

初始化：首轮S4 adapter SHA6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac；新optimizer，非原状态resume。训练30步×accum10=300次曝光，最多1800秒；第15步保存中间adapter和checkpoint，NaN/OOM/非有限梯度/数据契约失败停止。

数据：只用原可靠train池。150新视频，排除当前S3 stable视频200条的id/family/media，再在新集合中保持family/media唯一；按粗词规则temporal90、motion30、existence30，各组yes/no均分。新是相对当前adapter训练谱系，不能声称所有已废弃历史候选从未见过。其余text/image/screenshot/audio/crossmodal各30，从首轮固定train池原样抽取。全库存审计后才上卡，样本/数据/配置/来源/adapter SHA冻结。固定次序每step5视频+其他五类各1，不以开发正确性选训练样本。

比较：复用同一视频dev150、音频dev50及其它固定切片。第15步adapter保留，可在必要时诊断，不把保存说成自动暂停；本次最多30步结束后再决定继续。视频首先比较S3/首轮99/150；需至少102/150，temporal不得少于60/97、motion不得少于7/14，音频至少37/50。通过才执行剩余保留筛选：image≥39、screenshot≥49、text≥48；原crossmodal≥35/50、补充≥40/50。未达到则不扩量，保留首轮候选，使用开发汇总诊断下一假设。

这些是预先设定的小筛选规则，不能代替S4最强教师比率、完整dev/holdout、外部榜，也不降低任何正式质量门。资源耗时/显存/真实逐步数值和失败证据均记录，价格未知则cost=null。
