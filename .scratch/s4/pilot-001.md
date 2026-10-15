# S4 联合模态首轮试验（冻结前置）

状态：CPU准备，未训练。s0:05工程已通过；S4质量未通过。

假设：在S3 adapter上加入由两个原始train组件组成的AND真值，同时回放五模态，可改善跨模态组合判断，且不明显破坏单模态能力。唯一主要干预是训练混合中增加crossmodal监督；不同时改变loss、LoRA结构或推理预算。这里只判断候选是否值得追加，不能据单臂试验声称因果归因或SOTA。

数据：S3固定1000条回放，每旧cohort200；新crossmodal200，100双真/50图像假/50音频假。新train SHA d3b0be2adeadbd12cbfc01daa5fa6ef74fb687b8fab6148bc750d695bbab6977，库存SHA244a5c4f8e7dc0ae769d0b3c73b781f86c0c30aaa3f283f256d440d6c26b95ae。组合来源均train，保留family/media组件账本；不得使用cal/hold。

初始adapter：S3 stable SHA1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755。沿用冻结E4B、LoRA rank16/alpha32、CE、bf16、seed42、LR2e-5、accum10、图像1120、视频16帧。1200只是池大小，首轮30步实际300次样本曝光，不能写为全量一遍。实际shuffle后的六类曝光以日志为准，不假称每300条恰好均衡。

预算：首轮最多30optimizer步、外层墙钟1800秒；结束后回收模型并评测。现有midway只保存不暂停，配置严格相等才允许optimizer恢复。因此不预设30→60原状态续跑；追加候选的初始化/优化器状态需另行明确记录。OOM/NaN/非有限梯度/数据契约失败立即停，留失败报告。

开发筛选：复用已固定的原crossmodal50、补充crossmodal50及S3旧五模态筛选输入，先验证原基线报告/数据/sidecar哈希，再绑定候选相同输入和推理配置。原crossmodal基线33/50，补充40/50；两个来源分别报告，不混成正式总分。第一轮至少一个crossmodal切片净增≥2题且另一个不下降；旧五模态每切片最多下降1题，video150最多下降3题。仅作筛选信号，不是置信区间或阶段门。无收益停止该候选，不机械放大；单模态退化时先诊断回放/曝光，不查holdout逐题。

监控：逐步loss/lr/梯度范数、实际曝光、token、准备/step耗时、CUDA峰值；仅传数值至SwanLab。完整dev只给通过筛选的候选。正式S4仍逐项满足configs/gates.json，不以本试验退出规则降门槛。
