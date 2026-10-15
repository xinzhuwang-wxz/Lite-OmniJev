# S4 受控claim成对监督短对照

状态：两arm已真实完成；pair目标不采用，不扫权重，保留CE数据更新作为工作候选。用户授权小步创新、机时优先；不放量、不改变最终门槛。

证据：coverage0deef相较S3完整视频提升3.8pp，原cross100相较首轮少1题（72/73），补充cross50多1题（41/40），旧四模态短测保留。先前TRAIN构建每个组件只出现一次，缺少同组件受控正负对；新构建200group/600变体仍待最新全库存隔离。完整instruction因claim改变，不声称完全相同语义下标签翻转。

假设：在同一图像/音频组件上对正确claim和单侧错误claim加入相对margin，可能比独立CE更有效地训练联合判别。该约束没有新增gold信息，是结构化监督，不是RL，不预称新颖。

固定两arm：共同初始化coverage adapter0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818；同seed42、同120实际draw、10步/accum12及原LR2e-5/rank16/alpha32/BF16/1120图像/16视频帧。每步4cross（两个正负对）、4video、audio/image/screenshot/text各1；20不同pair group，两类false各10组，按固定hash排序；其他来自既有TRAIN准备002，不借dev题训练。第5步保存检查点，逐步SwanLab与本地耗时/CE/梯度/显存，预算每arm240秒。

CE arm使用原损失。Pair arm增加0.1×softplus(1−(s_positive−s_negative))，s=logit_yes−logit_no，每对与两例CE一起除accum12反向；固定权重/边距不扫参。两例共享component/family、不同claim，官方答案映射/全部合法答案含U的CE不变。pair侧car须绑定canonical samples、prepared输入、组件、顺序、gold、配置、ledger。无分类头/引擎/基座改动，无额外教师标签。

先CPU身份/全库存隔离/公式梯度与legacy兼容验证，再GPU。任何非有限值、错误身份/池、OOM或超预算立即停止并留证据。实际CPU准备或训练失败不冒称已完成。

评估优先固定original cross100+supplemental cross50，复用已有初始模型证据，报告配对纠正/退化与NLL（若已实现）、Brier/ECE、资源；有方向性收益再检查video150/旧模态50。小样本的一题差异由负责人结合多个来源、概率损失和成本判断，不为凑单题扫参。只有证据支持才扩大，完整S4验收仍pending。

结果：两arm在原cross100、补充cross50分别同为73/100、43/50，逐题预测完全相同；pair原来源NLL/Brier略差、补充略好，无一致收益，显存56.94GB高于CE约36.9GB。CE短保留video109/150（父108），old4持平。46份实际更新报告全部绑定，库存isolated。既有S4正式门仍未过；不宣称RL或新目标有效。证据见s4-claim-pair-{original100,supplemental50}-comparison-001与s4-claim-ce-retention-001。
