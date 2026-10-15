# 时序课程概率目标短对照

2026-10-16。时序CE视频150仅109→110，初步CPU配对诊断提示概率评分退化。运动优先取帧50与均匀取帧都30正确，Brier稍差、coverage下降，停止扩量。当前工作候选保持CE0 4b8e422f…。

假设：现有联合答案分布CE加0.5 Brier能缓解短时序课程的过度自信，相同gold数据无需增加教师调用或采样。不是RL，不宣称新算法或SOTA。只运行一次，不扫描权重。

严格配对：同CE0初始化、temporal120数据9b163f5b…、ledger50e536c7…，相同seed42、LR2e-5、rank16/alpha32、10步/accum12、uniform16及bf16；仅brier_weight=.5。配置SHA6f8282138c6e0e3326c5de2e15f8d709899541f9ac0a47d5409b66598e6697cb。复用经测试的trainer3fe0ab2，源码archive9132c2d…。未经修改的原CE报告和原prep保留。

240秒训练硬限、5步检查点、真实SwanLab；成功后固定video150单遍180秒、每25题记录。评估wrapper逐项比较训练配置（只排除Brier字段）与data身份；不满足立即拒绝。先比较CE110/150及原CE0 109/150，再看NLL/Brier/错误重分配；准确率小波动不是机械裁决。只有明显收益且旧模态保留筛查通过才扩大；失败则保留当前CE0，不继续同族扫描。

执行入口 .scratch/s4/launch-temporal-brier.py 与 evaluate-temporal-brier.py；预检已通过，运行中不计通过。远端launcher-001执行版本与本地只差首行docstring缩短（lint修正，无行为变化）。完整S4质量门、calibration/holdout及最终发布均未完成。
