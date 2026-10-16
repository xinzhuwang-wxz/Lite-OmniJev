# Continuous optimizer：现成resume完整，缺安全invocation停止接口

**固定较大池和全局config可以保留AdamW/RNG续训；现代码不能主动每10step安全返回。** 不应继续把新adapter当init来模拟resume：`train_smoke`每次新建AdamW，只有`resume_from`分支才恢复状态。此诊断只读源码/CPU现有测试，不改trainer、不GPU；cost=null。

现成事实：

- `train/checkpoint.py:124`保存全部trainable tensors、完整optimizer state_dict（含AdamW动量/step/paramgroups）、Python RNG、torch CPU与全部CUDA RNG、全局step/losses；tensor文件带SHA，临时目录完成后atomic rename。当前无scheduler/GradScaler状态需要另存。
- `train/checkpoint.py:216`、`:246`严格校验config/data/model/assets/order以及文件hash，拒绝别的output目录和旧于latest的checkpoint。init_adapter身份也写入config；续训必须原样重传，不能改dataset/maxsteps/config偷偷绕过。
- `train/trainer.py:1213`创建AdamW后恢复model/optimizer/RNG；`:1265`从checkpoint全局step继续。draw为`step*accumulation+offset`，再按固定seed顺序取`order[draw % N]`（`:1278`）。固定1200样本、100steps×12只走一遍；10step停后再resume应从draw120继续。若总draw>N则按现有语义循环，不能称从不重复。
- `scripts/train_s1.py:95`已有`--resume-from`；`LoopConfig`允许大池/大steps。不存在stop-after/invocation_steps/协作wall-budget flag；循环必跑至全局config.steps。checkpoint_every控制存盘，不控制退出；midway仅导出adapter，不能恢复optimizer。
- 外部超时/KeyboardInterrupt只能恢复最近完整checkpoint，不是安全分段：checkpoint_every=5时末尾未存盘的1–4次更新会回滚并重新处理其draw；即便每步保存，强杀落在optimizer.step到commit之间也有窗口。不建议SIGSTOP、双GPU进程或无人跑满长作业。
- 现有CE确为`decision_loss`的selected legal+U log_softmax CE（`train/trainer.py:663`），不是full-vocab softmax CE。此次仅核实，不改变训练目标。

最小待实现接口（非现成能力）：给`run/train_smoke`和S1 CLI加独立`--invocation-steps N`，作为执行控制参数，**不进入全局TrainConfig/checkpoint config dump**。全局steps/dataSHA/order及init身份始终immutable；这不是允许续训时换data。每次终点为`min(resume_step+N, global_steps)`；仅完成完整accumulation+optimizer step后停止，最后一步无论checkpoint_every是否整除都强制完整checkpoint；成功commit后导出对应step评测adapter并返回`paused`/exit0，只有达到global_steps才`completed`。resume必须latest，沿用所有现有校验；不重置optimizer或RNG。默认None保持原行为和旧checkpoint序列化。

若还要秒级invocation预算，应是协作deadline：检查下一步前的剩余预算，预留step+checkpoint/adapter flush时间；在完整step边界强制commit后返回。单step不可预测耗时意味着不能同时保证精确硬墙钟上限与绝不丢未commit更新；外层硬timeout仍须有flush余量。预算在首个更新前耗尽应明确`no_update/no_resumable_checkpoint`，不能伪称paused已有checkpoint（当前save_checkpoint要求step>0）。先实现step上限即可避免引入复杂暂停框架。

报告最少区分global_steps_completed/target、invocation_steps_completed、last_committed_checkpoint/adapterSHA、stop_reason、next_draw；保留旧全局losses和每次invocation receipt，勿把paused标completed或把累计曝光重算为新曝光。后续新数据段要开新immutable run，不能以放宽resume绑定保AdamW跨不同数据身份。

应有窄CPU验收：现成toy AdamW连续 vs pause+exactresume的权重/loss/动量/step及Python/torch随机下一值一致；固定1200索引分10step后拼接等于连续100step、边界不重复；暂停点不整除checkpoint_every仍强制保存且只保存一次；最后global step标completed、早停标paused；无预算参数旧dump/行为不变；换data/config/init/assets/order、旧checkpoint拒绝且原receipt不被覆盖；不足一个step/异常退出只报告持久化边界。真实GPU/dropout等价性与每次模型加载成本仍需一次有界实测，CPU不能替代。

本轮复用现有`tests/test_training.py`的checkpoint精确CPU续训与provenance拒绝测试；未新增流程/代码。是否选择100steps/1200draw及下一段门槛由root根据剩余时间/现有评测决定，本报告不授权自动跑满。
