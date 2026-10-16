# 固定program课程 monitored 曝光接口

CPU helper 延续同一个 program_stratified_invocations；旧首10报告023及其旧窗口保持不变。新fullpolicy SHA cd4a140bb45b0acc7dde986648a9f99b12c476bf078a9f8166791ce8e22f6399，controller实际部署 /root/autodl-tmp/run-program-stratified-monitored-full.py，SHA a80e74f2473edb20f6316721df9c208ce3e8ea89831d775778dfd416738027cf。

Controller 的 exposure_binding 固定引用022，helper另用 --continuous-base-exposure 绑定接续台账：首个monitored段使用023，后续使用上一段metadata报告。两种base分别保留，不能改写已冻结policy/launch。新policy/ADR0023/full-window/controller/old helper、前段DEV step/adapter绑定；screen不新增质量门槛，原formal质量证据不会变成通过。

支持10→20→40→60→80→100；全局100steps/1200draw固定，原config/data/source/init与AdamW/RNG完整checkpoint不变。每段final、named raw snapshot、source files、exact resume checkpoint及完整loss前缀强绑定；累计步数不相加。旧四条曝光history和计数240/240/120/120原样继承。

在所有GPU作业终态后可以重放已保存的segment receipts：

- --program-stratified-pool
- --continuous-launch-report 指向对应 launch-step0020/0040/0060/0080/0100.json
- --continuous-base-exposure 指向023或上一份metadata report.json
- --reconcile-saved-invocation 允许当前run100与旧snapshot20不同；要求当前terminal raw的config/data/init、loss前缀及完整checkpoint均属同一轨迹，记录实际current_runtime_steps_observed，绝不声称旧snapshot仍是live run。
- 中间段可加 --reuse-unchanged-inventory：只有manifest完全相同/append0/原始full-audit字节与SHA不变才复用，报告明确inventory_verification_mode=reused_unchanged_manifest。原fullaudit origin沿链绑定。
- 终态100不可使用reuse；必须fresh_full_inventory。源许可/canonical/gold/program strata/replay/ledger/shuffle及全glob unknown检测仍每次执行，metadata模式仅省去完整媒体隔离审计。

例如内部接续台账可以依序024(20)、025(40)、026(60)、027(80)，最后028(100)完整审计；实际输出编号由root冻结，helper不自动启动或续跑GPU。缺失/失败段不补造，按保存的optimizer末步计下界并拒绝越过失败的history。

CPU证据：38项相关tests/lint/format；实际7b75f9f环境对真实step20 launch绑定新policy/ADR/controller/旧源/窗口/前段DEV共10项，exit0、无inventory/GPU。真实target100总history与最终隔离仍待root终态通知后运行，不提前宣称完成或质量门槛通过。
