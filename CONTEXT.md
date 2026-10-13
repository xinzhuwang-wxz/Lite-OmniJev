# Lite-OmniJev 领域词表

- Decision：针对一个共享 state 的单步判断，不是自由文本生成。
- Primitive：Choice（选项分布）、Score（有序等级分布）、Noul（yes/no）。
- U / insufficient：证据不足的独立事件；不是一个普通业务选项。
- Answer distribution：条件于非 U 的选项概率；不能替代整体弃权概率。
- Confidence：(p_max − 1/K)/(1 − 1/K)，K≥2；不是模型正确率保证。
- Sample family：一个原始状态及所有改写、置换和反事实派生，用于隔离数据池。
- Pool：train、dev、calibration、holdout；holdout 不向训练者暴露内容。
- Gate：绑定阶段配置、来源和证据的 pass/fail/blocked 判定，缺指标不能 pass。
- Milestone：工程推进单位 M0/P0/S0…S6/FINAL；训练 Stage 0–6 不等于工程 P0–P4。
- Effective parameters：计算规模；E4B 4.5B 有效但含嵌入约 8B，E2B 2.3B 有效但约 5.1B 总量。
- Asset readiness：revision 锁定、文件完整、哈希与许可核对；不等于模型效果验证。
- Delivery：三档可复现发布物通过终局票；公开运营、付费和生产上线仍受授权边界约束。
