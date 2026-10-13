# P0 工程闭环验收

2026-10-14：按原始P0工程冒烟出口通过，S0严格概率一致性未通过，S1不放行。阶段归属审计见p0-original-contract-audit.md；没有降低阈值，不声明原方案全局CI/模型质量验收完成。

- 真实100样本/10步，仅34,881,536 LoRA参数训练、主干冻结；耗时67.165秒、显存峰值17,296,830,976字节：gpu-train-smoke.json。
- step5完整恢复至10，loss和最终adapter逐位一致：p0-resume-equivalence.json。
- 模板、答案token、权重、数据与adapter身份绑定；真实HF/vLLM差异完整保存：p0-*-hf-001.json、confirmation-*.json。
- 最终FP16/TRITON真实网关Chromium验收退出码0，Choice/Noul/Score/不足证据返回200、非法输入422，无页面错误/失败请求：p0-browser-final-triton-001.json。截图已查看；trace/截图位于artifacts/browser/p0-final-triton-001。API明确model_quality_verified=false、校准未拟合。
- checkpoint、优化器/RNG和adapter远程持久盘及本地双备份约561MiB，hash通过：p0-gpu-backup.json。
- 前置代码验证175 tests、Ruff/format/mypy/wheel/sdist通过；本次文档/状态另跑队列校验。

资源快照：H20 97871MiB，服务占29493MiB，GPU利用率0%；800GiB数据盘已用60GiB，可用741GiB。现无训练，自建vLLM9031/网关10404操作前重新核实。用户要求保持开卡，替代旧关卡通知要求；不关机、不更改租赁，保留权重/缓存/产物供S0复用。

限制：未完成教师复现、多模态正式parity、人工一致性、质量/性能和端侧交付。数值风险见p0-confirmation-review.md，归入S0:03；不以阶段收口掩盖失败。
