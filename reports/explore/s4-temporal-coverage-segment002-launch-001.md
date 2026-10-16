# 第二段父模型选择与短训练

2026-10-16。首段f2dbb83b…在固定video150从109→116（14纠正/7退化，temporal68→74），NLL .525804→.496081/Brier .351817→.329909；原cross10073→74且Brier改善，old50 audio39→38、image40/screenshot50/text49持平。依据ADR0022综合采纳为当前工作候选与下一段父模型，一题音频差异留观察，非S4晋级。yes预测79→100偏移及coverage@5%下降仍明确保留；正式完整质量/封存门未通过。

cross100原作业在100题全部保存后汇总相对路径hash失败，依赖队列拒绝继续。CPU恢复严格检查100数/IDs/全部prepared hashes与CE0相等、实际模型SHA/data/侧车/progress指标一致并重算指标；原failed001与recovered002同时保留，峰值CUDA元数据无法恢复为null，无GPU重跑。执行恢复脚本远端001保留；本地随后仅行换行修lint。后续cross入口已显式固定工作目录。old4真实完成，整体无以skip冒充通过。

第二段数据已真实CPU30.503秒/isolated准备60新video family，排除CE0 lineage350∪首段60=408，cross+old replay不变。原报告永久保持data_ready_only/CE0占位初始化。root独立launch选择首段f2db为实际init，绑定首段video/cross恢复/old终态与权重SHA；不是修改原准备证据为已获采用。data4fe42a806265af995f8f3d6f61ecf0aadfa82c63f86f64db2240ed138ce9aa5b、config3bca7236…。120draw/10steps/accum12/seed42/LR2e-5/CE/rank16/alpha32/bf16/uniform16/1120image；分段重新初始化AdamW，继续父adapter权重，明确不同于携完整optimizer state续训。

单段240秒hardcap、step5checkpoint与逐步SwanLab；成功才接固定DEV50单遍180秒，比较CE0 30与父段33及概率/yes-no分层。第二段更大切片/旧模态仅在信号支持后运行；无第三段预排。小差异按ADR0022综合判断，不修改正式质量门。

root launcher/eval/queue均存.scratch/s4，实际训练core3fe0ab2/archive9132c2d…，报告s4-temporal-coverage-{preflight,training,launch}-002；最终模型/结果待生成。原CE0与S3回退仍保留。
