# OpenAI Decisions 对照准备（2026-10-16）

状态：用户2026-10-16明确暂缓此项，继续训练主线。官方资料已核验；未发送 API 请求、未产生 API 费用、没有实测成绩。

## 来源与事实
- https://developers.openai.com/api/docs/guides/decisions （实际读取官方全文）
- https://developers.openai.com/api/reference/resources/decisions/methods/create （实际读取官方接口）
- Public beta；POST /v1/decisions；当前仅 gpt-6-luna。
- predicate 返回真概率；choice 返回选项及分布；score 返回等级分布及索引期望。支持共享输入多问题。
- 文本/图片输入；参考页明确不支持音频。视频抽帧只能算图像序列代理，不算原生视频/音频对照。
- 官方称相对 Responses 约10倍快，不是本项目实测或相对 Jev 的结果。标准输入价 $0.10/百万token，区域/长上下文规则另计。
- 图片 URL 支持在 guide 与 reference 中存在冲突；后续探针用二者共同支持的内联base64。
- 文档未证明返回概率等于原始未变换全词表概率，也不能将 confidence 直接等同我们项目的 p(U) 或校准指标。

## 权限状态
本轮仅检查当前进程 OPENAI_API_KEY / OPENAI_BASE_URL 是否存在，均否；未读取或展示密钥。工具列表无专用API凭据提供工具。Codex模型/子代理权限不等于 /v1/decisions 调用权限，不挪用客户端内部凭据。尚不能真实评测；需要用户配置有权限的API凭据及小额费用上限。

## 冻结的小试设计
凭据和预算可用后，先3条自有文本验证 predicate/choice/score、响应结构、usage、拒绝路径；随后复用公开开发固定文本/图像/截图各50题（每个cohort先10题，正常才扩展）。不读取封存内容，不发送私有原始数据。记录请求输入哈希、模型、返回概率、HTTP状态、usage、冷/热缓存、串行端到端p50/p95、准确率、Brier/ECE与实际费用。与本地候选使用相同题和媒体，单独注明网络和图像预处理差异。小样本校准只作诊断。音频及图像+音频标不支持；抽帧实验另报，不能混入原生全模态总分。

此对照作为s4:01补充研究，不替换冻结基座或既定质量门；缺账号不阻塞原计划本地GPU实验。
