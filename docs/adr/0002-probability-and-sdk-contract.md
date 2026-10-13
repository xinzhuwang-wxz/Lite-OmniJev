# ADR-0002：原始概率读出与 SDK 兼容边界

状态：accepted（保留原方案定义，未改变产品目标）；证据见 docs/research/2026-10-13-feasibility.md。

使用共享官方模板生成完整prompt，vLLM `/v1/completions` 请求 `logprob_token_ids` 覆盖全部合法答案及U，`return_tokens_as_token_ids=true`，返回按ID映射。
服务端必须 `--logprobs-mode raw_logprobs`；客户端 temperature=1，不使用 allowed_token_ids/top-k 截断代替完整原始读出。缺token、NaN或HTTP -9999裁剪哨兵均拒绝，不补零。
HF/vLLM同100题最大概率差严格<1e-3才能科学放行；仅确认协议或mock HTTP不能代过。

沿用 TypeSafe 请求/响应字段形状，U和版本信息为扩展；Choice置信度遵循原设计。
官方Score confidence是距离加权，原方案统一peak confidence；当前实现保留原设计并明确扩展semantics标签，不宣称数值语义完全兼容。需要改变定义时另立决策，不悄改训练门槛。
M0仅文本接口、真实上游转发和显式未配置错误；多模态processor、鉴权/限流/计量、Playground在后续业务票实现，不能用自动API文档冒充最终产品。
