# ADR-0007：恢复缓存验收的原方案测量口径

状态：2026-10-14主会话按用户“总体按方案、自主安排调节/决策、尽量少GPU时间”的技术授权采纳。原HTML保留，不更换基座/引擎，不改变≤2和严格<1e-3门槛。独立architect复核原文与本地实现，确认以下额外约束不是原方案要求。

## 问题与实测

原HTML第六节S0④要求：同一张图10道题总时延≤单题2倍，vLLM原生前缀缓存生效。S0③要求原始概率差<1e-3。本地04票和runner额外把所有cold/warm paired比率都作为≤2硬门槛，把跨cache-on/off原始logprob零差作为通过条件。

实际H20/FP32 IEEE eager命中288tokens，覆盖媒体前缀265；两个cold比率3.9318/3.9354，warm14.5437/14.8598。网关cold5.0202/5.0846。on/off跨路径raw-logprob最大差4.86374e-5。旧strict报告failed完整保留，当前cold性能仍不通过，本ADR不能使这些失败变为成功。

warm单题的primer是完全相同的第1题，所以完整question也命中；warm十题的后9题是新尾巴，仅媒体/共享文本命中。该比率测量“9个新尾巴相对于完整重复题”的成本，不能等价为原方案同图请求的增量效益。原概率门槛也不是跨不同KV计算路径逐位raw-logprob零差。

## 决定

- 保留默认`strict-sequential-v1`评估和全部历史结果；新评估必须显式命名`original-product-cold-v2`并输出新报告，不覆盖、不声称旧strict通过。
- 新主性能判定：每个cache-on冷态paired repeat的完整同图十题wall/单题wall≤2。单题和十题均独立reset至相同空KV/媒体状态，预热置于计时外；绝不混用warm十题/cold单题。当前仍保持严格顺序，不能用并发结果冒充顺序。
- cold/warm、on/off、每次请求/总wall、真实命中tokens、网关/预处理边界全量报告；warm与cache-off为诊断对照。命中未覆盖媒体、缺题、缺重复、错输入/版本或无效计时仍失败。
- 新数值判定从完整可信原始logprobs重算raw token probabilities、joint K+U、conditional答案和p(U)，四类最大差各严格<1e-3；等于阈值也失败。旧protocol的零raw-logprob容差字段不改写，新policy显式记录所用概率阈值及所有delta。
- 本次只纠正多加的测量条件，并使用原生CUDAgraph等既有运行时能力优化。若严格顺序仍不能达到原≤2，保留失败；有界多题调度须单独规格/报告，不能倒称已通过顺序验收。

## 验证与边界

旧strict结果保持失败；新policy必须有公共行为回归：暖态诊断不放行冷态失败；任意四类概率差≥1e-3失败；完整输入/命中/重复检查不减少。新policy可离线读取原始measurement重新计算，避免重复GPU。任何后续正式配置仍需真实GPU/网关证据，不以CPU或命中量替代2倍性能通过。S0整体未通过、S1不启动。
