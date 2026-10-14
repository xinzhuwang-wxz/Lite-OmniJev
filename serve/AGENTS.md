# 网关约定
- top-k缺token会制造错误分布 → sampled logprobs不是完整raw logprobs → 请求指定token IDs并拒绝缺失/非有限/正值/≤-9999裁剪结果。
- 描述首尾空格可绕过API去重 → core会strip后拒绝 → 请求层按同样规范校验并返回422，不能变成500。
- Score legend被包成对象会破坏SDK语义 → 原值本身就是Entry → 保留调用方criteria值，confidence差异按ADR明确标识。
- max_tokens=1仍有后端生成成本 → 产品output_tokens=0只表示不返回生成文本 → backend_usage记录真实上游计量。
- 已准备对象内dict仍可变 → 合法token ID交换会错贴答案概率 → HTTP前校验答案映射与准备hash绑定，和媒体payload一起拒绝篡改。
