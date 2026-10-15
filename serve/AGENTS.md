# 网关约定
- top-k缺token会制造错误分布 → sampled logprobs不是完整raw logprobs → 请求指定token IDs并拒绝缺失/非有限/正值/≤-9999裁剪结果。
- 描述首尾空格可绕过API去重 → core会strip后拒绝 → 请求层按同样规范校验并返回422，不能变成500。
- Score legend被包成对象会破坏SDK语义 → 原值本身就是Entry → 保留调用方criteria值，confidence差异按ADR明确标识。
- max_tokens=1仍有后端生成成本 → 产品output_tokens=0只表示不返回生成文本 → backend_usage记录真实上游计量。
- 已准备对象内dict仍可变 → 合法token ID交换会错贴答案概率 → HTTP前校验答案映射与准备hash绑定，和媒体payload一起拒绝篡改。
- 基础vLLM导入不证明音频HTTP可用 → 官方audio extra另含解码依赖，0.26 PyAV整数PCM fallback未归一化 → 服务锁包含audio extra，canonical WAV须由SoundFile精确解码并实跑HTTP。
- Gemma4仅设全局FP32可使音频塔混合dtype → native AutoModel读取子配置仍为BF16 → 精度诊断使用官方hf-config-path副本同步四处dtype，记录diff/hash，保留原checkpoint配置和权重。
- vLLM0.26原生LoRA shrink只接受FP16/BF16 → FP32参考不能直接挂LoRA → 独立合并部署副本保留基座/adapter哈希与数值对照；HF导出会省共享k_norm别名，按原权重补齐，不能改引擎或换阈值。
- 服务启动缺ninja但环境已安装 → 仅用绝对vllm入口未设置编译子进程PATH → 显式将服务venv/bin加入PATH，不重复安装依赖。
