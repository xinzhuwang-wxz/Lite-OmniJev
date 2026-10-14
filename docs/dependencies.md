# 依赖决策

用户授权按方案准备框架；只安装 M0 所需。具体解析版本及下载哈希在 uv.lock。
- Python 3.11 + uv/hatchling：原方案语言，统一锁定/构建；不引入第二套包管理器。
- Pydantic 2（MIT）：沿用方案的数据 schema，避免手写类型校验器。
- FastAPI（MIT）、uvicorn（BSD-3-Clause）、httpx（BSD-3-Clause）：沿用方案网关，异步上游与外部 HTTP 行为测试。
- huggingface-hub、tokenizers（Apache-2.0）、Jinja2（BSD-3-Clause）：固定模型 revision、断点下载和官方 tokenizer/template CPU 检查；不执行 remote code。
- pytest（MIT）、ruff（MIT）、mypy（MIT）：原方案的测试/静态验证。
- Playwright（Apache-2.0）：本环境无 Playwright MCP，实际 Chromium 验收 API 文档，后续复用在 Playground。

Transformers/PEFT/torch/vLLM/SwanLab/Data-Juicer 等在对应票中于 Linux 环境精确锁定和 smoke 验证；macOS 的 uv.lock 不证明 CUDA 组合有效。
不复制无许可代码；新库/源码复用在对应票记来源、license、用途。

## S0 原生音频HTTP依赖

真实H20服务暴露0.26基础环境未安装其官方audio extra的问题。服务锁显式启用 `vllm[audio]`，固定已缓存的PyAV15.1.0和SoundFile0.13.1，不升级Torch/Transformers。SoundFile复用官方PCM WAV解码（BSD-3-Clause，Linux wheel内libsndfile为LGPL）；PyAV复用FFmpeg解码（BSD-3-Clause wrapper，FFmpeg许可见wheel随附文件）。已有canonical PCM16 WAV必须经SoundFile得到精确float32样本；实测0.26的PyAV整数PCM fallback未归一化，不能用该fallback冒充同输入。服务重建必须包含audio extra，导入成功不足以证明真实音频HTTP可用。

## 可选训练环境
`uv sync --frozen --extra training --group dev` 安装torch、Transformers、PEFT、Accelerate、safetensors；具体版本在uv.lock。它是本地CPU损失/入口验证环境，不代表Linux CUDA兼容性已通过。safetensors用于权重与adapter，PEFT复用LoRA实现，不自建训练框架。

## 可选实验监控
用户指定 SwanLab（Apache-2.0），复用官方 SDK 记录真实指标；monitoring extra 不影响禁用监控的运行。实现与凭据边界见 ADR-0003 和 runbooks/monitoring.md。

## 无卡数据来源检查
`infra/data` 独立锁定PyArrow（Apache-2.0），用于读取下载的Parquet schema/行数，复用[官方Parquet接口](https://arrow.apache.org/docs/python/parquet.html)，不手写二进制解析器，不改变训练/服务环境。实际CLINC检查仅公开train文件。

## S0可选Qwen官方快速内核（隔离候选）

OneJev27B的公开231双轮实测647秒，日志确认Torch gated-delta fallback。仅准备一个fla-core==0.5.2候选（MIT，wheel SHA5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761），提供官方FLA/Triton gated-delta kernel，复用现有Torch2.11/Triton3.6，不引入causal-conv编译或升级模型/引擎。文件系统PYTHONPATH overlay隔离启用，primary环境未安装/改变，默认关闭。H20小tensor kernel已实测可用，非模型质量/性能证明；只有一次限额真实模型比较后才决定后续启用，并记录backend/version变化，旧分数不冒称同一数值实现。报告reports/s0-qwen-fla-preparation-001.json，官方来源与SHA在报告内。
