# 依赖决策

用户授权按方案准备框架；只安装 M0 所需。具体解析版本及下载哈希在 uv.lock。
- Python 3.11 + uv/hatchling：原方案语言，统一锁定/构建；不引入第二套包管理器。
- Pydantic 2（MIT）：沿用方案的数据 schema，避免手写类型校验器。
- FastAPI（MIT）、uvicorn（BSD-3-Clause）、httpx（BSD-3-Clause）：沿用方案网关，异步上游与外部 HTTP 行为测试。
- huggingface-hub、tokenizers（Apache-2.0）、Jinja2（BSD-3-Clause）：固定模型 revision、断点下载和官方 tokenizer/template CPU 检查；不执行 remote code。
- pytest（MIT）、ruff（MIT）、mypy（MIT）：原方案的测试/静态验证。
- Playwright（Apache-2.0）：本环境无 Playwright MCP，实际 Chromium 验收 API 文档，后续复用在 Playground。

Transformers/PEFT/torch/vLLM/MLflow/Data-Juicer 等在对应票中于 Linux 环境精确锁定和 smoke 验证；macOS 的 uv.lock 不证明 CUDA 组合有效。
不复制无许可代码；新库/源码复用在对应票记来源、license、用途。
