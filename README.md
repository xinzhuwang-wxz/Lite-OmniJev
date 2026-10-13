# Lite-OmniJev

轻量多模态单步决策模型的可复现工程。以 Gemma 4 E4B/E2B、决策LoRA和原词表概率读出为路线。

**当前阶段：M0 本地准备已通过，P0 等待无卡实例。没有已训练模型、GPU基线或可用的生产服务；CPU夹具不代表模型效果。**

```sh
uv sync --frozen --group dev
make check
make smoke
make dev
```

打开 http://127.0.0.1:8000/docs 。真实决策需要本地官方tokenizer和可访问的vLLM，配置见 `.env.example`（由shell注入，应用不自动加载该文件）。未配置时接口明确503。
GPU/训练环境在实际Linux实例单独锁定，不能拿macOS的CPU检查代替。

- [执行与验收](docs/02-project.md) · [当前状态](plan/STATUS.md)
- [循环规则](plan/LOOP.md) · [一屏启动词](plan/START.md)
- [原始设计](docs/Lite-OmniJev设计方案-最终版.html) · [上游核验](docs/research/2026-10-13-feasibility.md)
- [资产准备](docs/runbooks/assets.md) · [无卡/开卡交接](docs/runbooks/compute-handoff.md)

票据在 `.scratch/`，默认不向GitHub写入；大模型/数据留在忽略的 `artifacts/` 或持久外部盘。
