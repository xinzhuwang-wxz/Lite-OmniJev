# P0 真实浏览器验收

先完成真实 GPU 训练、checkpoint 重载及 HF/vLLM 概率对照，再启动配置模型的文本网关并转发到本机。此脚本只访问 localhost，不连接远程实例、不读取凭据、不启动服务。`--expected-model-version` 必须取真实网关配置的 `LITE_MODEL_VERSION`，不能用预期结果替代响应。

```sh
uv run python -m scripts.browser_p0 --help
uv run python -m scripts.browser_p0 \
  --base-url http://127.0.0.1:8000 \
  --expected-model-version '<实际 LITE_MODEL_VERSION>' \
  --output reports/p0-browser-<唯一运行标识>
```

输出目录必须尚不存在。需要已安装的 Playwright Chromium；Python 包在 dev 依赖中。当前使用实际 Chromium，报告注明未使用 Playwright MCP。准备阶段仅验证 CLI/静态检查；运行是否通过以独立运行目录的 `report.json` 与退出码为准。

脚本实际打开 Swagger `/docs`，点选 health 的 Try it out / Execute，要求已配置推理与模型版本匹配。随后在同一浏览器页面执行真实 fetch，提交 Choice、Noul、Score 及证据不足场景，检查答案标签、去 U 后概率归一化、概率有限且有界、独立 p(U)、版本与 `calibration=unfitted`。证据不足场景只验证 p(U) 契约，不要求十步模型作出正确弃权；没有名为 U 的独立 primitive。再检查单选项非法 criteria 与坏 JSON 均返回 422。

每次保存 `report.json`、`swagger-health.png`、`final.png`、`trace.zip`，失败时尽量保留已收集证据并退出 1。报告记录模型响应、浏览器版本、代码 commit、环境、耗时、页面异常及失败网络请求。预期 422 可能产生 console error，这些原文保留但不单独判失败。截图与 trace 仅包含本脚本的合成文本，不使用业务数据或封存样本。

此验收不证明准确率、校准质量、服务时延门槛、GPU parity 或最终 Playground 通过；对应真实证据仍须由各自验收提供。
