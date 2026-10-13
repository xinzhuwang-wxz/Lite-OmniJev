# P0 HF / vLLM 100 题概率对照

入口 `python -m scripts.p0_parity --help` 已在 CPU 实跑；CPU 行为测试只验证协议和拒绝路径。以下真实模型命令尚未执行，开卡由用户决定。必须分别保存 **base 与 adapter** 两组 HF reference / vLLM report；任何一组缺失或失败都不能宣称 P0 parity 通过，parity 本身也不等于 P0 集成完成。

## 顺序与准备

使用根训练环境的 Python 执行本入口；vLLM 使用独立服务环境，见 [service-environment.md](service-environment.md)。全部资产必须已由 manifest 校验，`MODEL_DIR` 的最后一级目录必须是固定 40 位 revision。`P0_DATA` 指向训练使用的真实 100 行程序真值 `data.jsonl`，每个样本只保留一题。拒绝媒体、holdout、重复 ID、截断、缺答案 token。

顺序：停止 vLLM → HF base reference → HF 进程退出 → vLLM base 对照 → 停止 vLLM → 十步训练并退出 → HF adapter reference → HF 进程退出 → vLLM adapter 对照。HF 与 vLLM 不同时驻留。脚本不启动或停止外部进程，操作员须确认已有 GPU 作业已退出。

```bash
# TRAIN_PY 是持久数据盘上的根训练环境 Python；这些路径须按实际实例设置。
"$TRAIN_PY" -m scripts.p0_parity hf-reference \
  --model-dir "$MODEL_DIR" --asset-manifest "$ASSET_MANIFEST" \
  --data "$P0_DATA" --report "$DATA_ROOT/reports/p0-base-hf.json"
```

成功产生 `reference-ready`（不是 parity passed）后，等待命令完全退出，按服务 runbook 启动基座服务。HF 完整 BF16、冻结所有参数、eval/inference_mode，使用训练同一官方 template 与 token 检查，显式关闭 thinking；全词表 FP32 log_softmax 后取所有合法答案及 U。默认 seed=42，其他训练配置通过 `--config` 绑定。

## 服务启动记录

创建独立 `p0-base-server.json`，结构如下；`provenance` 必须是 HF 报告里完整对象，不是这个占位内容。由操作员先核实实际启动命令、模型路径、revision、文件哈希及 adapter，然后复制该对象。不得为满足比较而把另一服务的记录改成参考报告。

```json
{
  "provenance": {"COPY_THE_COMPLETE_VERIFIED_HF_PROVENANCE_OBJECT": true},
  "served_model": "google/gemma-4-E4B-it",
  "logprobs_mode": "raw_logprobs",
  "dtype": "bfloat16",
  "launch_command": "完整实际执行的 vllm serve 命令及参数"
}
```

这是**操作员核验的启动记录**，OpenAI endpoint 不提供权重哈希远程证明。脚本检查记录与本地重新校验的资产、数据、模板、配置、adapter、实现代码哈希完全一致，并要求 `/v1/models` 广告该模型名称；模型名称检查不能证明权重身份。保留实际启动日志、独立环境版本/锁哈希与 GPU 环境证据。服务应使用 runbook 的 raw_logprobs、BF16、`--generation-config vllm`、固定 seed 等参数，监听本机环回地址。

```bash
"$TRAIN_PY" -m scripts.p0_parity compare-vllm \
  --model-dir "$MODEL_DIR" --asset-manifest "$ASSET_MANIFEST" \
  --data "$P0_DATA" --reference "$DATA_ROOT/reports/p0-base-hf.json" \
  --server-record "$DATA_ROOT/reports/p0-base-server.json" \
  --upstream-url http://127.0.0.1:8001 \
  --report "$DATA_ROOT/reports/p0-base-vllm.json"
```

完成训练并停止服务后，再执行 HF 命令，增加 `--adapter "$ADAPTER_DIR"`，报告用 `p0-adapter-hf.json`。HF 退出后按服务 runbook 注册 `decision` LoRA，启动记录使用 adapter 参考报告的 provenance、`served_model: "decision"` 和真实 LoRA 启动命令。compare 命令也必须带同一个 `--adapter`，对应 reference/server/report 均改成 `p0-adapter-*`。基座参考不能用于 adapter，变更任何输入需重新生成相应参考。

## 判定和证据

vLLM 请求直接提交 HF 使用的相同 prompt token IDs，按网关既有 `/v1/completions` 的 `logprob_token_ids` / `token_id:N` 协议索取完整答案集，`temperature=1, top_p=1, top_k=-1`。`logprobs=1` 不能作为截断 top-1 回退；缺任一指定 token、非有限值、正 logprob、≤-9999 裁剪哨兵均失败。校验 prompt token 数与真实输入一致。

每题保存全词表归一化后的原始 token logprobs/probabilities、去 U 条件答案分布和合法集联合归一化的 p(U)。门槛是所有 100 题、全部条件答案概率及 p(U) 的最大绝对差 **严格小于 1e-3**；全词表归一化的原始 token 概率最大差也必须 **严格小于 1e-3**；两个门槛同时通过才放行，不能用相同条件分布掩盖合法答案总概率质量变化。参考内每行的概率都会从 logprobs 重算核对，缺题或 token/输入哈希变化拒绝。

报告含代码 revision、实际实现哈希、模型/manifest/数据/template/配置/adapter 哈希、耗时、退出码、HF runtime/CUDA/GPU 版本；费用未知保存 null，由实际账单补充。报告只保存样本 ID 哈希和输入哈希，不保存 prompt、tokenized prompt 或业务原文。异常只记类型以免上游响应泄漏原文；结合命令退出码和受控服务日志排查。既有报告永不覆盖，失败重试使用新路径。确认两组报告通过后，继续 P0 API/浏览器/资源交接验收。
