# 现成 gold 与私有评测作业

首发合同以 ADR0012 为准：使用许可明确的原始数据集 gold 或可验证程序真值，不要求新增真人双标、κ或用户提供外部 gate 账号。κ 为 `not_applicable`。数据准入、模型执行、阶段质量是三个独立结果。

## 当前数据与证据

- `.scratch/s4/selected-gold-manifest-001.json` 冻结六个 cohort 各 3000 calibration、1000 holdout，以及各自原始 500 dev；补充 dev 单列。保留的旧 AOK/video 储备不重复计入主配额。
- `reports/s4-gold-data-admission-001.json` 绑定来源/代码/输出、原始题映射、媒体 URI 整理、已知训练与开发曝光、RGB 隔离及真实权限/失败探针。改变选集、输入或相关证据后须重新验证绑定，不能只修改一个 success 字段。
- 问题数量不等于独立家族数量。保留组件依赖组与重复使用统计，不宣称 IID 或未估计的有效样本量。
- `eval/gold_admission.py` 只验证结构，不能独自认证来源。实际数据准入由已执行作业及逐个工件哈希核验支持。

## 数据审计入口

`scripts.private_gold_job` 只接受固定 `inventory-audit` 作业。Linux root supervisor 验证并复制已绑定的输入，再以未使用的普通数字 UID 执行审计；输入只读，逐题数据和日志不公开。

```bash
python -m scripts.private_gold_job --job inventory-audit \
  --inventory /absolute/inventory.json --sha256 INVENTORY_SHA256 \
  --work-root /var/tmp/omnijev-gold-jobs-001 \
  --report /absolute/aggregate-report.json \
  --worker-uid FRESH_UID --probe-uid DIFFERENT_FRESH_UID --timeout 300
```

这是参数模板，实际 UID 使用状态见 `plan/STATUS.md`。固定使用同一 work root；每次尝试永久占用其 UID/GID lease，失败也不要复用。作业结束后 run 根目录撤销为 root-only，数据保留。timeout 指 audit child，不是全部快照准备时间。

此边界已通过普通身份访问拒绝、UID 复用拒绝、日志/异常/超时探针。它不防 host root，不强制网络或 namespace 隔离，也不等于外部安全证明。

## 模型评测入口

`scripts.private_stage_eval` 使用固定 HF Gemma 4 E4B + 可选冻结决策 LoRA；官方模板关闭 thinking，检查答案单 token 边界，从完整词表 log-softmax 提取原始答案概率。当前实现不能冒充 E2B、校准模型或三运行时发布验收。

```bash
python -m scripts.private_stage_eval run \
  --spec /absolute/hash-bound-job-spec.json \
  --report /absolute/aggregate-model-report.json
```

Spec 的精确字段由 `scripts/private_stage_eval.py:Spec` 定义。必须绑定全部可加载权重分片、配置、tokenizer/template、adapter、数据、媒体根、seed、图像/视频预算和超时；模型及 adapter 仅本地加载。

- `engineering` 只读 dev，可固定 seed42 抽取至多 50 题。它验证实际工程路径，不通过阶段质量门。未指定子集时数据本身也不能超过 50 题。
- `S4`/`S5`/`S6` 作业读取完整指定 calibration/holdout 工件，禁止子集。除逐工件 `Admission` 外，必须提供实际 `selected_manifest`、全局 `data_admission` 和 `evidence_root`。
- 阶段入口逐个核验固定 11 份证据文件、全局 evidence digest、原始 dev/配额、库存/代码绑定，以及当前 data 在主选集中的 cohort/pool/hash/数量；辅助储备或自声明 `status=admitted` 不能充数。
- 本地 `file://` 媒体仅在显式 media root 内验证后于内存中转换为相对路径；拒绝 host/query/fragment、越界、符号链接、遍历或错误 SHA。原 JSONL 不改写，报告绑定 canonicalized input digest。
- 同池不同原题可以共享 family；跨池隔离仍由已准入全历史图验证，报告不声称 IID。

Supervisor 启动固定受信任私有 subprocess，并清除凭据/追踪器环境，强制离线加载。run 目录 0700，spec/log/逐题 records 0600。该模型作业是 **root-visible trusted process**，与上面的非 root UID 数据审计是不同执行边界；不宣称网络、namespace 或对 root 的隔离。

公开 `Receipt` 只包含聚合指标及模型/输入/配置/代码/运行时绑定。`execution_status=complete` 与 `quality_status=not_evaluated` 可同时成立。阶段质量须另用同条件教师、来源分层和完整阈值判断。wall time 包含校验/加载/处理；不能作为常驻服务延迟。未知价格保持 `cost_usd=null`。

后端异常、非零退出、缺失/部分/不匹配 receipt、超时和受控 TERM/HUP 中断均拒绝，保存静态类别失败报告，不导出 worker traceback、逐题答案或 logits。TERM/HUP 会回收 worker 并恢复 handler；SIGKILL 不在可恢复保证内。报告不覆盖，重试用新报告名，失败工件保留。

## 已验证范围与接续

- `reports/s4-private-inventory-job-003.json`：实际最新清单的独立 UID 数据审计。
- `reports/s4-private-failure-probes-002.json`：实际 Linux 数据审计失败/日志探针；001 的测试脚本错误保留。
- `reports/s4-private-stage-engineering-001.json`：真实 H20、50 dev 题通过模型入口；重复性以独立 repeatability 报告为准。
- 模型入口 56 项 CPU 测试在本机及租机通过，含真实 TERM/HUP subprocess 回收；尚无完整 S4/5/6 模型质量放行。

开发 agent 不读取封存样本、逐题答案或私有日志。数据构建与 gate 作业只向其返回汇总。开发集可诊断，但不得用开发/已训练样本替换封存集；正式 holdout 只按阶段协议使用。

## 历史工具

`eval.sealed_gate` / `scripts.sealed_gate` 的 v2 双标 canary 保留兼容用途，不能作为当前首发现成 gold gate。其历史 κ、双标身份和外部 gate 要求已由 ADR0012 的首发合同取代；不要运行它来证明上述实际数据或模型验收。
