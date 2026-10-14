# 固定资产的CPU支持矩阵（S0:10）

实际Linux004：源码aa94581，/root/autodl-tmp/s0-baseline-cpu-004，32.917秒exit0；reports/s0-baseline-cpu-linux-004.json包含全部metadata size/hash、官方模板/输入/输出tensor摘要、各类别声明证据文件hash及字段。无权重加载/模型forward/GPU/质量准入。七资产对应五核心和Valen+Qwen依赖，固定revision/许可沿用configs/project-model-assets.json。

| 主体 | text | image | screenshot | audio | video |
|---|---|---|---|---|---|
| E4B | CPU通过 | CPU通过 | CPU通过 | CPU通过 | CPU通过 |
| E2B | CPU通过 | CPU通过 | CPU通过 | CPU通过 | CPU通过 |
| Jev-Omni12B base官方processor | CPU通过 | CPU通过 | CPU通过 | CPU通过 | CPU通过 |
| OneJev4B | CPU通过 | CPU通过 | CPU通过 | metadata_inconsistent | CPU通过 |
| OneJev27B | CPU通过 | CPU通过 | CPU通过 | metadata_inconsistent | CPU通过 |
| Valen/Qwen base元数据依赖 | CPU通过 | CPU通过 | CPU通过 | unsupported_by_metadata | CPU通过 |

OneJev的audio token提示与配置/模板音频路径不一致，不能据此臆称模型支持音频或直接当已通过；Qwen base的unsupported_by_metadata只指本次固定文件无可用声明，不是运行音频失败或所有版本能力结论。二者均必须在最终基线/发布兼容表明确，不以缺失分数填零/平均掩盖，若原阶段门槛必须该能力仍保留缺口。

官方关闭thinking探针实测：Gemma删除think标记，OneJev/Qwen通过各自官方enable_thinking条件的空think闭合分支；完整答案token映射覆盖Choice2/3/4/20+U、Score2/3/10+U、Noul yes/no+U。视频使用固定public case的8/16帧值与真实metadata，截图保留独立类别但通过image输入。

此处Jev-Omni的普通tokenizer/template探针不能证明发布教师head读出；教师专用head沿用s0:01真实原tensor安全参考与s0:02完整公开benchmark，reports/teacher-technical-admission-gpu-001.json、teacher-head-reference-gpu-001.json、teacher-benchmark-two-pass-complete-001.json。不能用通用student payload复现发布分数。

Valen配置来自checkpoint.pt对应安全metadata，不假称PEFT LoRA adapter；此票不加载checkpoint或把Qwen base推理当Valen成绩。其真实组合/官方专用任务协议尚待独立闭环，正常进程禁止pickle加载和上游module执行。

后续GPU基线须冻结真实dev500与公开JevBench231输入、每模型官方模板/专用读出、逐题概率/pU边界、帧/时长/token/时延条件及两遍/跨机验收；CPU通过不免除真实forward。同原始media组件跨数据池的family排除应在基线和训练两端绑定allocation ledger。S0:07继续展开这些具体依赖，S0:08全绿前不启S1。
