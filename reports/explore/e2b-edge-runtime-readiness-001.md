# E2B端侧：实际可执行入口与剩余闭包

2026-10-16，CPU只读核查。未启动GPU/服务、下载、安装依赖、构建/转换模型或访问私有样本。原HTML§6/§7.2、delivery spec仍以 **llama.cpp + 合并E2B决策LoRA + GGUF Q4_K_M** 为首选，LiteRT-LM是验证后加速选项。无需新增Android APK/同时双移动OS门；Mac可成为真实边缘设备支持项，但不能据此宣称手机内存/时延。ADR0023允许透明记录质量局限后推进阶段，不放宽模板、概率、来源、隔离或打包一致性。

## 实际资产与设备

本机MacBook Air/M4/10核/24GB统一内存/arm64/macOS26.5.1。PATH中无llama-cli/llama-server/LiteRT CLI；Python环境无mlx/mlx_lm/llama_cpp/ai_edge_litert。clang和cmake4.1.1可用；本机ollama --version实际0.24.0，不同于项目缓存Linux0.35.0，不能冒充锁定运行时。未启用Metal推理或创建daemon。

E2B原始权重**远端已驻留、本机未证明驻留**：`/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E2B-it/3e22461f65e89153144f8adb70e3b8c2cc9845a7`，model.safetensors实际10,246,621,918字节；既有完整资产验收记录SHA2db5482b…，本片未重哈希10GB权重。config实际SHA1b28f3d2…。架构/参数不是2GB驻留模型；原始5,123,178,051参数，Q4理想裸参数约2.56GB，实际还有量化元数据/非Q4 tensor、mmproj、KV与工作区。3GB必须实测，不把裸参数估算当RSS。现存E2B只是base/零样本基线；没有已接受的E2B决策adapter、蒸馏产物、GGUF或校准版本。E4B adapter维度不能直接用于E2B。

远端真实CPU入口：
`/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/bin/llama.cpp-v0.6.0-cpu/bin/llama-cli`，SHA83391985…与既有回执一致；本片 --version exit0，显示0.6.0/build24/commit45ebe32。仅四二进制：llama-cli、llama-quantize、llama-gguf、llama-gguf-split；没有已构建llama-server或C API共享库。源码目录无.git，显示的build commit不能当该归档真实Git revision；继续以源归档/具体源码SHA与二进制SHA绑定，不能继承父仓库提交。

## 已实跑的最低CPU探针

现有训练Python、CUDA_VISIBLE_DEVICES空、源码gguf-py加入PYTHONPATH，converter --help exit0；--print-supported-models实际包含Gemma4ForConditionalGeneration、Gemma4UnifiedForConditionalGeneration等。缓存`conversion/gemma.py:632–634`注册E2B decoder，954–956注册Gemma4VisionAudioModel。现有llama-quantize --help列出Q4_K_M，**Llama3示例ppl/大小不适用于E2B实测**。这些是可执行工具兼容证据，不是模型转换/端侧通过。

```sh
SRC=/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/src/llama.cpp-0.6.0
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$SRC/gguf-py" /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python "$SRC/convert_hf_to_gguf.py" --print-supported-models
```

冻结E2B决策模型并合并后，最小下一CPU导出顺序如下（**未执行；MERGED_E2B和OUT必须是新建、SHA绑定的真实产物目录，非当前E4B产物**）：

```sh
# Existing SRC/Python; no --remote, no model downloads.
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$SRC/gguf-py" /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python "$SRC/convert_hf_to_gguf.py" "$MERGED_E2B" --outfile "$OUT/e2b-bf16.gguf" --outtype bf16
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$SRC/gguf-py" /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python "$SRC/convert_hf_to_gguf.py" "$MERGED_E2B" --mmproj --outfile "$OUT/mmproj-q8.gguf" --outtype q8_0
/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/bin/llama.cpp-v0.6.0-cpu/bin/llama-quantize "$OUT/e2b-bf16.gguf" "$OUT/e2b-Q4_K_M.gguf" Q4_K_M 4
```

先对BF16 GGUF核模板/token/full-logit读出，再量化，才可区分转换与量化误差。[锁定v0.6.0官方量化说明](https://github.com/ggml-org/llama.cpp/blob/v0.6.0/tools/quantize/README.md)明确E2B decoder与mmproj分别转换。[官方多模态文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)及已缓存版本包含image/audio/video；支持文档不证明本项目输入/数值一致。没有运行上述导出，也没有借用社区base GGUF代替自己的决策模型。

## 必须先闭合的兼容点

|能力|具体存在/缺口|
|---|---|
|文本|Gemma4转换器/decoder存在。官方chat template、thinking关闭、答案全集单token及U映射还未与GGUF逐token核对。现成CLI采样文本不是决策API。|
|图像/截图|缓存源码有Gemma4视觉projector和CLI --mmproj/--image。需自己模型mmproj、真实像素处理/预算与官方1120输入对照；截图走相同图像路径但仍独立质量/延迟证据。|
|音频|源码Gemma4VisionAudioModel及CLI --audio存在。实际采样率/归一化/分段/token输入一致、完整概率和质量未验证；不能按通用音频支持直接放行或静默删功能。|
|视频|CLI --video默认4FPS，与项目固定16帧不同；需固定同16帧、顺序/时间戳和处理策略，不能默认CLI视频读入算sameinput。|
|跨模态|分别支持image/audio不证明同prompt混合顺序、联合媒体与官方模板等价；需真实混合输入读出与回归。PreparationCase单模态限制不能用合成替代。|
|概率/校准|缓存include/llama.h:1136–1142提供原始llama_get_logits_ith完整词表；sampled logits接口不同。项目尚无llama C API/完整raw读出接线，CLI help也没有已验证的完整候选概率出口。薄标准C API应用绑定可不改引擎，但必须实现/验收，不能用top-k或强制选择后概率。pU独立、legal分布去U归一化，温度绑定E2B及量化/模板/设备recipe，不继承E4B或未量化校准。|

最快真实端侧路线是**沿已选llama.cpp，先完成远端CPU标准读出/导出，再原生Mac构建同源运行时**；本片无新engine/base决定。Mac构建可复用缓存源码与现有clang/cmake，但源码目前仅远端驻留，未传输/构建，不假称已有Mac命令可立即加载模型。若改用MLX/LiteRT为首发引擎、或正式限制模态，需要root按现有范围决策，而不是本审计自行替换。LiteRT0.9 host帮助/缓存证据存在，但完整logits与自己E2B导出仍未验证；当前不另开研究分支。

最终须绑定真实E2B checkpoint→merged→GGUF/mmproj/quant→官方输入/答案映射→校准→运行时SHA，做同200题打包对照、各模态量化质量/资源/离线/回滚。原量化≤1pp/ECE≤.035、E2B相对E4B92%、Q4≤3GB/目标时延保持原实测报告；ADR0023的接受局限必须显式记录，不给数值/输入正确性豁免。当前ready=false；GPU全课程进行中，本报告只指出可复用的CPU入口和真实缺口，不要求新账号/下载或额外GPU。
