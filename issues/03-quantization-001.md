# 03 · 量化端侧:GGUF / 端侧 / Mac 三格式(交付链④)

状态:doing · 所有者:主会话 · 2026-10-18

## 闭环定义

把定版 release-002(D3P-600,以 serving 身份工件为源)转成三种端侧/本地运行时格式,
实测一致性差异与资源占用,产出报告;为 HF 发布补"可下载即用"的形态。

## 验收(冻结)

1. **三格式产出**:GGUF(llama.cpp,含 mmproj 可用时)、Mac 格式(MLX,mlx-vlm)、
   端侧格式(第三运行时,以可行域探测后冻结;候选 ONNX/ExecuTorch)。
2. **一致性**:量化版与 release-002 在公开 100 案例集(canonical 协议)上的选择一致率与
   概率差异入报告;原方案要求"服务器版/GGUF 版/端侧版同一 200 题结果差异写入报告"——
   用公开 100 案例集(300 决策)执行并如实标注规模。
3. **资源**:各格式量化档(FP16/Q8_0/Q4_K 等实际产出档)的文件大小、加载内存、
   单步时延(p50)入报告;Q4 档对原方案"内存 ≤3GB"门槛记录达标与否。
4. **模态覆盖矩阵**:各运行时×各模态(文/图/截图/音频/视频)可用性如实标注;
   不可用格标 unsupported+原因(如 GGUF E4B 音频需 gemma4uv projector,实测后定)。
5. 失败/不可行留证,不静默降级;量化掉分超 1pp 时停下待用户裁决。

## 范围外

- E2B 端侧包(原 S6 门槛的一部分)本轮不做,除非三格式验收富余;训练/校准不动;
- 生产压测、并发门槛。

## 依赖与现状

- 源:merged-fp32(HF 语义等价,1.53e-05)或 flat 物化版——量化源=HF compact 版即可
  (llama.cpp 走 HF 加载路径);bf16 merged 因合并舍入不作为量化源,除非 fp32 源过大不可行(如实记录)。
- llama.cpp 0.7.0-dev(9d1878a)已编译(CPU,盒 s6-quant):llama-cli/mtmd-cli/quantize 可用;
  转换器原生注册 `Gemma4ForConditionalGeneration`(`conversion/gemma.py`,tokenizer_model=gemma4)。
- MLX:mlx-vlm 有 gemma4 支持(含 E4B 音频;待实测)。
- 音频:GGUF E4B 音频历史缺口(#21325),gemma4uv projector 为修复路径,实测定。

## 记录

- 2026-10-18 开票;llama.cpp 工具链就绪;GGUF 转换发射待 serving 链让出 CPU/卡。
