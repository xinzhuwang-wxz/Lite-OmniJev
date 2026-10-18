# Lite-OmniJev

<div align="center">
  <img src="docs/charts/readout-diagram.svg" width="720">
</div>

**决策不该逐 token 生成。** Lite-OmniJev 是一个单步多模态决策模型:冻结 Gemma-4 E4B 基座 +
决策 LoRA,一次前向**同时读出**所有选项的校准概率、置信度与 **p(U) 证据不足通道**。
Choice(2–20 项)/ Score(2–10 级)/ yes-no,文本·图像·截图·音频·视频五模态。
没有推理链,没有 token 预算,没有"请以 JSON 输出"的祈祷——决策就是一次前向。

**模型与权重(HF):[Chinzhu/Lite-OmniJev](https://huggingface.co/Chinzhu/Lite-OmniJev)**
——fp32 合并权重(分片)、LoRA adapter、GGUF F16/Q8_0/Q4_K_M、MLX 4bit、全套 serving 验证回执。

## 实录:并行概率读出 vs 自回归生成

两侧同时起跑:左 = Lite-OmniJev 一次前向读出全部概率与 p(U);右 = 通用大模型自回归生成
同样的答案。文本场景 8 题同发。实况录制,无剪辑加速。

| 文本 · 8 题 | 图像 |
|---|---|
| <img src="docs/demo/demo-text.gif" width="400"> | <img src="docs/demo/demo-image.gif" width="400"> |
| **音频** | **视频 16 帧** |
| <img src="docs/demo/demo-audio.gif" width="400"> | <img src="docs/demo/demo-video.gif" width="400"> |

*GIF 为实况录屏压缩版;完整 webm 见 [docs/demo/](docs/demo/),原始分辨率版本随 HF 仓库分发。*

**同题实测(文本 8 题)**:Lite-OmniJev 网关端到端 **0.13 s**,通用大模型自回归 **2.5 s**——
19 倍差距,而且右侧给一段文字,左侧给每一项的概率。

<div align="center">
  <img src="docs/charts/latency-text.svg" width="700">
</div>

全部模态的网关侧实测(Server-Timing):

| 模态 | 前向(upstream) | 网关端到端 |
|---|---|---|
| 文本 · 8 题 | 119 ms | **128 ms** |
| 图像 | 343 ms | 454 ms |
| 音频 | 576 ms | 885 ms |
| 视频 16 帧 | 1152 ms | 2216 ms |

## 基准(同题同协议实测;对照引用其公开发布数)

| 榜 | 本模型(adapter 全模态挂载) | 基座 | 最强对照公开数 |
|---|---|---|---|
| JevBench 231 | **76.62%** | 76.19% | Jev-Omni-12B 86.15 |
| MMAU test-mini 1000 | 60.6%(挂载)/ **65.9%(音频·文本路由基座)** | 65.9% | Jev-Omni-12B 63.10 |
| MVBench 九任务 1800(参照) | **48.5%** | 47.2% | 教师 53.09(14 任务/2786,口径不同) |

<div align="center">
  <img src="docs/charts/mmau-routing.svg" width="700">
</div>

不利的数字也在表里(JevBench 距 12B 教师尚有差距——对方全参微调 3 万+ 题,我们是 600 样本
LoRA);MMAU 路由形态超过教师公开发布数,是当前推荐 serving 形态的直接依据。

## 量化:端侧零退化

<div align="center">
  <img src="docs/charts/quant-matrix.svg" width="700">
</div>

Llama.cpp 三档,公开案例集文本选答一致率(参考 = 已验收 serving):**F16 20/20 ·
Q8_0 20/20 · Q4_K_M 20/20**。Q4_K_M 以 **5.34 GB**(约 F16 的 1/3)零退化,
是端侧首选档;MLX 4bit(6.2 GB)已随发布。

## 接入

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "用户订单在配送第 3 天,物流显示已到末端站点。",
  "questions": {
    "q1": {"type": "choice", "instructions": "该订单今天最可能的状态?",
           "criteria": {"A": "已签收", "B": "派送中", "C": "滞留"}},
    "q2": {"type": "noul",   "instructions": "现有信息是否足以判定?"}
  }
}'
```

返回每题的概率分布、`choice`/`score`/`noul` 判定与 `p_insufficient`。四条部署路径
(transformers LoRA / vLLM 分片权重 / llama.cpp GGUF / MLX)见 HF 模型卡"快速开始";
网关启动脚本与 100 案例逐位 parity 回执在 HF 仓库 `gateway/`。

## 工程可信度

每个发布的概率都经过验证:vLLM 0.26 serving 两遍**精确可重复**,对 float32 HF 参考最大
偏差 **5.75e-6**(预算 1e-3);网关与该报告**逐位一致**;真实浏览器(Chromium → 网关 →
实际权重)全流程通过。回执随 HF 仓库分发,可独立复核。

## 仓库内容

- `serve/` 网关与部署适配 · `train/` 共享模板/训练/校准 · `eval/` 指标与门槛 · `data/` 数据契约
- `docs/` 决策(ADR)与 runbooks · `reports/` 全部实测证据 · `docs/demo|charts/` 发布素材
- 本仓为工程仓;**权重与模型卡在 HF**

## 许可

代码与模型权重均以 **CC-BY-NC-SA-4.0** 提供(非商业);基座 Gemma 4 适用 Google 原始许可。
