# Lite-OmniJev

单步多模态决策模型:冻结基座 Gemma 4 E4B + 决策 LoRA + 原词表概率读出。一次前向输出
每个选项的归一化概率和 **p(U) 证据不足通道**,覆盖 Choice(2–20 项)/ Score(2–10 级)/
yes-no 三种判定,文本、图像、截图、音频、视频五模态。

**模型权重与模型卡(HF):[Chinzhu/Lite-OmniJev](https://huggingface.co/Chinzhu/Lite-OmniJev)**
(含 fp32 合并权重、LoRA adapter、GGUF F16/Q8_0/Q4_K_M、全套 serving 验证回执)

## 实测榜单(同题同协议,单遍确定性;对照引用其官方公开数)

| 榜 | 本模型(adapter 全模态挂载) | 基座 | 最强对照公开数 |
|---|---|---|---|
| JevBench 231 | **76.62%** | 76.19% | Jev-Omni-12B 86.15 |
| MMAU test-mini 1000 | 60.6%(挂载)/ **65.9%(音频·文本路由基座)** | 65.9% | Jev-Omni-12B 63.10 |
| Decision Index 公共子集 | 上游官方套件不可得(已隔离的源缺 3 个上游源),打分阻塞中 | — | 教师公开数不在公共部分 |
| MVBench 九任务 1800(参照) | **48.5%** | 47.2% | 教师 53.09(14 任务/2786,口径不同) |

距离最强对照(12B 教师,全参微调 3 万+ 题)尚有差距;MMAU 路由形态超过教师公开数。

<div align="center">
  <img src="docs/charts/mmau-routing.svg" width="700">
</div>

量化(Llama.cpp,公开案例集文本选答一致率,参考 = 已验收 serving):**F16 20/20 · Q8_0 20/20 ·
Q4_K_M 20/20**——Q4_K_M 以 5.34 GB(约 1/3 于 F16)零退化,是端侧首选档。

## Demo(真实录屏:并行概率读出 vs 自回归生成)

两侧同时起跑:左 = Lite-OmniJev 一次前向读出全部概率与 p(U);右 = 通用大模型自回归生成
同样的答案。文本场景 8 题同发。实况录制,无剪辑加速。

| 文本 · 8 题 | 图像 |
|---|---|
| <video src="docs/demo/demo-text.webm" width="400" controls muted></video> | <video src="docs/demo/demo-image.webm" width="400" controls muted></video> |
| **音频** | **视频 16 帧** |
| <video src="docs/demo/demo-audio.webm" width="400" controls muted></video> | <video src="docs/demo/demo-video.webm" width="400" controls muted></video> |

<div align="center">
  <img src="docs/charts/readout-diagram.svg" width="700">
</div>

**一次前向 vs 逐 token**:同题实测(文本 8 题),Lite-OmniJev 网关端到端 **0.13 s**,
通用大模型自回归生成 **2.5 s**;右侧引擎给一段文字,左侧引擎给每一项的校准概率与弃权通道。

<div align="center">
  <img src="docs/charts/latency-text.svg" width="700">
</div>

评测态与 serving 态的完整时延/显存数据见 `reports/s4-perf-report-001.md`、`reports/s6-perf-report-002.md`。

API 调用示例:

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

端到端 serving(网关 + vLLM)启动脚本与 100 案例逐位 parity 回执见 HF 仓库 `gateway/`;
GGUF(F16/Q8_0/Q4_K_M,文本一致性三档 20/20)与 MLX 4bit 转换均在 HF 仓库。

## 仓库内容

- `serve/` 网关与部署适配;`train/` 共享模板/训练/校准;`eval/` 指标与门槛;`data/` 数据契约
- `docs/` 决策(ADR)与 runbooks;`reports/` 全部实测证据;`issues/` 票据
- 本仓为工程仓(构建/评测/复现);**权重与模型卡在 HF**

## 许可

代码与模型权重均以 **CC-BY-NC-SA-4.0** 提供(非商业);基座 Gemma 4 适用 Google 原始许可。
