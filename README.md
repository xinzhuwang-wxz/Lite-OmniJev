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
| Decision Index 公共子集 | 官方套件打分进行中 | — | 教师公开数不在公共部分 |
| MVBench 九任务 1800(参照) | **48.5%** | 47.2% | 教师 53.09(14 任务/2786,口径不同) |

距离最强对照(12B 教师,全参微调 3 万+ 题)尚有差距;MMAU 路由形态超过教师公开数。

## Demo

 served 决策(每题一行的概率 + p(U)):

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

返回(节选):

```json
{"answers": {"q1": {"choice": "B", "probabilities": {"A": 0.21, "B": 0.62, "C": 0.17},
                    "confidence": 0.62, "p_insufficient": 0.04},
             "q2": {"noul": 0.31, "p_insufficient": 0.31}}}
```

端到端 serving(网关 + vLLM)启动脚本与 100 案例逐位 parity 回执见 HF 仓库 `gateway/`;
量化格式的转换与一致性探针随 HF 仓库更新。

## 仓库内容

- `serve/` 网关与部署适配;`train/` 共享模板/训练/校准;`eval/` 指标与门槛;`data/` 数据契约
- `docs/` 决策(ADR)与 runbooks;`reports/` 全部实测证据;`issues/` 票据
- 本仓为工程仓(构建/评测/复现);**权重与模型卡在 HF**

## 许可

代码与模型权重均以 **CC-BY-NC-SA-4.0** 提供(非商业);基座 Gemma 4 适用 Google 原始许可。
