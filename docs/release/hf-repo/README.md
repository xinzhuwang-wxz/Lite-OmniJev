---
license: apache-2.0
base_model:
- google/gemma-4-E4B-it
pipeline_tag: image-text-to-text
library_name: peft
tags:
- lora
- multimodal
- decision-making
- calibrated-classification
- video-question-answering
- image-question-answering
- audio-question-answering
datasets:
- clevrer
- vqa2
- clotho-aqa
- librispeech
- gui-odyssey
metrics:
- accuracy
- ece
- brier
---

# Lite-OmniJev E4B Decision Adapter

单步多模态决策适配器:冻结 Gemma 4 E4B-it 之上训练的决策 LoRA,输出**校准的类型化决策**(Choice / Score / yes-no)。

**Key points**
- 教师分布蒸馏(D3P):蒸馏 27B 教师的完整选项分布,而非 argmax
- 程序切面监督(SAS):CLEVRER 程序真值编译为训练期结构化目标
- 七模态×原语分组温度校准:18000 道独立校准题拟合,校准零准确率损失

## 评测

封存集 6000 题(独立校准/封存池,温度校准后;两遍可复现):

| 模态 | Lite-OmniJev(本卡) | 冻结基座 | ECE(本卡/基座) |
|---|---|---|---|
| text | 99.7 | 99.7 | **.0014** / .0029 |
| screenshot | 97.8 | 97.8 | **.0108** / .0162 |
| image | **92.7** | 92.5 | **.0202** / .0587 |
| audio | **88.4** | 88.2 | **.0310** / .0836 |
| video | **72.6** | 72.0 | **.0321** / .1382 |
| crossmodal | 79.9 | **80.0** | **.0528** / .1787 |

外部基准(JevBench,231 决策,同协议两遍):**76.6%** vs 冻结基座 76.2%。
参考线(尺寸不同,非同尺寸对比):同协议 27B 教师直答 94.2%(仅作训练期分布信号)。

## Requirements

```
transformers>=5.17, peft>=0.21, torch 2.13(bf16)
```

## Quickstart

```python
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E4B-it", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "<this-repo>/adapter")
# 官方 chat template;thinking 关闭;答案 token 映射见 docs
```

## 训练数据

CLEVRER(CC-BY 4.0)/ VQA v2 / Clotho-AQA / LibriSpeech / GUI-Odyssey(商用)/ 文本合成池——六源样本族隔离,来源/哈希/隔离审计链齐备。教师仅作训练期分布信号,未使用 JEV-27B 权重或 API 输出。

## Limitations

1. video 准确率 72.6%,低于同协议 27B 教师(94.2%)——残余错误集中于空间关系/事件存在/时序
2. crossmodal 校准后 ECE .0528,未达内部 .03 目标
3. U(不可答)为内部量,输出形态无 U
4. 校准温度绑定 bf16/transformers/native_lora recipe;其他引擎需重新校准

## Citation

BibTeX 见仓库 CITATION.cff(发布时生成)。
