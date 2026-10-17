---
license: apache-2.0
base_model: google/gemma-4-E4B-it
tags:
- peft
- lora
- decision-making
- multimodal
- calibration
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
library_name: peft
---

# Lite-OmniJev E4B Decision Adapter (D3P-600)

单步多模态决策适配器:冻结 Gemma 4 E4B-it 之上训练的决策 LoRA,输出**校准的类型化决策**(Choice / Score / yes-no,Jev 式输出形态)。

## 模型描述

- **基座**:google/gemma-4-E4B-it(revision `ee0ef6023621cff504d758262d4e04895a5af4a2`),全参冻结
- **适配器**:LoRA r=16 α=32,bf16,官方 chat template,thinking 关闭,单 token 读出
- **训练范式**(我们的方法,区别于现有工作):
  1. **教师分布蒸馏(D3P)**:教师模型(27B,同协议)的完整选项分布蒸馏进直答权重——不是蒸馏 argmax,是蒸馏分布形状
  2. **程序切面监督(SAS)**:CLEVRER 程序真值在训练期编译为结构化切面目标(计数/颜色/运动/时序/碰撞)——把符号执行"编译进权重"而非推理时执行
  3. **校准温度**:七模态×原语分组温度,在 18000 道独立校准题上拟合

## 评测结果(封存集 6000 题,温度校准后,零准确率损失)

| 模态 | 准确率 | ECE |
|---|---|---|
| text | 99.7% | .0014 |
| screenshot | 97.8% | .0108 |
| image | 92.7% | .0202 |
| audio | 88.4% | .0310 |
| video | **72.6%** | .0321 |
| crossmodal | 79.9% | .0528 |

外部基准 JevBench(231 决策):**76.6%**(冻结基座 76.2%);同协议教师(27B)94.2%。

## 使用

```python
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained("google/gemma-4-E4B-it", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "<this-repo>/adapter")
# 官方 chat template,thinking 关闭;答案 token 映射见 docs
```

## 训练数据

CLEVRER(CC-BY 4.0)/ VQA v2 / Clotho-AQA / LibriSpeech / GUI-Odyssey(商用许可)/ 文本合成池——六源按样本族隔离,来源/哈希/隔离审计链齐备。教师仅作训练期分布信号(同协议直答 94.2%),未使用 JEV-27B 权重或其 API 输出。

## 局限(如实披露)

- video 类准确率 72.6%,未达同协议教师水平(94.2%)——残余错误集中于空间关系/事件存在/时序
- crossmodal 校准后 ECE .0528 未达 .03 目标
- U(不可答)为内部量,**产品输出无 U 概念**(Jev 式输出形态)
- 温度绑定 bf16/transformers/native_lora recipe;其他引擎需重新校准

## 训练与复现

固定种子、代码/数据/模型全哈希绑定;完整复现链见仓库 `docs/`(runbooks + ADR)。
