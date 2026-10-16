# Lite-OmniJev E4B 决策模型卡(草稿 2026-10-17,数字以最终封存汇总为准)

## 模型
- 基座:Gemma 4 E4B-it(revision ee0ef6023621cff504d758262d4e04895a5af4a2,Apache-2.0),冻结
- 决策 LoRA:rank16/α32,官方 chat template,thinking 关闭,单 token 读出(legal+U)
- 候选:D3P 蒸馏版(e465 起点上,OneJev-27B 教师 150 题 TRAIN 切片软目标蒸馏 40 步;教师仅作训练信号,不用于标签/权重/API)
- 校准:七组温度(18000 独立校准题拟合,per-group NLL 全改善),runtime_recipe=bfloat16/transformers/native_lora;11 组明确 unfitted,默认 raw

## 实测(开发集,同协议同输入,两遍可复现)
- video150:候选 122/150(81.3%)vs 起点 116/150(+4pp);JevBench231:e465 76.62% vs 冻结基座 76.19%(headline 基线门过);macro −2.46pp 如实披露
- text 99.6 / screenshot 96.8 / image 82.6 / audio 79.4(候选);封存集 e465 版:text 99.7 / screenshot 97.8 / image 92.5 / audio 88.2 / video 71.7 / crossmodal 80.0
- 校准零准确率损失(封存六模态 Δacc=0);ECE 校准后:text .0013/screenshot .0117/image .0193(过 .03 门),audio .0308/video .0557/crossmodal .0480(未达,ADR0023 披露)

## 已知局限(ADR0023 口径)
- 原 S4 教师比率门(video≥教师 97%)未达:74.4% vs 94.2%,不假称达标
- residual 错误集中:空间关系/事件运动/start-end(CLEVRER 类);ambiguous 家族弱于基座
- crossmodal coverage@error 低(.002);U 不作为产品输出(ADR0024),p(U) 为内部量

## 数据与许可
- 六 cohort 训练/校准/封存:CLEVRER(CC-BY 4.0)/VQA/Clotho-AQA/LibriSpeech/GUI-Odyssey/文本池;来源/哈希/隔离审计见 reports/s4-gold-data-admission-001.json 与 inventory 链
- 训练/校准/封存按原始样本族隔离;教师蒸馏仅用 TRAIN 池;禁用 JEV-27B/Jev API

## 复现
- 训练/评测/校准全部固定种子+代码哈希绑定;入口见 docs/runbooks/(training/sealed-gate/calibration)

## D3P 候选封存实测(6000 题,校准后,2026-10-17)
- 准确率:text 99.7 / screenshot 97.8 / image 92.6 / audio 88.1 / video 72.0 / crossmodal 79.7(校准零掉分)
- ECE:text .0013 / screenshot .0086 / image .0192 / audio .0293(四模态过 .03 门),video .0316 / crossmodal .0469(ADR0023 披露)
- 相对 e465 封存:video +0.3pp 且 ECE .0557→.0316,audio ECE 过门,其余持平
