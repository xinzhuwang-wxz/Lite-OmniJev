# Lite-OmniJev E4B 发布说明(2026-10-17 候选版)

## 交付物
- model/adapter-d3p150:D3P 教师蒸馏决策 LoRA(视频 +4pp over 起点,video150 122/150 两遍一致)
- model/adapter-e465-baseline:对照起点 LoRA(JevBench 76.62% ≥ 冻结基座 76.19%)
- calibration/:七组温度 + registry manifest(7 fitted/11 unfitted,bf16/transformers/native_lora recipe)
- evidence/:封存 holdout、留任、门评估全记录

## 实测摘要(同协议两遍可复现)
- JevBench231:e465 76.62%(父 76.19%,+0.43pp;macro −2.46pp 披露)
- video150:D3P 122/150(81.3%);DEV500 留任:text 99.6(0)/screenshot 96.8(−0.2)/image 82.6(−0.2)/audio 79.4(−0.4)
- 封存 6000 题(e465 版):校准零准确率损失;ECE 校准后 text .0013/screenshot .0117/image .0193 达标,audio .0308/video .0557/crossmodal .0480 未达
- D3P 版封存数字:【05:30 后填入】

## 质量局限(ADR0023,不假称达标)
- 原门"video ≥ 教师 97% 比率(≈91.4%)"未达(74.4%/教师 94.2%)
- ambiguous 家族弱于基座;crossmodal coverage@error 低
- ECE 三模态未达 .03 门;封存/开发差同源族 ≤3pp 达标

## 范围与边界
- U/不可答性不作为产品输出(ADR0024);p(U) 为内部量
- vLLM FP32 服务默认 raw 温度(跨引擎套用需数值验证);网关 bf16 路径 recipe 绑定
- Playground 完整浏览器验收、E2B 量化(S6)为 pending,不虚标
- 教师蒸馏仅用 TRAIN 池(OneJev-27B,Apache-2.0);未用 JEV-27B 权重/Jev API/其标签

## 待决/后续
- D3P-600 放大(教师缓存可续)、SAS 程序切面监督、R1-B-gold 置换一致性(材料就绪未跑)
- ADR 轨道:目标中心输入 token、时间锚索引(需产品决策)
