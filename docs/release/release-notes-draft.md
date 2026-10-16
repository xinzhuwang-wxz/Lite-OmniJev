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
- D3P 版封存(6000 题,校准后):ECE 门 4/6 过(text .0013/screenshot .0086/image .0192/audio .0293),video .0316(近门)、crossmodal .0469 未达;校准零准确率损失;封存 video 72.0%(+0.3 vs e465 71.7)、image 92.6%(+0.1)、audio/crossmodal −0.3 内

## 质量局限(ADR0023,不假称达标)
- 原门"video ≥ 教师 97% 比率(≈91.4%)"未达(74.4%/教师 94.2%)
- ambiguous 家族弱于基座;crossmodal coverage@error 低
- ECE 三模态未达 .03 门;封存/开发差同源族 ≤3pp 达标

## 范围与边界
- U/不可答性不作为产品输出(ADR0024);p(U) 为内部量
- 引擎边界实测:vLLM 0.31 native 服务已就绪并尝试激活校准温度,被 registry 绑定**按设计正确拒绝**(温度绑定 transformers/bf16 recipe;vLLM 需其引擎自身的校准证据)——S5 出口 4“未经数值验证不跨引擎激活”首次真实验证通过;vLLM 单卡服务当前默认 raw 温度并披露未校准,require_calibrated 拒绝激活
- Playground 完整浏览器验收、E2B 量化(S6)为 pending,不虚标
- 教师蒸馏仅用 TRAIN 池(OneJev-27B,Apache-2.0);未用 JEV-27B 权重/Jev API/其标签

## 待决/后续
- D3P-600 放大(教师缓存可续)、SAS 程序切面监督、R1-B-gold 置换一致性(材料就绪未跑)
- ADR 轨道:目标中心输入 token、时间锚索引(需产品决策)
