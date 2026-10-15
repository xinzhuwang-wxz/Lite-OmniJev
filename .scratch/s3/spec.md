# S3 音频/视频决策LoRA业务闭环规格

Owner: main-session-2026-10-15
Parent: S2（进行中；本票在S2链运行期间并行准备,池/代码/配置就绪后无缝接卡）
State: 一个闭环;沿用ADR0013校准原则与s2:01联合单LoRA架构

## 数据与池(已冻结)

- 视频(CLEVRER,CC0-1.0,revision official-train-json-sha11181da6…):候选9999(全10000 scene去1),dev剔除500→train 8299/calib 900/holdout 300,双遍字节一致(train SHA见s3-pools-video-001);16端点帧协议,媒体为逐文件sha验证的mp4(免解码,解码在训练侧)。
- 音频(Clotho-AQA,license QA:MIT;media:{CC-BY-3.0×426, CC0-1.0×520}两档批准):候选946,dev剔除500→train 326/calib 80/holdout 40,双遍一致;16k mono≤30s。
- 联合池:s2-joint(12559)+audio326+video8299=21184行,拼接报告绑定五输入SHA;训练账本=五ledger并集校验。

## 训练协议

在s2:01 mixed机制上增量:modality扩展audio/video媒体分派(processor官方kwargs:audio→input_features+mask@16k,video→pixel_values_videos+video_position_ids@16帧≤60s);joint单LoRA继续覆盖全部五模态(产品契约与旧模态回归门的载体);固定seed42/lr1e-4/r16α32;samples-seen=10000(约0.47 epoch,视频样本前向开销大,epoch上限与文本收敛证据共同约束);midway 50%;账本五并集前置校验。

## 评测协议

- audio dev500、video dev500各双遍(Noul题,yes/no+U三向读出,字段与S0基线一致);视频另跑8帧对照一遍(--video-frames 8),8/16两套齐全即frame_ab_complete。
- 旧模态回归:text/image/screenshot dev500各双遍,与S2记录同条件对比,下降≤1pp。
- 反事实翻转:S3门无此项;Noul不适用置换,记not_applicable。

## 门槛(configs/gates.json S3原值,教师=Jev-Omni-12B同条件S0基线)

| 门 | 值 | 基线 |
|---|---|---|
| audio_teacher_ratio | ≥0.9 | 教师audio 38.0%→bar 34.2% |
| video_teacher_ratio | ≥0.9 | 教师video 69.8%→bar 62.8% |
| old_modalities_max_regression | ≤1pp | text 99.6%+image/screenshot(S2实测) |
| frame_ab_complete | 1 | 8/16两套齐全 |

未达标如实记录;结构性不可达按ADR0013格式另录。

## 证据

同s2:01标准;SwanLab真实指标;STATUS+commit。

## Out of scope

量化/E2B(S6)、封存holdout构建(s0:05)、新数据源、门槛静默调整。
