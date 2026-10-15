# S3 音频/视频决策LoRA业务闭环规格

Owner: main-session-2026-10-15
Parent: S2（已通过；reports/s2-stage-acceptance-budget1120-001.json）
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
| video_teacher_ratio | ≥0.9 | 教师video 69.8%→bar 62.82%（500题至少315正确） |
| old_modalities_max_regression | ≤1pp | S2 text 99.4%、image 82.0%、screenshot 96.4%；图像/截图均1120预算 |
| frame_ab_complete | 1 | 8/16两套齐全 |

未达标如实记录;结构性不可达按ADR0013格式另录。

## 证据

同s2:01标准;SwanLab真实指标;STATUS+commit。

## Out of scope

量化/E2B(S6)、封存holdout构建(s0:05)、新数据源、门槛静默调整。

## 2026-10-15恢复会话准备

联合池与500行探针已用`.scratch/s3/prepare-joint.py`生成并两次核验一致，报告`reports/s3-joint-preparation-001.json`。每cohort选100行训练数据的均衡探针；30步仅作首次媒体/耗时诊断，不代替300步smoke或质量验收。候选配置在同目录main-config.json/probe-config.json，max_length=4096为视频输入预留，须实测预处理长度和显存后冻结。不读取holdout内容，不启动S3主训练；先完成S2门与小批量前向。

门槛汇总入口：`uv run python -m scripts.s3_gate --audio REPORT --video REPORT --image REPORT --screenshot REPORT --text REPORT --s2-image REPORT --s2-screenshot REPORT --s2-text REPORT --video8 REPORT --output METRICS`。CPU报告夹具验证CLI与拒绝不匹配数据/adapter路径；真实S3报告仍待训练，不作通过声称。

## 当前实测候选（2026-10-15）

原lr1e-4短探针发生遗忘并停止。按用户小批量优先要求，当前使用S2 v2 warm-start、lr2e-5、五模态各200行、100步×10累积的有界候选，真实完成843.25秒；step50保存midway，最终adapter1bebae31…。
`reports/s3-training-exposure-audit-001.json`绑定实际保存的输入与权重：1000次曝光，image199个family、其余每模态200个；首/末10步loss均值.722695→.207186。不是原300步smoke或10000次主训练完成证据。
小筛查全过后进入正式五模态500题双遍+8帧对照；图像82.0%已完成，剩余链进行中。是否以该候选替代原执行预算，待全门结果后用明确ADR决定，不为补足任意步数盲目延长训练，不降低质量门。
