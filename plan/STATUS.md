# STATUS

更新:2026-10-17 21:05。**负责人:SOTA冲刺3(本轮起)。主线:拼最新预发布候选 → ADR0025 榜单实测差距 → 按宏观规划定提升。GPU 空闲可随时发射;盒上 build 样本作业进行中。**

## ⚡ 接续点(第一动作)

1. **等 MVBench 两份报告落地**(~22:25-22:30,`reports/s4-public-bench-mvbench-{d3p600,base}-001.json`;screen `pbm-d3p`/`pbm-base`)。**MMAU 已出**:基座 65.9%、D3P-600 60.6%(均两遍一致,学生协议);教师 Jev-Omni-12B 62.0 为教师协议口径,严格判定须同协议重测。
2. **分数齐后填 ADR0025 差距表**,关键议题已浮出:D3P-600(视频 LoRA)在 MMAU 音频上 −5.3 vs 基座,发布形态需考虑按模态路由 adapter(内部管线自由,ADR0024 修订二)或重校准;ECE 未套发布校准,勿直接比较。
3. 同协议对照矩阵(需要时再烧卡,勿盲目排队):OneJev-4B/27B(**qwen3_5 系**)与 Valen(**Qwen3.5 系**)需先给 s1_dev_eval 加多模型族模板/processor 支持;Jev-Omni(gemma4_unified)可直接跑——夜间链已排:当前两屏退出后自动跑教师同协议 MMAU+MVBench(`chain-teacher-student-protocol.sh`,log `rl-logs/teacher-student-protocol.log`)。Valen-2B 权重下载被盒上外网不可达阻塞(repo `Valen-Team/Valen-Preview-0923@30096aaa`,网络恢复后重下)。之后:Decision Index 27 子集打分(输入已 staged)、JevBench 对照补测。
4. 提升路线按差距表+宏观规划定(`reports/explore/s4-sota-macro-plan-001.md`)。

## 现役资产(已核验)

- **基座**:`/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef60…`(112GB 全量在盒)。
- **产品候选 = D3P-600**:`s6-d3p-student-600-001/run/adapter`,adapter sha `767be09b…`(rank16/α32,600 CLEVRER 样本,80 步,video 模态);S5 封存证据 `reports/s5-holdout-eval-008.json` + 校准 `reports/s5-calibration-fit-011.artifacts`(均绑 767be09b)。
- `lite-omnijev-release-001` 内是**旧 adapter-d3p150**(82525f39)——已过时,仅作结构模板;`s5-registry-export-001` 同样绑 d3p150,勿直接复用。
- RL 形式化器是内部管线组件(级联线已闭,见下),**不进发布包**。

## 已闭线(勿重跑)

- **RL 形式化器级联:场景 150/150 全覆盖,终判 118/150 < D3P 123,线停**。`s4-ns-cascade-002/final/gate.json` 为准;`rescue/` 三分片 37/37/35 覆盖 rest-109。`/root/autodl-tmp/launch-rest2.sh` 是 18:54 的过时计划(rescue 已完成同工作),**不要再跑**。merged 文件有 1 行重复 id(clevrer-train-03324-5),不影响判定。
- 已证伪路线清单见 reports/explore/(同骨干集成、COMBO/SAS 混合、自适应取帧、温度转移、纯 gold RL、K3 自一致、v3 并集)。
- G1 原子门未过(presence AUC .949<.97);PTR 门未过。

## MMAU/MVBench 样本管线(本轮修复记录)

- 设计定案脚本:盒 `.scratch/s4/public_bench_samples.py`(632 行,自校验:FFmpeg 无损拼装 16 帧 mp4 + RGB24 回读断言 + 冻结 S5 loader 终检)。MVBench 原始视频不在盒,用教师协议冻结的 canonical 16 帧(`public-video-protocol-freeze-001`)拼装。
- 本轮补丁:①MMAU 46 行内嵌 MP3 → ffmpeg 解码为 pcm_s16le WAV(Media sha 覆盖转码文件,provenance 留 meta);②>30s 音频样本级裁剪(既有约定)。已过 trim 崩溃原因:旧版 stream-copy 超冲,新版 wave 精确切。
- 已知协议差(如实记录):`load_media` 会把帧 scale 到 640 内而教师路径未做;MMAU 46 行 mp3→wav 转码;报告见 `build-summary.json` 的 `trimmed_rows_over_30s`/`transcoded_rows_nonriff`。

## 判定口径(ADR0025,不变)

- 五榜:Decision Index(含视觉)/ JevBench 231 / MMAU test-mini 1000 / MVBench 九任务 1800 / 我方公开多模态决策评测集;准确率+可靠性+成本;四对照(OneJev-4B、Valen-2B、Jev-Omni-12B、OneJev-27B)同协议同题全超才称 SOTA。
- **JevBench 为外部提供**(repo `fstandhartinger/jevbench@7625ecb`,公开 original/easy/hard 231 行,未用于训练)——不自建。CLEVRER 按用户指示保留在评测表(方法试验场),不作为 SOTA 声明口径。

## 盒上纪律(传承)

- 长作业一律 `screen -dmS` + 显式绝对路径 + 幂等跳过(report 存在即 SKIP)+ 禁止 `rm` 输出;发射脚本先 `pgrep` 防双跑;`timeout` 按实测速率。
- 多维监控:发射即挂监控(逐阶段 log 关键行/异常/退出码),不等跑完。
- 私有作业前清 /var/tmp overlay 媒体快照(30G 限制)。
- SSH 别名 `seeta`;凭据不入仓;Python env `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`(pyarrow/PIL/pydantic/torch 全)。

## 待我决策

- (无新增;租卡/开关卡、HF 上传 token、范围变更照旧须用户。)
