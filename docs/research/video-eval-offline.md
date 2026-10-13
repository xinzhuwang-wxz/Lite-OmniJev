# 音频/视频公开评测原料准备

2026-10-14；仅来源和离线准备，不构成模型质量、完整资产或商业使用验收。原方案用 HTMLParser 读取文本，未搜索 HTML 内嵌媒体。原门槛保留：阶段 0 教师 MMAU 63.1 / MVBench 53.1，允许 ±2 个百分点。

## 已锁来源及使用边界

- [MMAU 官方仓库](https://github.com/Sakshi113/MMAU/tree/110127f54c0dfba3faa5ec9feee4a7e4148679c5) 公布 1,000 条 mini；9,000 条 test 答案隐藏。本次仅读取 mini，不读取 test。官方 README 指向 [gamma-lab-umd/MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini/tree/ccd9696c0111ea7060827598f310558df0b71b0a)。该数据卡为 **CC-BY-NC-4.0**；代码 Apache-2.0 不能覆盖音频。唯一 parquet 1,213,407,227 B，上游 LFS SHA256 已记配置；远端研究评测隔离目录下载中，尚未实算完成哈希。商业训练与发布准入均 false。
- [MVBench 固定数据卡](https://huggingface.co/datasets/OpenGVLab/MVBench/blob/230a2d4fac8900333c61754641c7a13e069ac9c6/README.md) MIT 声明旁明确注明原视频版权归来源、仅学术研究使用，且其他协议仍适用。官方 20 个公开任务 JSON 每个 200 条，共 4,000 条，逐文件实算 SHA256 完成。官方 [任务→媒体映射](https://github.com/OpenGVLab/Ask-Anything/blob/4dd8210b030374bb0c510a57dfbe916f70d3ef71/video_chat2/mvbench.ipynb) 固定，视频路径、片段 start/end、帧目录类别写入新配置。不能把 metadata MIT 当作媒体许可。
- [允许教师固定卡](https://huggingface.co/akhilaaa3/Jev-Omni/blob/5addda86ddee081a68fb067477ea100c221b8917/README.md) 只有「14 evaluated tasks / 2,786 questions」及 task average 53.10 / micro 53.09；固定树没有评测脚本、任务名单、逐题结果或 exclusions。14 个完整官方任务应为 2,800 条，无法自行猜测少掉的 14 条。**教师同条件复现仍 blocked：需原 14 任务、2,786 row IDs、排除原因、提示/裁剪/解码设置**。不通过枚举任务组合反向选择好分数。MMAU 教师卡也没有 dataset revision；必须核对其实际 1,000 IDs/version，不能直接宣称当前 mini 同条件。

## STAR 原始来源更正

本方案 STAR 是 [Bo Wu 等，NeurIPS 2021，STAR: A Benchmark for Situated Reasoning in Real-World Videos](https://openreview.net/forum?id=EfgNF5-ZAjM)，对应 [csbobby/STAR_Benchmark](https://github.com/csbobby/STAR_Benchmark/tree/8c3971891609e9c8e0665ffa0de1b28a0bfc166d)，不是 Self-Taught Reasoner STaR（2203.14465）或 Video-STaR。仓库 Apache-2.0；视频来自 Charades，帧标注来自 Action Genome，不由仓库许可统一覆盖。

[Charades 原始许可](https://prior.allenai.org/projects/data/charades/license.txt) 允许学术/非营利/政府研究并另允许其他主体作 evaluation；禁止向第三方分发、修改后公开发布数据，商业使用须联系 AI2。不能准入商业训练/公开媒体包。[STAR 官方 README](https://github.com/csbobby/STAR_Benchmark/blob/8c3971891609e9c8e0665ffa0de1b28a0bfc166d/README.md) 给出视频分段 CSV 与 ffmpeg trim 流程。公开 split_file 已下载并哈希，Train 45,731 / Val 7,098 / Test 7,377（这是 question IDs 数，非视频数）。未获取 STAR test 答案。下一步只先 HEAD/限定大小下载 train/val QA、分段及 frame IDs；原视频按 Charades ID 家族隔离，许可文本和归属一起保存。

## 逐媒体下载决定

全部 MVBench 公共 ZIP 共 **17,328,579,712 B**，不含需手动申请的 NTU 320 videos；根会话批准最多 17.33 GB 研究评测隔离预算。实际本轮启动三个权利来源明确的 ZIP：

| 归档 | 字节 | 原始许可证据 | 本轮状态 |
|---|---:|---|---|
| perception.zip | 11,199,611,859 | [DeepMind 固定 README](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)：软件 Apache-2.0，其他材料 CC-BY-4.0 | 与 FunQA 串行下载；仅研究评测 |
| FunQA_test.zip | 1,350,969,357 | [FunQA 固定 README](https://github.com/Jingkang50/FunQA/blob/a45ad6592dbc977d8d037d326f2edf591193d1e6/README.md)：CC-BY-NC-SA-4.0 | 下载中；仅研究评测 |
| star.zip | 727,887,646 | Charades 原许可包含评价用途和分发限制 | 单独下载中；保留原许可、仅内部评价，禁止公开媒体包 |
| clevrer.zip | 1,123,944,836 | [CLEVRER 主页](https://clevrer.csail.mit.edu/)未找到媒体许可 | 查生成器/发布页许可，不能从 MIT 推断 |
| Moments_in_Time_Raw.zip | 67,108,520 | Moments in Time 原条款尚待固定 | 查询官方 release 许可；如需注册/接受条款交根会话 |
| ssv2_video.zip | 16,819,261 | Something-Something V2 原条款尚待固定 | 查询 Qualcomm/TwentyBN 官方条件；不代填表 |
| sta.zip | 1,085,038,855 | Action Localization 映射的来源归档，原媒体许可尚未固定 | 从官方 annotation/source 描述确定完整来源再决定 |
| scene_qa.zip | 525,972,733 | Scene Transition 源归档，原媒体许可尚未固定 | 固定 Scene-QA 原始权利与原视频来源 |
| tvqa.zip | 1,147,595,693 | Episodic Reasoning，原媒体是 TV 帧 | 固定 TVQA 官方研究许可和剧集媒体限制 |
| vlnqa.zip | 40,331,555 | Egocentric Navigation 来源许可未固定 | 固定 VLN-QA 原始环境/视频许可 |
| data0613.zip | 43,299,397 | 官方 task mapping 未引用该目录 | 明确用途/来源再下载，不能按文件名猜 |
| NTU RGB+D | 清单 320 个视频，媒体 bytes 未知 | 官方 README 明确必须从 ROSE Lab 手动下载 | 未提交表单/接受协议；访问条件缺口独立保留 |

## 远端作业与复核

远端 `/root/autodl-tmp/lite-omnijev/artifacts/data/research-eval/`；source 元数据在 `video-eval-source-metadata/`，包含实际下载的固定 JSON/README/LICENSE、API trees、split 文件，逐文件 hashes 见 `reports/video-eval-source-audit-001.json`。未入训练池。

- MMAU 子进程 PID 8759，curl PID 8760；日志 `mmau/ccd9696c0111ea7060827598f310558df0b71b0a/download.log`，完成后打印 SHA256 和 exit_code。未验证完成前保持 running。
- MVBench 串行脚本 PID 9338，启动时 curl PID 9340；脚本 `artifacts/video-eval-download-approved-001.sh`，日志 `artifacts/video-eval-download-approved-001.log`。审计用相同脚本提交在 reports；每档 curl 有有界重试/超时，随后 sha256sum --check，最终 exit_code。
- 三 ZIP 预算总计 **13,278,468,862 B**；加 MMAU 总计 **14,491,876,089 B**。禁止把正在下载的部分文件当作 ready。
- STAR 单独脚本 `artifacts/video-eval-download-star-001.sh`，日志 `artifacts/video-eval-download-star-001.log`；启动 wrapper PID 9701。远端 metadata scp/tar 返回 0，逐文件远端哈希复核命令尚未返回，不能视为哈希通过。审计报告内嵌配置快照并记录配置 SHA256，后续扩展不改变此次下载预算/来源证据。
- 结束后根会话读日志最终退出码、核对文件大小、SHA256，安全列 ZIP 路径再提取；实际帧率/解码、音频时长/采样率、全量 IDs→媒体覆盖率仍待下载完成。公开数据原始真值仅评测，不能用模型预测覆盖。

本轮为文档/配置/离线素材变更：校验 JSON 可解析、20×200 记录、MMAU 1,000 唯一 ID、原始文档字节与哈希、下载脚本 bash 语法。没有模型 forward/GPU 分数；未宣称阶段 0 通过。
