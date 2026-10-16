# s6-teacher-cache-prep-001 — OneJev-27B TRAIN 真 teacher 缓存 JobSpec(D3P/I5-plain 第一步)

状态:**JobSpec 级 CPU 准备完成;GPU 实跑由 main 排期(本任务不启动 GPU)。**

## 范围与许可

- 教师=OneJev-27B(repo OmniJev/OneJev-27B,revision aec969d68e7e8e8b0df051e065c04304c0cd7163,
  **license apache-2.0**,54.7GB,15 文件),是项目资产清单必选教师;**非**禁止的 JEV-27B。
  许可与 provenance 引用:`configs/project-model-assets.json`(逐文件 lfs/git sha)、
  `reports/s0-development-video-onejev-27b-fla-002.json`(实跑绑定)。
- 输入:TRAIN 池视频 `s3-pools-video-001/run-a/train.jsonl`(8299 行,sha 912c871f…,CC0-1.0,
  仅 train;**不读 calibration/holdout 内容**,两者仅元数据 900/300 行)。
- 首步切片(I5-plain,~150 题):按预注册哈希 `sha256("i5:d3p:"+id)` 排序取前 **150** 个
  video 行(确定性、可复算;清单见 JSON bindings 的 selection rule);**不做反事实探针**
  (那是 ADR 轨,本 JobSpec 只做 plain 同输入分布缓存)。
- 记录格式=已验证的 s0_development_baselines records:逐题 `conditional_without_u`(legal 归一)
  + `joint_k_plus_u` + `p_U` + raw 摘要,即"legal+U 完整分布",非 SoftTarget/U=0 伪造。

## 免费 CPU 前置检查(已实跑,去留判据)

教师 video dev500 逐题分布熵:mean **0.731 bits**、median 0.845、仅 5.8% 题 <0.1 bits
(`artifacts/s0-teacher-development-gpu-001/video/records.jsonl`,pass1)→ 远高于 I5 放弃线
(熵<0.1 → 收益封顶放弃),**D3P 有真实软结构可蒸馏,继续**。

## GPU 作业(main 排期;H20,固定 seed 42,bf16,官方 template,thinking off)

1. I5 切片教师缓存(~11 min):
   `s0_development_baselines.py --build-report <video-a report> --dev-jsonl <i5-slice-150.jsonl> --media-root /root/autodl-tmp/s3-media-candidates-001/video/media --cohort video --model-root … --model-id onejev-27b --report …/s6-teacher-cache-150-001/report.json`
   (同 dev 实跑入口;~2.09 s/题/遍 ×150 + 模型加载 ≈ 11 min;两遍一致性可选)。
2. 同切片纯基座缓存(I2 输入,~4 min):同命令 `--model-id gemma4-e4b-it`(~0.8 s/题/遍)。
3. 验收:records=150、分布有限值且归一、p_U 记录、report sha 钉住、无 cal/holdout 访问、
   退出码 0;证据回传 worktree reports/。

## 后续扩量(需 main 决策,非本切片)

- 全 8299 TRAIN:~2.09 s ×8299 ≈ 4.8h 单遍(+加载)≈ 半张卡租期;同一入口只换 data 文件。
- 反事实/提示探针变体(报告 I5 完整版)须先过 ADR(教师查询契约扩展),不在本 JobSpec。
- D3P SFT 消费:学生 SFT 用缓存分布(SoftTarget=去 U 归一 + p_U 作 U 监督),与 B1/WSFT
  试点同一 trainer 入口。
