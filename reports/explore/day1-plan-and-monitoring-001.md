# day1-plan-and-monitoring-001 — 开卡一日执行清单与硬监控设计

前提(用户已澄清):旧卡=关机、磁盘保留、SSH 地址不变;开卡即 `/root/autodl-tmp` 原样可用。
**供给/重建任务全部取消。**本文档=一小时级清单+每个训练的内嵌中途闸门;监控由 main 的
Monitor 工具挂日志链,阈值全部写死在本文档与脚本内。

## 硬闸门总则(所有训练脚本的不可协商规则)

| 脚本 | 中途闸门(内嵌) | 杀死条件 | 终点门 |
|---|---|---|---|
| RL 形式化器(run_formalizer_rl.py) | 每 20 场景批次出事件谓词 F1(checkpoint 行含 `gate` 字段) | 连续 2 次不升(kill_stalled)或较最佳掉 >0.05(kill_degraded) | 七谓词 F1 各 +0.2 且级联 ≥124(dev150) |
| 顺序课程 SAS→D3P(train_s1 两段) | stage1 完成即评 video150;**未达基线不许启动 stage2**(wrapper 强制) | 中途 video150 抽样较上次劣化 >2 题 | video150 ≥124(超 D3P-600 的 123) |
| 任何其他训练 | 同上 video150 抽样规则 | 同上 | 各自预注册 |

Monitor 挂链命令(main 执行;阈值已写进 grep):

```
# 形式化器:每 checkpoint 一行 JSON,含 gate 判定
tail -f /root/autodl-tmp/s1-text-lora-001/artifacts/s6-formalizer-rl-001/rl-log.jsonl | grep --line-buffered -E '"gate": "kill|PROBE_F1'
# 顺序课程:DEV/video150 中途抽样行
tail -f <course-log> | grep --line-buffered -E 'DEGRADED|video150|mid_gate'
```

闸门判定逻辑在脚本内(`gate_decision()`,已 CPU 自测:continue/kill_stalled/kill_degraded
三分支全断言),Monitor 只做旁路告警,不承担判定——**杀死动作由脚本自身执行并留证**。

## 小时级清单(H0=开卡确认后)

### H0–0.5 环境/资产核验(零训练)
- `ssh` mux 重建;`nvidia-smi` 确认卡;`uv sync --frozen --extra training --group dev`。
- 资产核验:`python -m scripts.assets verify --manifest artifacts/manifests/e4b.json --root <models>`(全文件哈希);
  OneJev-27B 同法(configs/project-model-assets.json 内含教师条目)。
- 关键产物在位抽查(sha):i5-slice-600(2644b84f…)、b1-wsft(41ebacc4…local/远端同)、
  ii5v-full-mixed(a8e65288…)、combo-full(08363c9a…)、vlm-scenes-v4(150 行)、
  d3p600-video150-records(6db62a8c…)。任一缺/不一致→停,报 main。

### H0.5–1.5 RL 形式化器:数据集构建+冒烟
- CPU:`build_formalizer_dataset.py`(600 真值场景+探针问题集,输出 formalizer-{train,probe}.jsonl,
  自带 gold 审计;**远端磁盘保留则真值场景已可从 annotation 重建,命令已备**)。
- GPU 冒烟:`run_formalizer_rl.py --max-videos 20`(G=4,GRPO 组优势,KL 锚 0.05,SFT 混 0.3,
  lr 1e-5);**第一个 checkpoint(H0.5 内)必须出现且 gate=continue**,否则杀,读 rl-log 诊断。

### H1.5–5 RL 形式化器主训练(60 视频/epoch)+ 门判定
- 每 20 视频 checkpoint+probe F1 自动闸门(上文);Monitor 挂 rl-log。
- 每 checkpoint 预期 ~25–35min(G=4×20s 生成+2 次前向)——60 视频≈3 checkpoint,H5 前完成首 epoch。
- **终点门**:dev150 重生成 150 场景(严格 prompt)→ns_quality_gate →七谓词 F1 各 +0.2 且级联 ≥124。
  过门→保留 final adapter 进 H5;不过→留证,跳 H5 的形式化器项,课程照常。

### H2–5(与 RL 并行不冲突,分时:RL 用卡间隙跑课程)顺序课程两段
- stage1=SAS(ii5v-full-mixed 3215 行,steps=90);完成即 video150 两遍——**≥123(D3P-600 基线)
  才许 stage2**,wrapper 在两段间强制判定,未达即停课程留证。
- stage2=D3P-1500(i5-slice-1500 教师缓存如未完成先补:run_teacher_cache_1500.py,2400s 上界)
  →学生 480s(steps=80)→video150 门 ≥124。
- 中途抽样:每段内每 20 步抽 video150 50 题快评,劣化 >2 题杀(train wrapper 内嵌)。

### H5–7 胜者重校准(S5)
- 取当日最佳 adapter(RL 形式化器/课程/D3P-600 中 video150 最高者);
- 按 docs/runbooks/calibration.md 执行私有校准 fit(UID 沿用 s5 既有 spec;不碰 holdout)。

### H7–8 封存 holdout
- docs/runbooks/sealed-gate.md 流程;只回汇总门槛报告,agent 不读内容。

### H8–9 汇总+HF 上传
- 全天证据汇总(每作业 exit code/报告 sha/闸门判定链);HF 上传按 docs/runbooks/huggingface-release.md
  (模型完成才上传;不冒称 SOTA,同条件实测说话)。

## 失败即停原则
任何哈希不一致/闸门 kill/非零退出:停止该线、保留全部产物、立即报 main;不重试掩盖、不改阈值放行。
