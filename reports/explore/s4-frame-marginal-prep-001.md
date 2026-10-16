# s4-frame-marginal-prep-001 — 帧采样边缘化探针:CPU 准备与预期校准

状态:**切片清单/索引规则/命令模板准备完成,GPU 由 main 排期;离线侧信号已算,预期增益有界且小。**

## 设计(任务书原文落地)

- 固定切片:video150(`s3-stable-probe-001/video-dev150.jsonl`,sha e9b91fed…,150 题)
  + 固定 temporal DEV50(选择规则 sha256("42:s4-motion-coverage:"+id) 前 50,data sha
  795f69fe…);两遍一致才算数。
- 两种 16 帧采样:
  - A=现行默认均匀格 `round(i·(T−1)/15)`(dev 媒体内已规范化,无需重解码);
  - B=中点交错格 `floor((i+0.5)·T/16)` 严格递增、与 A 逐位错开(需显式 frame_indices)。
- 边缘化规则:逐题 `p̄(a)=mean(p_A(a), p_B(a))`(U 去除后的 source 分布),argmax 判题;
  同时报 A/B 单采样与 oracle(任一对即正确)。
- 实现路径(全部复用已验证机制):B 采样的显式索引走 `train/multimodal.load_media(
  frame_indices=…)` + 进程内 hook(与 `.scratch/s4/evaluate-video-motion-coverage.py`
  `--policy` 完全同机制,GPU 上真实跑过);manifest 由 CPU 预生成
  (`scripts/video_motion_manifest.py` 模式,逐视频 ffprobe total_num_frames → 两组索引)。
- 命令模板(main 排期,~10–12 min GPU):对每个 policy 跑
  `s1_dev_eval --data video-dev150.jsonl --passes 2 …`(B 采样经 hook),DEV50 同理
  (evaluate-video-motion-coverage 已内置);预算 150+50 题 ×2 采样 ×2 遍 × ~0.73 s/题。

## 离线校准(已有数据,零 GPU)

旧 CE0 adapter 的 DEV50 双采样记录(uniform vs motion,同 50 题)已可直接联算:
单采样各 30/50,**边缘化 31/50(+1 题)**,采样间分歧率仅 **4%**,oracle 上限 62%(≤+2 题)
(`s4-video-ensemble-oracle-001.json` frame_marginalization_side_signal)。
含义:帧维互补远小于模型对互补(19.2%);本探针预期增益 ≤+2 题/50,**不是主攻方向**,
GPU 窗口紧张时排在 B1+WSFT/I5/I2 之后,可砍。

## 判定规则(预注册)

- 过门:边缘化在 video150 ≥+3 题(121/150 当量)且 DEV50 不回退 → 值得扩 500;
- 放弃:≤+1 题或概率指标(Brier/ECE)变差 → 帧采样方差不是残余错误主因(与 oracle-001
  结论一致),停止该子路线。
