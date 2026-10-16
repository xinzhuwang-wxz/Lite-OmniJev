# s4-i2-twin-u-prep-001 — I2 孪生分歧 U 监督:CPU 信号提取与诚实弱信号结论

状态:**CPU 信号分析完成(dev500 全量离线);结论=video 族信号弱,I2 不排 video GPU 前向,
若继续应换 ambiguous/text 族;TRAIN 切片基座前向已并入 s6-teacher-cache-prep-001 JobSpec。**

## 预注册规则族(前沿报告 I2,简单函数族)

U 目标 = 1[基座熵 > τ] ∨ 1[基座-教师 argmax 分歧],τ∈{0.5,0.8,0.95} bits;
仅当 flagged 组的候选错误率显著高于 unflagged 才有监督价值。

## 离线实测(video dev500,零 GPU,三份已存逐题记录联算)

纯基座(无 LoRA,55.6%)+ OneJev-27B 教师 + 候选 e465 同 500 题联算
(`.scratch/s4/analyze_i2_signal.py`,输入 sha 见 oracle-001):

| 信号 | flagged 占比 | flagged 错误率 | unflagged 错误率 |
|---|---|---|---|
| 基座熵三分位(低→高) | 33%×3 | 25.3% / 29.3% / **22.2%** | — |
| 基座-教师分歧 | 147/500=29.4% | 30.6% | 23.5% |
| τ=0.8 ∨ 分歧 | 34.8% | 29.3% | 23.6% |

## 结论(问题→原因→解法)

- **问题**:video 族上基座熵与师生分歧都不能有效定位候选错误(最高熵三分位错误率反而最低)。
- **原因**:基座 video 仅 55.6%,其"认知上限"在 video 上不可靠——报告 I2 风险条款
  ("基座在部分模态自身答案性差,先只对 video/image 族启用")被实测否定 video 一半。
- **解法**:I2 的真实命中是 JevBench ambiguous 族(基座 5/7 > 候选 2/7,P0b 证据);
  若排 GPU,用 ambiguous/text 族切片做 U 监督,video 不排。程序构造 U(B1/WSFT 的
  24+66 题)不受影响,继续为主通道。
- 已并入 JobSpec 的部分(s6-teacher-cache-prep-001 作业2):同 150 题 TRAIN 切片基座缓存
  ~4 min,供 main 在 ambiguous/text 方向决定后复用;不单独排 video I2 前向。
