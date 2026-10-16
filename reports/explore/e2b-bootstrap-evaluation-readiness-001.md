# E2B step10：固定公开 DEV50 配对评估已完成 CPU 准备

真实CPU：text50 21.886秒、video50 41.158秒，两者exit0；全部100题实际官方processor输入准备通过。E2B33答案token/thinkingoff、8运行时资产完整hash、实际step10训练receipt、e49adapter、公开500题源SHA及逐题张量摘要均已绑定。尚未运行GPU，本报告不是质量证据。

video50直接复用既有motion142e路径的`select_samples`，相同namespace/hash排序，不选择新有利切片。text为原text500顺序前50，与gold/模型预测无关。代码保持旧E4B guard文件原状，自己精确使用E2B snapshot及e49step10身份。

root已另启动候选text完整500，**不重复运行本helper text候选**；当前优先交付video50。text500的现有records只有在源码/精度/实际prepared tensor绑定都相同情况下才可复用；不能因为ID相同就把其分数拼入本paired结果。校准任务优先级由root决定。

## 根会话可启动的 video 配对命令

```sh
cd /root/autodl-tmp/s6-e2b-eval-code-001
PYTHONPATH=. timeout --signal=INT --kill-after=10s 230s \
 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python \
 .scratch/s6/evaluate-e2b-bootstrap.py --cohort video \
 --output /root/autodl-tmp/s1-text-lora-001/artifacts/s6-e2b-paired-video50-001 --execute
```

源码归档SHA `46c87396411a069fcec808e63ce5ec984701e9b8d2911ae2dc0be929b09675dd`，runner SHA `505095b6dac2fdae01707a9dfe4b85c8813755cc9fc302b087f9bf5d316e7b9e`。该目录代码已冻结，执行入口重新核对CPU preparation身份，拒绝覆盖已有report/records。

单个进程只加载一次原E2B+e49 PEFT，每题prepare一次并搬到CUDA一次，基座arm在`disable_adapter()`内前向，候选arm恢复LoRA。两臂复用同一个实际输入字典；每次前向前后核对包括BF16张量字节的hash，同时检查与CPU完整输入摘要一致。无模型权重合并、不继承E4B指标。两臂各50次，共100forward；内部230秒报警，外部230秒INT+10秒kill封顶240秒，包含加载与预处理；进度每10题输出，失败留下report/已完成records，不自动续跑。

共享`baseline.make_record/score_records`保持原typed答案和去U后归一化指标。records同时保留原raw logprobs、联合K+U、p_U和真实prediction；root应另算含U产品正确数，不能把条件accuracy自动视为产品accuracy。基座是实际无LoRA E2B，不是随机模型或E4B。

8项CPU针对测试、ruff及diff检查通过。未读cal/holdout、未下载、未启动GPU。实际配对一致性和模型质量只能由后续GPU完成receipt确认。
