# Order程序gold已见TRAIN12学习诊断

结果：**已见训练子集有学习信号，但未饱和；没有泛化获益证据，不推广。** 固定hash选择实际TRAIN120中的12条video，12独立family、6yes6no，不按模型对错选择。所有60条程序gold视频的canonical Sample、ledger、media、rawsource及程序真值hash核对通过；完整原训练data/config及actual init f2db、after831d身份绑定。

| 同12题 | before f2db | after order831d |
|---|---:|---:|
| 正确 | 8/12 | 9/12 |
| 平均gold legal CE（含U） | 0.6237266262 | 0.4497224937 |
| 平均gold yes/no margin | 0.1875000000 | 0.8229166667 |
| wall（含准备/加载） | 24.349秒 | 24.849秒 |

CE相对下降27.8975%，margin增加0.6354167。12题完整prepared tensor/input、canonical sample、gold与token映射前后全部相同。这支持参数对已见order TRAIN确有响应，不能证明已学会通用事件顺序；after仍有3/12错误，12题规模有限。结合root已有DEV50仍33/50且Brier更差，当前不足以称“已充分学会，只是没有泛化”，也不支持扩大路线。

CPU严格验证与6项选择/篡改拒绝行为测试先通过；官方16帧预备12.417秒。GPU每arm单遍、硬90秒、seed42、BF16、sdpa、确定性算法，串行执行，两arm exit0。每arm前nvidia-smi无compute app，结束亦空；没有训练、改参数、重复评估、读calibration/holdout或启动服务。成本未知保持null。

运行解释器 `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`；core `/root/autodl-tmp/s4-claim-pair-code-001`（3fe0ab2）；入口 `/root/autodl-tmp/evaluate-order-train12-001.py` SHA `16c461e743e9bcffc4eb10cfe37ea5481184514d909311ffef163d8e6e7a6290`。复用evaluate-train-learnability的官方prepare、decision_loss、readout与完整输入hash逻辑；没有改既有helper/生产模块。

TRAIN原题和逐题记录只在远端0700/0600目录：`artifacts/s4-order-train12-{input,cpu,before,after}-001`。Git只保存本汇总和SHA。完整身份、环境、wall、receipt/sidecar SHA见同名JSON。原训练data SHA `f16fbf93029461cf520522f4037ff829284a7a4a7c531d1ecd4b9e0b7bbe8ef1`；诊断subset `c6a1f0641dcb852a90527c42cb3b4744c603a60ce927c86c800a92a655614d4f`。

本地验证：`PYTHONPATH=. python -m pytest tests/test_order_train_diagnostic.py -q`，6通过；ruff check/format check通过。本诊断不是S4 gate或候选晋级证据。
