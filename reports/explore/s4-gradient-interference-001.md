# S4 TRAIN 梯度冲突诊断

依据：视频CE、reference KL、跨模态覆盖三轮均未同时守住固定video/cross筛选。最新覆盖候选video108/150、cross34/50；较保留首轮纠正1题、退化2题，no→yes由10增至12。准确率取舍不足以证明梯度冲突。

本次只测量，不更新参数：固定保留首轮adapter6bb86b09…与coverage准备002的360条TRAIN、seed42训练顺序。前3个step各12draw，固定同一权重分别计算video-only、image+audio、other的gold CE梯度，每draw按原accum12缩放。总36draw，三组每step预期5/3/4。模型eval关闭dropout；不是重演训练轨迹。保留原官方模板、答案映射、1120图像预算、16帧，禁止读dev/calibration/holdout；不保存adapter，不触发optimizer。

记录LoRA组梯度norm/dot/cosine及sum-gradient与cross梯度内积、gold loss、输入/权重/代码hash、时间、峰值显存，零norm返回null。CPU预检与失败拒绝通过后GPU预算180秒，最多一次。若video与cross内积至少2/3为负且sum与cross至少1/3为负，才将保护跨模态方向的投影更新列为下一有界候选；否则不以本诊断支持该路线。此判据只指导实验选择，不是质量门，也不代表AdamW实际更新必然保持旧任务；后者受预条件和动量影响。

所有最终候选仍需既定小筛选及完整S4门。诊断不能证明SOTA、普遍冲突或训练集之外的因果关系。

实测完成：reports/s4-gradient-probe-gpu-001.json，112.807秒，36draw的三组计数均5/3/4。video/cross cosine为0.017642、0.012605、−0.110873；sum/cross为0.834209、0.749420、0.838818。未达到预先条件，不启动投影候选。仅支持此固定权重/局部TRAIN诊断，不排除后段或其他样本上的冲突。无参数更新。
