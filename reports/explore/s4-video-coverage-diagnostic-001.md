# S3 → coverage 全video500诊断 001

2026-10-16，仅既有公开DEV记录与已准入TRAIN容量CPU聚合；无GPU、新样本或封存读取。原始题目、媒体和逐题记录未输出。实际执行与三个身份拒绝检查（target、答案全集U缺失、family变动）通过；完整数据见同名JSON，含report/data/sidecar/adapter/source SHA。

**coverage 337/500，对S3保存预测318/500净增19（+3.8pp）：纠正52、退化33、共同正确285、共同错误130。** 改善主要来自存在启发式组；时序仍占当前163错误中的122（74.8%）。这些是screen同款词匹配启发式，不是官方task type。

| 启发式 | N | S3正确 | coverage正确 | 纠正/退化 | coverage Brier | coverage ECE15 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| existence | 129 | 85 | 99 | 18 / 4 | .351862 | .104018 |
| motion | 42 | 31 | 31 | 8 / 8 | .306837 | .154260 |
| temporal | 329 | 202 | 207 | 26 / 21 | .465616 | .127696 |

沿用的准确规则：instruction.lower().split()词集合优先命中before/after/when/while→temporal，次命中moving/stationary→motion，其余existence；标点不另标准化。没有人工重判或扩充任务分类。

| 真实→预测 | S3 | coverage |
| --- | ---: | ---: |
| no→no | 206 | 194 |
| no→yes | 29 | 41 |
| yes→no | 153 | 122 |
| yes→yes | 112 | 143 |

coverage减少31个yes漏报，同时增加12个no误报；mean p_yes .369813→.419692。当前时序组no→yes31、yes→no91，主要残余是时序yes漏报。运动组净收益0并有8纠正/8退化，不能把总体改善理解为全部视频能力等比例改善。Brier（去U条件分布、双类平方和）.479230→.422930；canonical yes/no ECE15 .125338→.112247。仅开发诊断，不选阈值或宣称教师门通过。

## 旧汇总与身份边界

S3原report accuracy=.626/ECE=.126459，但product_accuracy=.636。核对固定代码 `b6b827b...:scripts/s1_dev_eval.py`，字节SHA `cda2b7bf06ea2516d1bc6ed0732f57f741059d6c7a29effeaaf188b490b44be8` 与原report一致：旧metrics_from_records用sorted(probabilities)即no/yes平票顺序。当前按canonical yes/no重算准确率.636，与保存prediction及raw logits重算完全一致；JSON保留旧report值及旧no/yes顺序重算，不覆盖历史证据。当前预测纠正/退化主表以保存prediction计算，与canonical结果相同。

两次都严格验证data SHA `79ed30be...`、sidecar SHA、实际adapter safetensors SHA、500唯一ID及全部原Sample相等、family/target、完整yes/no/U标签与rawlogits→probability/pU数值绑定。S3仅pass0/original，coverage单pass。报告command的同model dir、media root、16frames和记录的image_max_soft_tokens设置相同；环境版本与两份script SHA留在JSON。**这只证明same data + recorded config，不证明逐prepared tensor/token匹配**：sidecar不含input_sha、token IDs或processor序列化SHA；数值token映射未持久化。effective280是image_max_soft_tokens字段，不是视频每帧280；视频soft-token预算未由这两report直接记录。不同源码SHA不冒充同实现。媒体没有重新解码。

## 已准入TRAIN容量

从coverage prep002的第二轮prepared-data绑定，经video pilot source provenance核验 `/root/autodl-tmp/s1-text-lora-001/artifacts/s3-pools-video-001/run-a/train.jsonl` 与其report/源SHA；仅解析TRAIN全部8299条，不看其他池。按同款启发式：

| 启发式 | 行数/唯一family | no / yes |
| --- | ---: | --- |
| existence | 2206 / 2206 | 1076 / 1130 |
| motion | 705 / 705 | 366 / 339 |
| temporal | 5388 / 5388 | 2711 / 2677 |

这是已准入TRAIN总容量估计，**没有做历史曝光排除，不代表新增未见容量**。未按dev单题匹配或选训练样本，未建立新课程。结果支持主会话优先分析时序差距，是否进行下一训练由主会话决定。

## 复执行与验证

只读SSH stdin运行，不部署或修改远端代码；wrapper对实际收到的脚本bytes计算SOURCE_SHA，结果绑定本地提交源码SHA。命令形态：

```sh
ssh -o BatchMode=yes -o ControlPath=/tmp/omnijev-ssh-root-final -p48926 root@region-42.seetacloud.com 'PYTHONPATH=/root/autodl-tmp/s4-reference-code-fix-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python -c '\''import sys,hashlib; src=sys.stdin.buffer.read(); exec(compile(src,"read-only-video-diagnostic","exec"), {"__name__":"__main__","SOURCE_SHA":hashlib.sha256(src).hexdigest()})'\'' --root /root/autodl-tmp/s1-text-lora-001' < .scratch/s4/diagnose-video-coverage.py
```

实际CPU成功、源码SHA等式核验、三种身份拒绝、ruff check/format check、py_compile通过。成本null；真实耗时在JSON。未补跑GPU，不代表prepared输入parity、全量教师门或训练收益。
