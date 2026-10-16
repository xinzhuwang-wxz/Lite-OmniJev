# S5：真实校准、稳健性与同版本服务

当前主线E4B候选e465；前序S4原质量目标未达，阶段推进按ADR0023，实际S4出口仍待独立封存汇总。允许准备后续工作，S5验收不能代替S4完整证据。

## 范围和数据
冻结基座、官方模板、global33答案映射、thinking关闭、legal+U原始读出。校准只用独立calibration：六cohort各3000，实际7组18000题；audio choice1600/noul1400，text/image/screenshot choice各3000，video/crossmodal noul各3000。其余11组无拟合数据，明确unfitted；禁止借温度或伪造覆盖。trusted固定新UID作业读取数据，仅导出七温度及严格汇总/provenance。原始行、logits、日志不供agent读取。holdout只评估，不拟合/调参。

## 出口
1. 完整18000真实推理/拟合，实际模型、数据、代码、runtime、template/token、elapsed/exitcode和失败证据齐全；每100题安全汇总进度并监控资源。
2. 在独立holdout和公开DEV汇总原始/校准准确率、joint/legal+U NLL、Brier、ECE和coverage。原ECE≤.03、coverage@5%≥.65、准确率下降≤.005目标原样观测，未达按ADR0023单项披露，不静默改阈值。
3. 完整真实置换≤.02、正反偏差≤.05与坏输入/OOD/不足信息评估；支持的原语不能以缺cal数据默默删除。C1–C4/R1按实测瓶颈有界对照，数学等价reward不能冒称新RL。
4. 服务registry明确fitted/unfitted精确18分区，未fit默认raw且标明未校准；require_calibrated拒绝未fit/未激活；模型/adapter/官方template/globaltoken/rawreceipt/runtime recipe全绑定，不跨HF BF16与vLLM FP32套用温度。
5. 真实GPU、网关和浏览器验证同版本正确性，独立集成复核后记录stage原gate与实际受限出口。S6正式验收另票，不扩5b/生产运营。

## 当前执行
immutable fe58854已CPU60行为测试；七组JobSpec002真实metadata预检14.024秒，SHA4b345752a5bada15426845c305f206500694ad1191ee00e8c862e42b0ed1aa77；GPUfit未起。先审查新增隔离/progress，完成当前串行DEV，再启动fixed UID64201/2，保留旧准备001。
