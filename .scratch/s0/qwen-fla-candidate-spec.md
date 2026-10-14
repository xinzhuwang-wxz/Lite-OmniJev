# OneJev官方FLA单候选（s0:07内，不新票）

依据：真实27B公开双轮647秒，运行日志及实际模块确认torch gated-delta fallback；官方fla-core0.5.2已固定wheel/hash/MIT且H20 tinykernel通过。只做一个模型候选，预算600秒，停止失败/不确定候选，不做依赖或配置穷举。

运行：同一固定OneJev4B、JevBench231、官方模板/关闭thinking、BF16/seed42/noTF32、完整词表原概率→答案+U读出，两完整pass；仅增加文件系统PYTHONPATH官方FLA overlay及独立Triton缓存，不修改primary环境/模型/引擎。

必须：231对实际输入一致，raw概率/条件/pU/预测两遍精确一致、有限值/固定分母；失败不降容差。比较旧torch-fallback逐题分布/预测及官方计分，真实变化完整记录。旧基线不覆盖。

内部采用依据：正确性条件通过、公开准确率回退不超过1pp、逐题warm forward中位数至少快2倍，即使当前另有质量评测作业。共享GPU时延只作内部诊断，不声明隔离性能/S6达标。未达或不确定则不启用。后续质量比较保留已验证backend中的更强教师基线，不能通过换较差backend降低自己的目标。

所有后续启用必须显式记录fla版本/wheelSHA/overlay、并以新报告版本保存结果；Gemma的HF/vLLM数值参考保持，不把FLA激活差异冒称Gemma1e-3已验。
