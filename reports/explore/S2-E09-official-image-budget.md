# S2-E09：官方视觉token预算对照

状态：prepared，2026-10-15。E08失败才启动；不修改官方模型资产或在训练链运行时切代码。

依据：实际processor_config.json默认max_soft_tokens=280；Google官方https://ai.google.dev/gemma/docs/capabilities/vision/image说明所有Gemma4支持70/140/280/560/1120，官方示例修改image_processor.max_soft_tokens控制细节与速度。保持冻结E4B、同v2 adapter、模板、原图与概率语义，仅用官方1120预算单次前向；不是新增分类头/推理引擎。

假设：丢失图像细节是部分错误的原因。已有dev聚合显示颜色13/19、计数12/18正确，虽不能据此证明分辨率是原因，但足以做不重训的有界对照。相同dev150与v2=123/150对比，至少125才扩大；只试1120一次，不扫全部档位。总GPU上限5分钟，记录准确率/ECE/实际耗时与显存；超预算或无收益停止。

.scratch/s2/image-budget-probe.py只在独立进程覆盖官方processor参数，调用原评测代码，报告额外记录research_input_profile、代码哈希、原/新预算；原模型目录不变。若有收益，必须正式贯通训练/服务配置、校准和部署性能，并重新完整验收，不能把研究参数混入默认基线或直接宣称S2通过。

CPU官方处理器实测通过：同一train图片在280/1120预算下分别260/1107个实际图像token，prompt339/1186，pixel tensor分别[1,2520,768]/[1,10080,768]，均小于2048序列上限。reports/s2-image-budget-preflight-001.json。CPU耗时含冷启动差异，不作为速度收益证据。
