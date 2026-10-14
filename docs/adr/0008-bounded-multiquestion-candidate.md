# ADR-0008：同图多题的有界调度候选

状态：2026-10-14技术候选，用户授权按原方案自主安排；尚未性能准入，不开启可选探索或生产部署。

原方案每题独立序列、使用vLLM原生共享前缀，要求同图10题wall≤单题2倍；未要求串行。实测串行native冷比3.93、网关5.05，官方FULL CUDAgraph短测仍3.80，继续相同配置没有收益。

新增显式`bounded-scheduled-v1`候选：先完成首题建立原生KV，再有界调度其余题，仍每题完整独立官方输入与答案+U原始概率。默认顺序模式不变；Settings/LITE_QUESTION_CONCURRENCY显式设2–10才启用。改变scheduler配置须on/off一致并记录launch/hash；不能把新结果说成严格顺序通过。

图像CPU重复准备通过独立processor浅拷贝上的请求内单entry复用候选解决，按RGB内容、布局与处理kwargs绑定。共享processor不修改，不跨请求/租户存数据，不手工展开媒体token。输入资源预算先于像素hash，持有的processed输出最多32MiB，过大或不可稳定绑定则明确bypass。仅LITE_REQUEST_IMAGE_REUSE=1且同请求多题纯图像时尝试；100题官方CPU输入精确等价通过前不把候选当准入。

并行Server-Timing分别标注preprocess_work/upstream_work/queue_work与total；工作量可以重叠，不能相加冒充wall。默认顺序四字段保持。结果按qid回收，单题失败取消并等待兄弟任务清理，不返回部分成功；CPU线程取消后的占槽/drain契约保持。

独立collector使用新schema保存每题真实HTTP事件、实际inflight上限、完整原始概率/输入hash/KV覆盖；batch wall包含准备/调度/解析收尾。相同独立冷态paired重复仍全部要求≤2，四概率严格<.001；warm/on/off完整诊断保留。GPU短native canary有明确收益且数值通过后才扩展实际gateway/onoff两重复。没有证据就保持candidate/failed，不放行S0或S1。
