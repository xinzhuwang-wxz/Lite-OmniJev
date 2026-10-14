# S0:04 下一技术切片：同图多题的有界调度与请求内复用

Owner: main-session-gpu-transition
Parent ticket: s0:04
State: prepared, not implemented or performance-admitted

## 问题与目标

真实eager native冷比3.93、网关冷比5.05；FULL CUDAgraph短canary仍3.80。KV确实覆盖媒体，瓶颈是每题重复准备与严格串行执行。原方案要求每题独立序列、原生vLLM共享前缀，没有要求全部串行。本切片验证原生scheduler有界调度能否达到同图10题/单题≤2，保留旧顺序诊断，不能把新结果说成顺序通过。

## 验收先冻结

- 复用相同公开10题/媒体与官方模板、完整答案+U/raw读出，不换权重/引擎，不改≤2及四分布严格<.001。
- 默认顺序模式保留；新候选显式记录`bounded-scheduled-v1`、并发上限、scheduler/maxseqs/精度、首题建cache策略与冷暖状态，不隐式改旧测量collector。
- 请求内同媒体验证/准备能复用时须使用原官方processor、按内容与处理参数严格绑定；不修改共享processor实例，不跨请求/租户持久缓存；100题输入摘要与逐题独立路径精确一致后才启用。
- 有界请求内调度，首题完成建立共享KV后再有限扇出；结果按qid回收，保持API选项/primitive/版本/usage/错误边界。缺题、错误qid或部分失败不得返回整体成功。
- 保留CPU线程取消后占槽直到真实结束、shutdown drain和有限媒体资源预算；不能无界gather32个大媒体或靠新假header证明性能。
- 顺序模式保持原Server-Timing语义；并行候选分开记录wall与工作量，重叠区间不得冒充可相加时长。捕获层记录真实inflight上限/请求起止和每题输入/原始概率/KV，明确测量模式。
- CPU先验公共行为：输入等价、选项映射不串题、上限与失败/取消/关闭行为。GPU只做短native canary；确定收益后才完整native/gateway on/off两重复，不重复无收益配置。
- 主性能仍为两个独立冷态paired重复，每个10题完整wall/单题wall≤2；warm/onoff作为全量诊断，四类概率最大参考差严格<.001。结果需真实KV覆盖、同配置/源码/hash/成本/退出码和独立review。任何未达项保留failed。

## 实施边界

主要模块：serve/gateway/app.py、train/multimodal.py及独立的调度采集路径/行为测试。先做最小可测候选，是否开启以实测决定；不提前实现第十一节持久媒体编码缓存B5、产品auto_questions B6或其它可选扩张。教师benchmark继续在冻结DF树，禁止与正式性能测量共用GPU；先CPU准备，完成后暂存教师checkpoint再独占短测。
