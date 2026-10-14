# S0 调度候选接续

终局goal active，S0未全绿，S1未启动。旧BF16/顺序cache失败保留；不换基座/引擎、精度与门槛，不自动开关卡或付费。

## 已落地

- 388ce87 修复bounded失败时本请求shielded CPU job drain，独立review APPROVE；普通失败等待已运行CPU工作，取消尚排队job，外部取消保留原占槽/shutdown语义。
- 官方Linux100题300次准备结果精确，40imagecache hits；报告s0-request-image-reuse-cpu-001.json。CPU等价不等于性能准入。
- 原生测量与网关专用adapter明确bounded-scheduled-v1，完整独立输入/答案+U/raw概率，首题结束后有界fanout，wall与work分离。网关保存逐题API答案摘要并验证JSON roundtrip、重复/漏题/错误qid与概率拒绝。
- 定向48pass；78源mypy/lint/282文件format/build/32票validate通过；网关adapter独立review另见STATUS。

## GPU结果

冻结执行树 /root/autodl-tmp/s0-scheduled-gpu-001 HEAD388ce87；服务原生026、FP32 IEEE、maxseq10/mmcache1，官方权重/config副本/模板未改。报告和源码实际绑定不依目录名推测。

- eager c10 cold2.3469/2.3250；原生cache288，实际并发9，四概率对顺序参考max2.36e-7。独立review确认fanout2..9为耗时尾部、非采集意外串行。
- FULLgraph capture[32,64,128,256,384,512,768,1024]：c10 cold2.5843/2.5512；c2=3.1902/3.1878；c3=3.0452/3.0468。全部失败，四概率通过。
- 两server167248/167657均已TERM，确认GPU进程清空后恢复教师；未扩展失败配置到gateway/off。原始小文件在artifacts/s0-cache-scheduled-20261014及远端artifacts/graph1024-001；追踪报告s0-scheduled-native-eager-canary-001.json与s0-scheduled-native-graph-canaries-001.json。成本未知null，不伪造。

## 当前作业与下一步

教师resume003在2610/5600显式SIGINT，exit1已记录。resume004监督167948/timeout167949/教师167950，原/root/autodl-tmp/gpu-reference-fp32-complete-001冻结DF源码，相同输入/记录，最多3600秒。先核/proc、行数和004.exit，未终态不得重启。原始benchmark输出artifacts/teacher-benchmark-pilot-001.json/jsonl；新日志004。第一遍MMAU620/1000仅部分进度，不是完整质量准入。

保持教师占卡。scheduled_native只读实际026源码，提出至多一个官方async/chunked调度候选及证伪条件；不穷举。scheduling_review复核网关测量adapter。主会话整合结果、写票/STATUS，准备完再决定独占短测；性能须冷态每次≤2/四概率<.001，完整on/off/gateway证据就绪才能关04。独立gate身份/两真人κ仍外部条件，不用agent自标代替。
