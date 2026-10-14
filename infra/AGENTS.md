# 实例与离线准备
- 无卡任务异常慢/超时 → 主机 free/os.cpu_count 不等于容器配额 → 先读 cgroup memory.max、cpu.max、cpu.stat；报告实际限制，重型 CPU 作业用 /tmp/omnijev-cpu-heavy.lock 串行，网络下载继续。
- 旧实例重启后 SSH 无输出 → 观察通道可能暂时阻塞 → 检查同一 handle 和实际 PID/退出报告；不得因观察超时重复启动下载或转换。
- 持有重CPU锁仍exit137 → 锁不限制网络下载等常驻RSS，且137本身不能证明OOM → 同时记录cgroup memory.events与进程RSS；先减常驻开销，或仅将小型公开媒体/processor metadata迁到本机CPU复验，Linux失败证据保留，不把本机通过当Linux/GPU通过。
- 多模型同时启动会干扰vLLM自动显存估算 → 外部进程占用变化可计入profile → 初始化优先串行，单并发诊断显式设置足够的KV字节预算；正式性能测量不得并发其它GPU作业。
