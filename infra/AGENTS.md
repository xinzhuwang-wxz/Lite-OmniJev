# 实例与离线准备
- 无卡任务异常慢/超时 → 主机 free/os.cpu_count 不等于容器配额 → 先读 cgroup memory.max、cpu.max、cpu.stat；报告实际限制，重型 CPU 作业用 /tmp/omnijev-cpu-heavy.lock 串行，网络下载继续。
- 旧实例重启后 SSH 无输出 → 观察通道可能暂时阻塞 → 检查同一 handle 和实际 PID/退出报告；不得因观察超时重复启动下载或转换。
