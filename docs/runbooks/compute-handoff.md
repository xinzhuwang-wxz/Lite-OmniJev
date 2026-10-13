# 无卡准备与 GPU 交接

## 下一次用户动作

本地 M0 验证通过后租一台可切换 GPU 的 Linux x86_64 无卡实例；推荐 Ubuntu 22.04/24.04、8 vCPU、32GB RAM，持久磁盘至少200GB用于E4B阶段0，完整E4B/E2B/两教师及视频数据建议500GB以上，按实际manifest重新算。
这只是资源建议，不代表任何平台已验证或已下单。提供主机/端口/工作目录；SSH私钥留本机ssh-agent或用户配置，不贴聊天。先不开GPU。

## 无卡阶段 agent 执行

1. 记录机器架构、磁盘、Python、网络可达性；获取本地代码快照（无需先push）。
2. 安装 uv；冻结CPU开发依赖，分别为训练/服务解析Linux锁，锁CUDA轮子与容器digest，避免vLLM和训练torch互相改版本。
3. 用资产工具固定E4B revision，按LFS清单下载并verify；metadata-only不得当full ready。
4. 完成训练数据卡与manifest、split去污染、token和processor探针；断点续传缓存留持久盘。
5. 准备100样本、十步LoRA配置、checkpoint目录、HF/服务启动和测试脚本；实际命令必须在该环境预检。
6. 输出 `reports/no-gpu-readiness.json`：代码hash、环境锁hash、资产hash、磁盘、数据hash、检查项、GPU待验项目、预计时长/预算。
7. 全部必需准备通过后通知用户开卡；仅有包安装成功不能承诺kernel/显存可用。

## GPU 阶段

优先单张80GB级卡进行E4B BF16/教师验证；24GB仅用于已验证的精简试跑配置，不能承诺全部教师可用。具体型号由当前预算和作业峰值决定。
先 `nvidia-smi`、torch CUDA算子、vLLM模型加载验证，再启动P0作业。记录峰值显存、tokens、时延及训练步数；出现OOM/NaN立即保存诊断并小规模修复。
P0通过后生成S0详细票，按模型卡相同split/预处理复现；教师许可或head格式未通过准入时不加载。

## 用完交接

记录进程清单、checkpoint文件hash、持久路径、最后成功步数、恢复命令、报告和预计下一次GPU任务。
结束自己创建且已无用途的作业；发“本阶段GPU已用完，可关卡”的通知，由用户关卡。
禁止因保存了STATUS就宣称数据安全：必须核验产物已写入持久盘。
