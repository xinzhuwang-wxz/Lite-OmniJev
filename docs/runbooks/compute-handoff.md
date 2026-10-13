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

## 2026-10-13 租赁选型补充

以下是本项目容量建议，未绑定具体平台报价，不是已实测的训练资源下限。

| 方案 | 正常开卡CPU / RAM | 系统盘建议 | 持久数据盘 | 后续单卡 |
|---|---|---|---|---|
| 报价较低时的备选 | 16 vCPU / 96–128GB | 80–100GB | 500GB，可扩容 | A100 **80GB**，不选40GB版 |
| 推荐主线 | 16–24 vCPU / 128GB | 100GB | 500GB起，可扩到1TB | H100 **80GB**整卡，SXM优先 |
| 大显存余量 | 24–32 vCPU / 192–256GB | 100–150GB | 1TB | H200 **141GB**整卡 |

只有CPU阶段且支持弹性调整时，用8 vCPU/32GB准备，GPU阶段再升至表中规格。平台CPU/RAM若按卡绑定，选择对应每卡配额的主机，不能假定可独立调整。
系统盘固定较小时，把uv/HF缓存、环境、权重、数据、checkpoint全部放持久数据盘；不为了系统盘容量被迫升级GPU。
四个主要模型的原始safetensors合计约104.87GB（E4B15.99、E2B10.25、Jev-Omni23.92、OneJev27B54.71），另需数据、解压、环境、baseline与checkpoint余量。200GB只作为E4B起步，不推荐全流程都留200GB。

镜像首选官方Ubuntu22.04 LTS x86_64（24.04亦可）。只能选框架镜像时选PyTorch类，Python3.11优先；训练和vLLM分环境安装，不依赖预装PyTorch。
当前锁定torch2.14.1的Linux依赖含CUDA13.0组件，因此主机优先NVIDIA驱动R580或更高，并核对具体构建最低补丁要求。镜像内装CUDA不能升级宿主机驱动；驱动较旧则须另做CUDA12兼容环境锁和GPU验证，不能直接承诺现锁可用。
vLLM官方Gemma4 Hopper配方有cu129镜像，推理环境与训练环境可采用不同CUDA用户态运行库，但共享的宿主机驱动必须兼容二者。

A100可作为BF16路线备选；最终FP8服务验证另在H100/H200/L40S等合适硬件上实测。L40S48GB可作为后续推理服务候选；24GB4090不作为覆盖27B BF16教师与多模态训练的主机方案。
80GB建议按学生/教师分时运行制定，不承诺学生与两个教师同时驻留；首轮不用8卡，后续是否并行由实测吞吐与预算决定。

平台特殊点：AutoDL官方无卡模式说明为0.5核/2GB，会释放GPU、不会保证之后有空闲卡；同一实例数据保留。它适合串行下载/轻量校验，不能替代8核32GB的独立CPU准备机。重数据处理需要独立CPU配置或用户确认后的正常开机时段。AutoDL本地盘无冗余副本，关机不等于停止数据盘计费，重要产物另行备份。

核验来源：
- [NVIDIA CUDA兼容范围](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [vLLM Gemma4 E4B配方](https://recipes.vllm.ai/Google/gemma-4-E4B-it)
- [H100规格](https://www.nvidia.com/en-us/data-center/h100/) · [H200规格](https://www.nvidia.com/en-us/data-center/h200/) · [A100规格](https://www.nvidia.com/en-us/data-center/a100/)
- [AutoDL无卡模式](https://www.autodl.com/docs/save_money/) · [磁盘规则](https://www.autodl.com/docs/local_disk/) · [CPU/RAM分配](https://api.autodl.com/docs/env/)
