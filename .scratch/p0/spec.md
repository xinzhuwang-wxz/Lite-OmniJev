# P0：从无卡准备到真实基座十步闭环

Status: ready-for-agent

## Problem Statement
本机无CUDA且磁盘不能容纳全部基座/教师；CPU单测无法验证真实Gemma训练、logit或vLLM。
## Solution
先在持久无卡Linux实例锁环境和资产，再由用户开卡；最小100样本10步闭环成功后才进入大规模评测/训练。
## User Stories
1. 负责人看见完整下载与校验报告后才开始GPU计费。
2. 执行者能在无卡机预装训练与服务环境，失败可恢复、不重新下载已校验资产。
3. 训练者用程序真值100样本跑10步，观察有限损失、真实梯度和仅LoRA可训练参数。
4. 评测者通过相同模板从HF/vLLM读出完整合法token概率，不能用mock代替。
5. API使用者可经官方SDK形状提交问题，获得版本化结果，未配置模型时清楚失败。
6. 负责人收到GPU用完通知、检查点和续跑命令，决定关卡。
## Implementation Decisions
训练和服务分环境，Python/CUDA/torch/Transformers/PEFT/vLLM各锁定具体版本和镜像digest。先BF16正确性，再FP8性能。
无卡预检只能检验安装/下载/静态兼容；CUDA kernel和显存仍需开卡后实测。
初期只下载当前闭环必要基座E4B，不一次性下载全量教师/数据；S0教师准入后分批。
## Testing Decisions
同数据/种子重复构建哈希相同；错hash/缺文件/权限异常必须非零退出。GPU证明冻结主干、十步训练、checkpoint重载、真实HTTP答案。
## Out of Scope
大规模教师打标、S1训练、生产部署、收费与租卡自动操作。
## Further Notes
科学门槛未过不扩大数据；完整终局仍需要人工封存标注和端侧实机。
