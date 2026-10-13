# P0 无卡准备交接

下一次由用户完成开卡后，执行真实E4B十步→checkpoint/adapter重载→HF/vLLM基座及adapter对照→API浏览器验收。先读plan/STATUS.md与AGENTS.md。

无卡结果：reports/no-gpu-readiness.json。15项必需检查通过；真实模型训练0步、GPU未验证。模型/环境/样本/代码哈希、SwanLab真实准备记录均在该报告引用的文件中，不重复复制证据。
实例与持久目录见STATUS，具体环境与命令见docs/runbooks/autodl-no-gpu.md、training.md、p0-parity.md、service-environment.md。
凭据仅远程项目外用户配置目录保存，不在本交接复制。

## 下一轮顺序
1. 用户开卡后检查nvidia-smi、BF16 CUDA算子；核对实例资源和实际报价，禁止自行改变租赁电源状态。
2. 使用infra/bootstrap/run-p0.sh，GPU初试训练超时1小时，监控项目Lite-OmniJev/maxen；完成10个真实优化步才记录训练成功。
3. HF与vLLM分时运行，先基座再adapter；对照脚本及实际启动记录要求见p0-parity.md。服务端资产身份由本机已校验启动命令证明，OpenAI接口本身不证明权重hash。
4. 浏览器按docs/02-project.md已实现范围验收；记录真实HTTP/概率/版本/错误路径。不能用CPU或准备SwanLab记录代替GPU证据。
5. 保存产物hash与持久路径，结束自己启动的GPU作业，通知用户本阶段可关卡；关卡操作仍由用户做。

断点恢复已做CPU等价性和损坏/错配/旧checkpoint拒绝测试，独立审查见reports/p0-preparation-review.md；真实E4B GPU恢复仍待验。
CLINC公开train只在raw区，后续映射票.scratch/s0-data/issues/01-clinc-admission.md；没有读封存集，没有下载OneJev混合整库或声称E2B/教师已准备。

## Suggested skills
- 继续按项目已配置的to-spec → to-tickets → triage，在P0实测通过后展开S0。
- 遇实际GPU错误再用diagnosing-bugs；模型优化须先定义指标与退出条件，再按autoresearch执行。
- 只在P0里程碑结束/新高风险点做review；闭环结束用handoff更新plan目录。
- 不启用superpowers或重量级OMX运行时。
