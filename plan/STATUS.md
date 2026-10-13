# STATUS

更新时间：2026-10-13 20:55（Asia/Shanghai）
当前里程碑：P0无卡准备完成，等待用户开GPU；M0完成，真实训练仍0步。
终局目标不变；native goal工具仍返回blocked，工具不支持主动resume，不能将P0准备完成当终局complete。

## 当前票
- bootstrap:01–08、p0:01、p0:02：done。
- p0:03：ready-for-human / blocked，等待用户开GPU；agent不得代开。
- p0:04：open，依赖p0:03；真实十步→重载→HF/vLLM→API集成。
- s0-data:01：open，依赖p0:04，CLINC题型/样本族映射准入后续。
- delivery:01及S0–S6：未完成；原设计门槛保持。
- 本轮子代理全部结束并整合；无活跃开发认领。

## 实例与目录
- SSH：root@region-42.seetacloud.com:25624；用户授权接入、下载、无卡准备。
- Ubuntu22.04 x86_64，0.5CPU/2GiB，无GPU设备；系统30GiB。
- 数据盘/root/autodl-tmp，实际1050GiB（免费50+付费1000），约1010GiB可用；覆盖之前450GB选型。
- 远程项目/root/autodl-tmp/lite-omnijev；训练.venv（Python3.11.17）；服务/root/autodl-tmp/envs/serve（Python3.11.14）；数据/root/autodl-tmp/envs/data。
- uv缓存/root/autodl-tmp/cache/uv；Python/root/autodl-tmp/python；模型artifacts/models，原始数据artifacts/data/raw。
- SSH密码未落盘；SwanLab key仅远程/root/.config/lite-omnijev/swanlab.key（0600），不得输出；运行时读取。

## 无卡验收
- reports/no-gpu-readiness.json：15项检查全部通过；GPU/training_steps明确false/0。
- E4B revision ee0ef6023621cff504d758262d4e04895a5af4a2：8文件16,024,822,159字节完整校验；reports/e4b-full-manifest.json保留上游与实际hash。
- 权重结构：2130个BF16 tensor，参数7,996,157,418；只读header，无模型前向。
- Linux训练Torch2.14.1+cu130/Transformers5.19.0/PEFT0.21.2/SwanLab0.10.2；服务vLLM0.26.0/Torch2.11.0+cu130，独立环境依赖/导入通过。
- Linux与本地136项测试全部通过，无skip；锁定wheel CLI另7项；Ruff/mypy23文件/build通过。
- 33答案token、官方模板thinking关闭、100条样本编译通过；最长77tokens。
- reports/linux-train-preflight.json退出1，唯一阻塞为CUDA未启用，这是预期无卡结果。
- checkpoint保存优化器/CPU与CUDA RNG/训练状态，文件hash校验；错配/损坏/旧checkpoint提前拒绝，保留原报告。独立复核见reports/p0-preparation-review.md。
- HF/vLLM分时对照命令已准备，原始概率与条件答案/U均须严格<1e-3；真实对照未执行。
- SwanLab真实鉴权/准备指标写入通过：https://swanlab.cn/@maxen/Lite-OmniJev/runs/6hi6om41 （不是训练run）。凭据泄漏扫描通过。
- CLINC公开train 15,250行、312,096字节及许可已保存，SHA匹配；仅raw验证，未训练准入。没有读取holdout。

## 待我决策
- D001已解决：无卡实例已接入。
- D003（现在）：用户开GPU，开好后通知agent。预计先预留约1小时最小试跑，训练入口硬超时1小时；真实吞吐尚未测，不折算H100时。
- S4放量预算、阈值/方向改变、生产发布再提交用户；不提前默认授权。

## 下一步
1. 用户开卡后核对型号/驱动/显存，执行CUDA/BF16检查。
2. 运行infra/bootstrap/run-p0.sh；按docs/runbooks/training.md验证10步、保存、重载、实际GPU恢复。
3. 按docs/runbooks/p0-parity.md分时执行HF/vLLM基座与adapter对照，再真实API/浏览器验收。
4. 保存产物和资源清单，通知用户GPU用完可关卡；P0全通过才展开S0。

## 资源与Git
- 下载/安装作业已结束；不重复启动。保留数据盘模型、缓存、环境和下载日志；不清理用户Jupyter等原有进程。
- 仅本地提交和SSH传输Git bundle，未push、未创建GitHub issue/PR。
- 本轮接续：plan/handoff-2026-10-13-no-gpu.md；通用入口plan/START.md。
