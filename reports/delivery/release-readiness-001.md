# 三档发布 CPU 准备核验 001

2026-10-16；基于 `d285887e98f95af951f8ea78218f16730c1cf16f`。结论 **blocked / 不可发布**。本片完成资产清单校验，不运行 GPU、训练、模型推理、服务或真机，不改变阶段门槛。

- 已复用：外部模型/训练/服务/量化/端侧依赖已有离线闭包回执。11份回执/当前入口文件按大小与SHA256实核通过；完整清单见 `release-readiness-001.json` 及 `.scratch/delivery/release-readiness/inventory-001.json`。
- 远端实核：llama.cpp四个二进制SHA全部匹配历史回执，`--version` exit0：`0.6.0 (build 24, commit 45ebe32)`；LiteRT-LM v0.9.0 Linux/Android arm64 CLI、NDK r28b版本文件和AArch64 hello探针均仍在，摘要匹配。**llama入口多一层bin**：`/root/autodl-tmp/lite-omnijev/artifacts/runtime-builds/bin/llama.cpp-v0.6.0-cpu/bin/llama-cli`。
- E4B `ee0ef6023621cff504d758262d4e04895a5af4a2` / E2B `3e22461f65e89153144f8adb70e3b8c2cc9845a7` 基座目录存在；config与官方template摘要已刷新，权重分别15,992,595,884 / 10,246,621,918B。26.24GB权重未重复哈希，沿用既有资产回执；不把基座缓存当蒸馏模型。
- NDK授权/下载不是当前阻塞：旧runtime报告的license blocker由android-tools-preparation和project-runtime-offline-acceptance覆盖。Android CLI是已有端侧准备路径，AAR直接依赖缓存存在；没有选定Gradle app，不扩出APK/Gradle完整图。

## 仍缺的具体交付

| 档位/阶段 | 当前缺口 | 最早CPU可闭合路径 |
|---|---|---|
| S4前置 | 质量未过；最新已知full视频356/500，目标至少457/500 | 不可由CPU清单补足；等待有效候选与正式独立gate |
| API / S5 | 没有温度拟合/保存/版本绑定加载入口；网关仍`calibration=unfitted`，`model_quality_verified=false` | 先实现六模态独立校准契约/fit/save/load、绑定model/template/token-map及失败测试，再扩可信私有作业S5汇总；实际fit/holdout依赖合格S4模型 |
| 单卡 / S6 | FP8/W4A16/GGUF产物待生成；最终版本耦合启动/回滚包未实现 | 使用现有量化env和llama转换器，先冻结bundle清单/启动参数/失败与回滚行为；量化掉分和概率验收仍须真实模型 |
| E2B / S6 | 合格蒸馏/合并/量化决策权重待生成；三运行时完整原始答案logit与媒体适配/同200题对照缺失 | 沿现有固定引擎实现决策adapter/完整legal-token读取和媒体失败拒绝；不能以生成top-k代原始概率 |
| 端侧 / FINAL | 无真实Android设备执行/内存/时延证据；Playground正式业务链、最终模型卡/数据卡/许可证/性能/回滚材料待交付 | CPU先打包已选CLI及hash清单；设备/adb测量窗口必须真实具备。当前CPU构建不能替代设备验收 |

没有S5/S6专用执行runbook；现有`offline-runtime-preparation.md`和`runtime-cpu-builds.md`只负责工具准备。`scripts.parity_calibration`是100题数值对照，不是六模态S5拟合。开发服务runbook不是三档最终发布入口。最后6小时不能据此推断足以完成剩余模型/工程/真机门槛。

## 本片可复现闭环

使用项目已有 `scripts.assets.verify_file` 校验文件，无新增依赖/生产模块改动。清单保留10个待生成/待验收项，命令返回 **exit1 / inventory_incomplete** 是正确结果；即使文件全齐，也仅返回`inventory_verified`，不声明阶段通过。

```sh
PYTHONPATH=. python .scratch/delivery/release-readiness/check_inventory.py \
  --manifest .scratch/delivery/release-readiness/inventory-001.json --root .
PYTHONPATH=. python -m pytest .scratch/delivery/release-readiness/test_check_inventory.py -q
```

实际4个公共CLI行为测试通过（完整清单只声明integrity、pending拒绝、篡改/缺文件拒绝、越界路径拒绝）；ruff check/format check通过。`python -m scripts.assets verify --help`及`python -m scripts.private_stage_eval --help` exit0，只证明入口参数可用。未重复全套测试/重建环境。

历史失败保留于JSON：远端login PATH无python3（127），改显式已锁解释器；首次误把llama包根作二进制路径（1），定位实际`bin/`后版本及四hash通过。无付费、下载、生产写入、公开push。成本未获实际价格，保持null。
