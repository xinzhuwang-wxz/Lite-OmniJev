# STATUS

更新：2026-10-15（Asia/Shanghai，16:25恢复）。**S1已通过；S2 s2:01 claimed，v2图像已完成（78.8%、cf88.6%、双遍一致），截图/文本链在H20运行。发现并修复反事实统计错误：旧ADR0014结构性结论失效，S2未关闭、S3依赖未解除。**

## 当前结果（复用已有证据）
| 闭环 | 证据与边界 |
|---|---|
| S1全门 | reports/s1-stage-acceptance-001.json；dev500 99.6%；ADR0013。 |
| S2 v1准确率 | image 77.2%、screenshot 94.6%、text 99.6%；image仍未达80.6%。 |
| 反事实修复 | bbdc041：置换字母必须用置换选项解码。两个报告级测试修复前失败，修复后相关65项通过、ruff/mypy通过。 |
| 历史GPU预测重算 | reports/s2-counterfactual-rescore-001.json；原数据/sidecar哈希验证；v1 image 91.8%、screenshot 96.6%（双遍相同）；v3探针92.0%/94.67%、零样本image87.33%。仅修正汇总，不冒充新GPU评测。 |
| S3前置 | reports/s3-joint-preparation-001.json：21184行联合池+500行均衡探针，两次数据哈希一致、五ledger核验通过；配置与脚本在.scratch/s3/；未启动训练。 |

## 资源与恢复
- 原生goal active；当前票owner main-session，工作树 /Users/bamboo/Githubs/Lite-OmniJev。
- SSH mux /tmp/omnijev-ssh-root-final 仍有效，端口48926；凭据不入仓。
- 远端 /root/autodl-tmp/s1-text-lora-001 已fast-forward至bbdc041，未stash；冲突的四份未跟踪原报告保存在 /root/autodl-tmp/s2-evidence-recovery/pre-bbdc041/。
- v2 adapter artifacts/s2-text-lora-001/joint-v2-001/adapter（1600步）复用，不重训。
- 运行链 /root/autodl-tmp/eval-chain.sh；日志 artifacts/s2-text-lora-001/v2-eval-corrected-001.log；启动PID32732，image初始PID32739，GPU占用15864MiB（16:22核验）。
- 输出 reports/s2-dev-eval-{image,screenshot,text}-v2-corrected-001.json；输出目录 dev-eval-{cohort}-v2-corrected-001。旧path-failure报告与旧目录保留。
- S3 CPU预检PID33718，脚本 /root/autodl-tmp/s3-preflight.sh；报告 reports/s3-probe-preflight-001.json；不做GPU训练。
- 教师env python用于评测；SwanLab既有项目maxen/Lite-OmniJev，key仅远端项目外。

## 当前下一步
1. 截图/文本链PID32737完成后，已排队的 /root/autodl-tmp/s2-midway-probe.sh（PID33584）自动执行v2 step800的image150单遍探针，日志 midway-v2-probe-001.log；不重复启动。
2. 三组报告齐后拉回并按原configs/gates.json S2门判定；image全量78.8%已证实未过80.6%，cf88.6%通过。对比step800探针与v2最终同150题切片，改善才扩大评测。
3. image若仍<80.6%，保持失败并定位改进；不得用ADR0014或性能校准豁免质量门。
4. S2原门全通过且训练/集成证据齐全后才关票，再接S3；评测链运行时禁止替换远端代码。
5. 旧交接plan/handoff-2026-10-15-s2-miditeration.md中的cf结论与直接关票指令已过时，以本状态和ADR0014失效说明为准。

## 待我决策
暂无新增账号/租卡请求。GPU正在评测；不自动关卡，结束后记录资源并通知。

## 执行约束
不push/付费/自动开关卡/生产部署；5b/探索不开；先小批量验证再全量；质量/许可/隔离门槛不降低。开发集可读，封存集只接收独立gate汇总。
