# S0评测合同与开发数据接续（尚未完成阶段0）

## 本轮可复现产物

ADR0009在7cbaeda落地：六评测cohort各500dev/1000holdout、四物理输入不变；dev来源ledger不含题目/标签，人工双标κ≥.7只用于封存。75定向测试、82源mypy、ruff/298format/build通过；合成六类CLI及独立复核证据reports/s0-sealed-v2-{canary,canary-validation,review}-001.json。独立review发现同item跨pool、不同family哈希逃逸，已修复；真实root隔离/人标始终blocked。

CPU基线003在72e5e3546682bbf4d2780912c47da2974dece040、远端/root/autodl-tmp/s0-baseline-cpu-003完整源码树执行，27.967秒exit0、六主体均passed。证据reports/s0-baseline-cpu-linux-003.json；显式Valen路径是artifacts/quarantine/Valen-Team__Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1。路径重映射仍校验全部required metadata size/hash；OneJev false分支与Qwen true/else空thinking分支已实测。只证明text/image/token/thinking，audio/video尚待补探针；专用教师head、Valen组合真实forward及模型质量不凭此准入，s0:10继续claimed。

## 正在做的切片

s0:13已done；实现f8edd50，真实Linux树/root/autodl-tmp/s0-text-dev-001，两遍约2.3sec/exit0。artifacts/text-dev-001-a和-b各有dev.jsonl500、train.jsonl14750、development-ledger.jsonl500、allocation-ledger.jsonl15250。四输出字节精确相同，全部原样本只改pool/version，源文件未变、family交集0，dev覆盖144intent、train151。证据reports/s0-text-development-build-001/002.json、s0-text-development-acceptance-001.json。旧15250全train版本已superseded，未来训练须绑定余train SHA12a42005…8cb11及allocation SHA54b3d818…ab7d51，不凭本票自动准入。仅intent Choice，不代完整文本任务覆盖。10本票/54相关回归、83源mypy/ruff/302format/build通过。

s0:12的实际TRAIN候选、family单位、来源许可/语义缺口在.scratch/s0/development-source-contract.md。GUI固定首TRAIN逐步标注已用Mac官方HTTPS取得22471B、SHAf27cabeeba9f7f79d228cf2979484661b70e9915f2b583fe575f2e10de5f3cef，并SCP双端核SHA；远端/root/autodl-tmp/lite-omnijev/artifacts/s0-gui-step-annotation-audit-001/0000628297785099.json。初次远端直接联网失败收据保留reports/gui-odyssey-step-annotation-audit-001.json。当前/tmp/lite-dev-source-map-001代理只读核动作/schema/16截图对齐，交新local-002报告，不再重复下载。单episode不代表500dev或完整语义通过。

## GPU与下一步

教师resume005是唯一GPU作业，冻结/root/autodl-tmp/gpu-reference-fp32-complete-001 HEADdfdfd935，PID169823；最新快照5275/5600、Rl、43:10。同冻结records路径和3600秒supervisor，不改源码；先核实际PID/exit，活作业不重启，终止124才显式新编号resume。完整5600后读取最终汇总、核两benchmark固定分母/逐题重复/概率/MMAU门槛及独立review，才关s0:02。MVBench按ADR0006独立九任务1800，不伪称发布53.10复现。

缓存最佳冷比2.3164/2.3335仍fail，未启动下一GPU候选；服务已停。TF32x3只完成源评估，若推进必须新隔离编译缓存、精度显式绑定、完整100概率与重复门槛，不能当IEEE相同。先让教师完成。

接手先读AGENTS、STATUS、docs/02-project.md、当前claimed票及本文；先核真实作业/代理，不重复下载、已完成CPU检查或失败GPU配置。S0全绿才执行s0:08并写正式s0-complete handoff和可复制新session启动词；本文不是阶段完成证明。goal仍active，不push、不自动付费/开关卡、不读取封存内容。

GUI小审计local-002已完成：首episode16steps与16实际截图逐一对齐；CLICK12/TEXT2/SCROLL1/INCOMPLETE1，仅11CLICK合法目标bbox。报告reports/gui-odyssey-step-annotation-audit-local-002.json，只1family，没有全候选元素/开放参数语义或500dev准入。继续具体动作/候选任务协议，不把粗动作Choice替代原完整GUI任务。

最新教师快照5422/5600、Rl/48:27、005.exit不存在。当前唯一活跃CPU子切片是/tmp/lite-baseline-cpu-fix-15ea9ab的全类别官方probe扩展，两文件范围不变；父须整合验证后Linux004，不能拿已通过003的text/image替代音视频。下一缓存候选规格.scratch/s0/cache-tf32x3-spec.md已冻结，但未启动；先教师完成，保持所有精度/重复/冷比阈值。
