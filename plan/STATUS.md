# STATUS

更新：2026-10-14；用户要求先完成S0并写无缝新session交接，规划和验证保持粗粒度。完整目标仍active，S1未启动，S0未全部通过。

## 当前执行方式
- 路线唯一入口：plan/DELIVERY-PLAN.md。剩余数据与基线统一s0:07；性能s0:04已收口；外部封存资源s0:05；最终S0出口s0:08。
- 原06合并05、原12合并07，历史在.scratch/s0/archive，所有未完要求保留；14已完成，后续不按来源/模型/探针另拆票。
- 普通修复只跑相关行为测试；一批合并后必要静态/构建；独立review集中里程碑，复用代码/输入未变的有效证据。当前不再增加流程/反复全检。
- 不换基座/引擎、估计性能按用户授权ADR0010实测校准，质量/正确性保留；5b/探索关闭，不push/付费/自动开关卡/生产部署。

## 已通过（复用证据，不重跑）
| 闭环 | 结果与证据 |
|---|---|
| P0与离线准备 | 真模型10步/P0工程闭环、7模型70文件121.09GB、29来源与offline runtime已准备；reports/p0-engineering-acceptance.md及project-*-offline-acceptance-001.json。资产不是质量准入。 |
| s0:01 教师技术 | 安全head转换、原tensor GPU功能等价与公开text/audio/video两遍；reports/teacher-technical-admission-gpu-001.json。 |
| s0:02 教师效果 | 5600记录/2800对完全一致，MMAU两次620/1000=62%，MVBench九任务两次976/1800=54.222%独立baseline；独立PASS，reports/s0-teacher-complete-acceptance-001.json。005终止exit0，不重启。 |
| s0:03 数值/网关/浏览器 | FP32 IEEE HF/native100与gateway100严格<.001、重复精确；Chromium153六类×三primitive通过；reports/s0-full100-gpu-acceptance-001.json。仅数值参考，不当最终服务性能。 |
| s0:10 官方CPU能力 | Linux004六主体全部declared类别通过32.917秒；OneJev audio metadata_inconsistent、Qwen audio未声明不伪passed；reports/s0-baseline-cpu-linux-004.json及.scratch/s0/baseline-support-matrix.md。 |
| s0:11 JevBench | 固定231公开题、可逆标签/官方计分、两遍构建hash相同；教师195口径未证实，oracle不是模型成绩；reports/jevbench-adapter-*.json。 |
| s0:13 文本开发首批 | 原CLINC500dev/14750余train，family交集0、两遍字节相同，dev144intent；reports/s0-text-development-acceptance-001.json。仅intent Choice，旧15250全train已superseded。 |
| s0:14 音频开发首批 | 原Clotho一致Noul500/500sound families，500WAV完整PCM验证、最长29.968秒，yes314/no186；973train family中500排除/473余；两遍字节相同。reports/s0-audio-development-acceptance-001.json。仅Noul，非真人κ。 |
| 六cohort gate工程 | ADR0009，dev3000合法gold/program truth、holdout6000×两人=12000判断，κ≥.7仅holdout；同item跨池漏洞已修复，合成CLI及独立CLEAR。真实资源未取得。 |

## 正在做与下一动作
- s0:07 claimed：整合剩余image/screenshot/video/crossmodal开发500与必需同条件基线；不再另拆小票。来源事实见.scratch/s0/development-source-contract.md。
- 图像/视频builder已合入d23dc39；/tmp/lite-media-dev-001 executor在新冻结remote树执行image-a,b/video-a,b真实CPU500构建与输出hash核对（OMP2），不重复下载。/tmp/lite-gui-dev-001 executor推进GUI目标定位Choice500，限600份原TRAIN小标注，无新媒体/GPU。
- 新实现已合入d23dc39并同步/root/autodl-tmp/s0-development-next-001；E4B preflight-002真实CPU通过（001仅源目录笔误失败保留）。E4B GPU双轮231题exit0，78.944秒，accuracy176/231=76.1905%，231对raw/conditional完全一致，reports/s0-public-baseline-e4b-001.json；同supervisor已依序E2B（各1200秒上限）；每条记录增量JSONL留存。artifacts/public-baselines-001/{e4b,e2b}-run-001.{json,jsonl,log,exit}，先核实际状态，不重复启动。
- s0:04 done（ADR0010用户授权调整预估性能验收）：FP32 TF32x3完整100精度通过，四概率max3.37e-5/两遍精确；cache-on冷比2.3144/2.2861仍>2，四概率漂移<2.91e-10、KV288/实际inflight9。已停止两自有服务，失败候选已停止。reports/s0-tf32x3-cache-result-001.json。
- 原≤2未达到，现作为诊断目标；真实native/gateway on/off冷暖双测、KV命中及四概率<.001构成S0已通过基线，reports/s0-cache-measured-baseline-acceptance-001.json。停止S0性能调参，S6按实际产品候选校准性能标准；不宣称原目标或发布p95通过。
- s0:08 ready但依赖未齐，不可认领。全部门槛通过后一次集成review、正式plan/handoff-日期-s0-complete.md及可复制启动词，再交新session进入S1。

## 资源与精确路径
- SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final；凭据不入仓库。H20 97871MiB/driver580.65.06，cgroup150GiB/16CPU，盘余391GB（2026-10-14本轮实测）。开关卡由用户操作。
- 原资产树/root/autodl-tmp/lite-omnijev，模型在artifacts/models；Valen在artifacts/quarantine/Valen-Team__Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1。
- 冻结教师/IEEE参考树/root/autodl-tmp/gpu-reference-fp32-complete-001 HEADdfdfd935；教师完整records仍叫artifacts/teacher-benchmark-pilot-001.jsonl，本地copy artifacts/s0-teacher-complete-001/records.jsonl，双端SHA75a8b653…d14aa一致。不要改冻结树代码。
- primary Python /root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python（vLLM.26/TF5.13/Torch2.11）；teacher env /root/autodl-tmp/envs/before-gpu-teacher-job-final-001（TF5.17/Torch2.13）。
- 文本实际产物/root/autodl-tmp/s0-text-dev-001/artifacts/text-dev-001-a；音频/root/autodl-tmp/s0-audio-dev-001/artifacts/audio-dev-001-a；未来训练必须绑定对应allocation/exclusion ledger，不能继续旧全量train。
- TF32x3失败原始证据/root/autodl-tmp/s0-scheduled-gpu-001/artifacts/s0-cache-tf32x3-001；API171528/193071已TERM、教师169823已自然exit0，旧PID不可凭文件判活。
- 接续入口plan/handoff-2026-10-14-s0-cohorts.md。历史详细状态plan/archive/STATUS-before-consolidation-2026-10-14.md，不作为当前调度源。

## 真正外部缺口
- s0:05 blocked：真实外部gate身份、两人原始双标与六类κ汇总未取得，agent不能代签/合成。原门槛仍在S0；若要变更阶段位置/数量/阈值须用户明确决定，当前没有自动豁免。
- 跨机重跑/S6真机/额外放量预算/生产运营到其实际节点再处理，不提前制造细票或默认采购。当前仍有可独立执行工作，goal不blocked、不complete。
- 原005 runtime cost=0.0只是历史硬编码占位，真实费用未知null；未来runner已修null，3280.63秒仅最后一次resume外层时间，不冒称全项目GPU时间/账单。
