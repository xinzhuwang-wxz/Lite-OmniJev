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
| 图像/视频开发首批 | 原AOK Choice500、CLEVRER exist Noul500，逐媒体decode/hash与整family排除；Linux两遍输出字节一致，reports/s0-{image,video}-development-acceptance-001.json。 |
| 跨模态开发首批 | 图像+音频合取Noul500，yes250/no250；1000媒体hash通过、500组件配对、双构建一致；复用已在dev的组件并关联family，不虚增独立family。reports/s0-crossmodal-development-acceptance-001.json。 |
| 六cohort gate工程 | ADR0009，dev3000合法gold/program truth、holdout6000×两人=12000判断，κ≥.7仅holdout；同item跨池漏洞已修复，合成CLI及独立CLEAR。真实资源未取得。 |

## 正在做与下一动作
- s0:07 claimed：六cohort各500已实际构建/双轮确定性通过；剩余必需同条件真实基线；不再另拆小票。来源事实见.scratch/s0/development-source-contract.md。
- 图像/视频实际Linux双遍已完成；crossmodal实际双遍也已完成，均每类500。截图Choice4也已实际双构建/500RGB decode/整episode排除；前600标注仅497合法，补30固定TRAIN小标注达到500，未按模型表现选题。
- 新实现已合入d23dc39并同步/root/autodl-tmp/s0-development-next-001；E4B preflight-002真实CPU通过（001仅源目录笔误失败保留）。E4B GPU双轮231题exit0，78.944秒，accuracy176/231=76.1905%，231对raw/conditional完全一致，reports/s0-public-baseline-e4b-001.json；E2B也已exit0，97.030秒，147/231=63.6364%，两遍231对完全一致，reports/s0-public-baseline-e2b-001.json。E4B/E2B两作业均完成；新批OneJev/教师公开基线见下条；每条记录增量JSONL留存。artifacts/public-baselines-001/{e4b,e2b}-run-001.{json,jsonl,log,exit}，先核实际状态，不重复启动。
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

## 当前活跃作业（接手先核实际状态）
- GPU公共基线batch002已全部exit0：OneJev4B75.7576%/142.934秒，27B83.1169%/647.003秒，Jev-Omni85.7143%/125.720秒，各231对概率精确一致；连同E4B/E2B五核心表reports/s0-public-baselines-table-001.json。旧supervisor287865已完成，不重启。
- 当前开发GPU队列：/root/autodl-tmp/s0-development-baselines-002 HEAD731c3d2，scheduler391342；文本E4B93.6%/ECE.04346、OneJev4B96.6%/ECE.06404、E2B88.4%/ECE.10150已各500×2通过实际输入/概率精确一致（reports/s0-development-baselines-table-001.json）；其后按text→image→audio→video→crossmodal，各E4B/OneJev4/E2B/OneJev27（OneJev音频类别明确unsupported，不填零），每job1200秒，JSONL保留，artifacts/development-baselines-001。原生CPU真实首题16支持对全部通过（4音频不支持对明确列出），教师5类首题CPU官方processor通过6.677秒，见两份cpu-preflight001报告，非GPU质量验收。
- 开发runner已合入731c3d2，流式逐题处理适应150GiB上限，8192token拒截断，重复还绑定实际tensor输入hash；相关10测试通过。教师开发runner已合入05e29ec，同16帧端点与音频PCM16共同输入协议，当前实际GPU已启动。
- Valen开发adapter已合入05cb978/18b5b45，子代理切片完成；实际图像参数schema修复后CPU四支持类别首题通过5.342秒。已找齐官方固定源码16小文件/59561B，四原固定SHA全匹配，artifacts/valen-source-audit-42a6-001/retrieval.json；安全转换权重原已存在，不重做转换、不执行上游pickle/module。
- GUI与crossmodal子切片都已完成，不重复执行。截图报告reports/s0-screenshot-development-acceptance-001.json；实际/root/autodl-tmp/s0-gui-dev-001/artifacts/run-k-actual-001与run-l-actual-001，source脚本SHA在报告，source实际执行SHA保留在报告及远端冻结树；主树随后仅ruff格式化，不能用新字节冒称原运行SHA。学生旧GPU队列仅捕获五cohort，截图需另外调度，不能误称已评测。
- image首跑无path字段KeyError保留；修复6e9c467实际双遍已通过，remote /root/autodl-tmp/s0-media-development-002。跨模态remote /root/autodl-tmp/s0-crossmodal-development-001 HEADc4ce657。

## 待我决策（已准备具体方案，尚未生效）
- docs/adr/0011-human-holdout-stage-placement-proposal.md：是否保持6000封存/12000真人判断/逐类κ≥.7与隔离不变，仅将人工资源交付截止从S0移到S4放量前。当前没有自动豁免；未批准仍按S0硬门。技术独立工作继续。

## 本批新增实测
- 六公开模型基线齐全：Valen实际231×2/462记录精确，109/231=47.186%，88.279秒；保留真实较低成绩，不用Qwen裸基座冒充。reports/s0-public-baseline-valen-001.json。
- 开发基线已核验9对：text四模型93.6/96.6/88.4/98.2%；image四模型74.4/83.6/69.8/87.4%；audio E4B63.4%；各500×2实际输入/概率精确一致。剩余旧队列继续，不重跑已完成对。
- 官方FLA单候选完整231×2通过，准确率仍75.7576%，warm forward中位数诊断快2.505倍；可用于显式新profile，旧环境/当前队列保持冻结。共享GPU时延不当发布性能；reports/s0-qwen-fla-candidate-result-001.json。

- 教师六cohort真实GPU已启动：冻结/root/autodl-tmp/s0-development-final-001 HEAD0aec3eb，artifacts/teacher-development-001/result逐cohort增量、run.exit/finished权威；2400秒作业上限，共享GPU质量专用。六类真实首题CPU已通过6.352秒（含新截图），reports/s0-teacher-development-cpu-preflight-002.json；初次仅报告目录不存在的控制失败已识别，未启动GPU重复。
- 截图四学生queue692449已调度/root/autodl-tmp/s0-development-final-001/run-screenshot-queue-001.py；等待旧学生队列结束/明确停止且教师finished后串行执行，artifacts/screenshot-baselines-001。OneJev显式FLA profile/hash，每job1200秒，意外失败停止保留，不以exit0冒称全矩阵完成。

## 当前队列接续补充
- 学生原scheduler391342已SIGSTOP，仅暂停未来派发，当前video E4B子进程653655继续。替代supervisor892905在/root/autodl-tmp/s0-development-final-001/run-development-queue-fla-002.py等待同一子进程终止，保存kernel waitstatus/退出码后结束旧scheduler；已完成题不重跑。其后video OneJev4/E2B/OneJev27及crossmodal四模型转独立artifacts/development-baselines-fla-002，每job1800秒，OneJev显式FLA profile。新队列完成再写旧supervisor.finished释放截图队列；失败明确写旧queue-blocked并留证据。不能仅凭原scheduler停止误判GPU job已停止。
- Valen开发queue796345已运行：/root/autodl-tmp/s0-valen-development-final-002 HEAD18b5b45，artifacts/valen-development-001；text/image/screenshot/video各500×2，原torch fallback保持与公共Valen同配置；audio/crossmodal明确unsupported；共享GPU质量专用、每job1200秒。text500×2已核sidecar精确一致，35.8%，已加入通过表。
- 六cohort统一dev排除清单实际双构建通过：/root/autodl-tmp/s0-valen-development-final-001/artifacts/dev-bundle-{001,002}/dev-exclusion.jsonl，SHA64d96a…1e5ed相同；3000 family节点/2500原始family/2000组件，跨模态与原image/audio同池连接。reports/s0-development-bundle-{001,002}.json；未来train须检查family与媒体SHA，calibration/holdout仍pending。

- 学生E4B视频已正常exit0：500×2实际输入/概率精确一致，55.6%，797.732秒（共享GPU、非发布时延）；已收拢reports/s0-development-video-gemma4-e4b-it-001.json。kernel waitstatus=0收据已保存，旧391342已结束，892905接续后续FLA作业。开发通过表当前12支持对，E2B音频44.2%同样保留。

## 最新接续修正（以此替代旧控制PID/截止）
- 开发通过表已核17/30支持对；其余13明确列在reports/s0-development-baselines-table-001.json，6个音频不支持组合不填零。新增Valen image66.4%及教师text93.8/image78.6/screenshot89.8/audio38.0，各500×2原始概率/媒体精确重复。教师这四份是已完成cohort证据，父作业未终止，不能冒称全教师完成。
- 教师实际video约3秒/题，原40分钟统一上限会截断固定1000前向；已按ADR0010调整运行估值，总模型上限5400秒。watchdog48305在run-teacher-deadline-handoff-001.py，暂停原timer675329但同一模型675330持续；模型终止后保存kernel退出码至model.exit/deadline-handoff-001.json，再结束旧timer。原shell的控制退出137保留，不代替model.exit，也不当模型失败/通过。所有质量/分母不改。
- 截图原692449只等待、无子进程/无任何GPU记录，已终止并留waiting-controller-replaced.json；新60167执行run-screenshot-queue-002.py，输出artifacts/screenshot-baselines-002，等待上限7500秒适配实际视频队列；每模型GPU仍1200秒。避免旧估计等待上限导致未开始项被误阻断。
