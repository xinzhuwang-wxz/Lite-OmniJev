# STATUS

更新：2026-10-14。完整goal active；当前s0:07 claimed，S0尚未全部通过，S1未启动。用户要求先完成S0并写新session交接；不细拆/不过度验证。执行路线plan/DELIVERY-PLAN.md。

- 性能/资源估值按用户授权ADR0010实测自主校准，限额优化后推进；准确率、概率正确性、数据隔离、真人证据保留。
- 普通修复只跑相关行为测试，一批必要静态检查；里程碑一次独立review。有效证据复用，不重跑已完成项。
- 5b/探索关闭，不换基座/引擎、不push/付费/自动开关卡/生产部署。

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

## s0:07当前完成与剩余
- 五cohort各500已真实双构建/媒体验证通过；截图v1因坐标捷径撤回，v2双构建exit0、哈希一致/500原family相同。当前数据切片：文本CLINC Choice、图像AOK Choice、截图GUI目标Choice4、音频Clotho一致Noul、视频CLEVRER exist Noul、跨模态image+audio合取Noul。均首primitive切片，不冒称所有任务分布。
- 截图固定前600标注仅497合法，补30固定TRAIN小标注达到500；筛选基于许可/格式/几何，不看模型分数。RGB实际decode通过。
- 六公开模型JevBench231各双遍完成，reports/s0-public-baselines-table-001.json：E4B76.1905%、E2B63.6364%、OneJev4 75.7576%、27B83.1169%、教师85.7143%、Valen47.1861%。低成绩原样保留；专用head无U则null。
- 开发GPU已核24/24非截图支持对，截图v2 E4B已完成51.4%/500×2，教师74.0%/500×2也已完成，Valen73.0%/500×2也完成，OneJev4也完成，累计28/30支持对，只剩E2B/OneJev27截图继续，另6个音频不支持组合明确unsupported。唯一矩阵入口reports/s0-development-baselines-table-001.json；旧截图成绩superseded。
- 教师原六cohort6000记录/3000对、exit0报告保留；其中截图v1已superseded，当前截图v2独立1000记录/500对完成74.0%、exit0，reports/s0-development-screenshot-jev-omni-12b-v2-001.json。其余五类直接复用原报告。
- v1历史统一dev排除清单继续生效，v1/v2已合并历史union，3000节点，SHA3110f9740d9e7dd9c39023b106d977e927b3446f5dfa82e168f729c127aeda1e，reports/s0-development-exclusion-history-v2-001.json。原清单两遍SHA相同：3000family节点/2500原始family/2000组件。跨模态复用原image/audio同池，未来train同时查family和媒体SHA。reports/s0-development-bundle-{001,002}.json；calibration/holdout分配仍pending。
- s0:04 done：完整真实native/gateway cache-on/off冷暖双测、实际KV命中及四概率<.001；原≤2按ADR0010仅诊断目标。FP32 TF32x3失败候选停止，停止S0性能搜索，原失败证据保留；不宣称最终p95通过。
- 07收拢后仍须全部S0依赖通过再认领08，一次集成验收及正式s0-complete交接；当前接续文档plan/handoff-2026-10-14-s0-cohorts.md。

## 当前作业与恢复入口
- 非截图学生队列892905已完成finished；E2B crossmodal50.4%、1000记录/500对精确已核。教师/Valen/video全部结束，不重启。
- 截图v1结构捷径：无图无指令坐标规则497/500=99.4%，reports/s0-gui-coordinate-shortcut-v1-001.json。旧六模型截图成绩全部撤出当前矩阵，保留证据。queue43477及child343447已结束。
- scripts/s0_gui_dev.py v2随机合法坐标修复7相关tests/ruff/mypy通过。真实CPU双构建/root/autodl-tmp/s0-gui-dev-v2-001/build.sh已exit0；产物artifacts/run-{a,b}逐字节一致，500原family保留、498原题/2题因合法几何重选；冻结旧坐标规则19/500=3.8%。630标注/媒体复用，无下载。
- 截图六模型补测已启动：/root/autodl-tmp/s0-screenshot-v2-final-001 HEAD7b4f382；学生queue343888（artifacts/screenshot-baselines-v2-001，四学生顺序、各1800秒/显存等待）；教师→Valen queue343927（teacher-screenshot-v2-001及valen-screenshot-v2-001，各1800秒）。OneJev4 344069已exit0，原queue343888自动开始E2B child344735；Valen344135已正常exit0/finished；教师343936已exit0/281.014秒（cohort244.528秒），E4B343893已exit0/232.000秒；全部只测screenshot v2，不重跑其余模态。源配置已重绑，历史dev排除union通过，07首条恢复勾选。
- 为减少占卡，27B已提前独立并行：queue344403/child344408，artifacts/screenshot-27b-parallel-v2-001，原单job2400秒，启动free76749MiB，当前与OneJev4共用约64GB。交接watcher344402/stop-original-before-27b.py允许原343888继续收OneJev4/E2B的实际kernel退出，E2B终态后停止原队列未来派发，防止27B重复。收据artifacts/screenshot-parallel-handoff-001.json届时生成；接手核实际进程，不能再启动原27B。 根据共享GPU实际前向约3.8秒/题与启动时间，已将同一344408交接watchdog344615/retain-27b-measured-deadline-001.py，总上限4500秒；原parent344403仅暂停计时/采集，child正常计算。模型终态后采kernel wait status并CONT原parent正常记录exit。证据reports/s0-screenshot-27b-deadline-handoff-001.json（当前running，不冒称完成），避免估计timeout截断/重跑。
- 已汇总28对真实token/耗时诊断分布：reports/s0-development-compute-budgets-001.json；复现python3 .scratch/s0/summarize-development.py，只读本地已验证sidecar并绑定报告/记录SHA，不重跑模型。Valen未测的预处理耗时保持null。08已同步ADR0010缓存与07同机双轮/S6跨运行时责任，清除过时额外前置。

- 所有共享GPU作业仅质量评测，耗时是诊断数据，不当隔离发布性能。
- 官方FLA0.5.2单候选231×2质量/重复通过，warm诊断约2.505倍，reports/s0-qwen-fla-candidate-result-001.json。只在后续OneJev作业显式profile/固定wheelSHA启用，primary环境不变，旧baseline保留；不用较弱backend降低目标。
- 教师MMAU/MVBench完整作业及公开基线均已自然exit0；旧PID不代表作业活跃，不重启。Valen四支持类真实CPU首题通过5.342秒；size嵌套schema修复与失败001保留。

## 数据精确路径（/root/autodl-tmp）
| cohort | 冻结产物 |
|---|---|
| text | s0-text-dev-001/artifacts/text-dev-001-a；dev500/余train14750，旧全量15250已superseded |
| image/video | s0-media-development-002/artifacts/media-dev-002/{image-a,video-a}；media_root各自/media |
| screenshot | s0-gui-dev-v2-001/artifacts/run-a；media_root为原资产artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb/screenshots |
| audio | s0-audio-dev-001/artifacts/audio-dev-001-a；原资产artifacts/data/quarantine/clotho-aqa/6473207/approved-audio-002 |
| crossmodal | s0-crossmodal-development-001/artifacts/crossmodal-dev-001-a；media_root为其/media |
| global dev排除 | s0-screenshot-v2-final-001/artifacts/dev-exclusion-history-union-v2-001/dev-exclusion.jsonl；SHA3110f9740d9e7dd9c39023b106d977e927b3446f5dfa82e168f729c127aeda1e |

- 资产根/root/autodl-tmp/lite-omnijev，模型artifacts/models；六cohort完整入口configs/s0-development-cohorts-h20.json。S0全部子切片实现已整合，无活跃子代理待交付；截图v2 GPU作业运行。
- SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final；凭据不入仓。H20 97871MiB、cgroup150GiB/16CPU；开关卡由用户操作。
- primary /root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python（TF5.13/Torch2.11/vLLM.26）；teacher /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python（TF5.17/Torch2.13）。冻结教师/IEEE树gpu-reference-fp32-complete-001 HEADdfdfd935不改代码。
- 原教师5600records双端SHA75a8b653…d14aa，本地artifacts/s0-teacher-complete-001/records.jsonl；历史cost0只是硬编码占位，真实费用未知null，不冒称累计账单。

## 真正外部缺口与待我决策
- s0:05仍blocked：真实外部gate身份、真人原始双标与六类κ汇总未取得，agent不能代签/合成。仍有技术作业，goal不blocked、不complete。
- ADR0011仅proposal，等待用户明确答复：保持6000封存/12000真人判断/逐类κ≥.7与隔离不变，是否仅将资源截止S0移至S4放量前。未批准仍按S0硬门，不把性能授权当真人豁免。
- 跨机/S6真机/额外预算/生产运营在对应节点处理，不提前制造细票或默认采购。GPU用完记录资源并通知用户关卡。
- 历史详细状态在plan/archive/STATUS-before-matrix-consolidation-2026-10-14.md及更早归档，只作历史，不作当前调度源。
