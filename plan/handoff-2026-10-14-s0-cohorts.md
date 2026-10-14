# S0当前接续入口（阶段尚未完成）

2026-10-14更新。先读AGENTS.md、plan/STATUS.md、plan/DELIVERY-PLAN.md和s0:07；本文件只保留接续指令，历史证据见STATUS归档及reports，不重复跑已通过检查。

## 当前事实
教师5600记录/2800对全部一致，MMAU62%通过，MVBench54.222%为ADR0006九任务独立基线；教师已自然exit0。六主体官方CPU能力及文本/音频各500开发首批已完成。100题数值/网关/浏览器已通过，FP32仅数值参考。

用户最新明确授权按实测调整整个方案中的估计性能标准。ADR0010已生效：S0缓存原≤2改为非阻断诊断目标，完整真实native/gateway on/off冷暖双测、KV命中及四概率<.001通过s0:04实测基线；历史未达标结果全部保留。停止S0配置搜索，不宣称原目标或最终产品p95通过。后续性能在对应阶段按实际候选限额测量并自主校准；正确性、质量与隔离不豁免。

## 继续工作
1. s0:07统一推进图像/截图/视频/跨模态开发集及同条件基线，不再按来源/模型拆新票。已完成证据直接复用。
2. executor /tmp/lite-public-baselines-001 代码已合入d23dc39并部署/root/autodl-tmp/s0-development-next-001；E4B/E2B真实231双轮均exit0，分别76.19%/63.64%，两模型各231对概率完全一致，78.944/97.030秒；reports/s0-public-baseline-{e4b,e2b}-001.json。两作业已完成，勿重启。预处理使用官方模板，避免重复BOS；完整两遍逐条概率留存。
3. executor /tmp/lite-media-dev-001 正准备AOK原Choice及CLEVRER exist各500，父合入后Linux实际双构建。媒体资产已在原树，不重下载。
4. GUI与跨模态还缺实际构建；原始step标注首例已核，禁止用粗动作标签假称完整GUI能力。所有组件整family排除，不能跨池。
5. s0:05仍缺外部gate身份与真人双标/κ，agent不能伪造；继续独立工作。性能授权不自动豁免真人门槛。
6. 全部S0依赖完成后执行08，一次集成验收并写正式s0-complete交接与可复制启动词；当前不得冒称阶段完成。

## 资源
SSH root@region-42.seetacloud.com:48926，mux /tmp/omnijev-ssh-root-final，凭据不入仓。H20上述公共基线GPU作业已完成，当前无自有GPU作业，磁盘余391GB。权重及已有媒体/root/autodl-tmp/lite-omnijev；primary Python /root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python。冻结教师/IEEE树/root/autodl-tmp/gpu-reference-fp32-complete-001 HEADdfdfd935不可改代码。不要凭历史PID文件重启已完成作业。

goal active；5b/探索关闭，不push、不自动租卡/开关卡/付费/生产部署。
