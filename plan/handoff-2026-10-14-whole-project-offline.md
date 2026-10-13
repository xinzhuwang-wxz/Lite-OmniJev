# 无卡准备接续

继续本会话已授权的首发阶段0–6准备；先读 `plan/STATUS.md`、`docs/02-project.md`、当前票。不要开GPU，不push；用户租卡，工程与无卡准备自主推进。

## 本轮交付

- 模型票已关闭：`reports/project-model-offline-acceptance-001.json` 与完整下载receipt；保留150GB checkpoint余量，全项目仍未就绪。
- 数据、CPU与运行时证据直接看STATUS引用；包括AndroidControl/Mind2Web全部hash、CommonVoice、Perception/FunQA/STA、ScreenSpot、LiteRT host、RAGTruth恢复。
- 用户决定首发省略AudioCaps，按 `docs/adr/0004-defer-audiocaps-for-first-release.md` 执行；用户确认DocVQA手续已办，已看到账号激活/登录，不等于原数据下载通过。
- 教师音频空文件问题最小CPU复现成立；`offline-prep:06`已认领，video代理新worktree实现适配。原教师精确MVBench14任务/2786协议仍缺，不能反推删题。

## 接续机制与作业

本会话heartbeat自动接续已创建，ID `lite-omnijev`，每15分钟；仅在有可行动失败、需用户具体决定或全部就绪时通知。全部无卡准备通过后通知可开卡并暂停heartbeat，不能代用户开卡。

Clotho004仍在4并发8MiB Range恢复，最终必须MD5，已排好源锁→重CPU锁的提取链；GUI19.5GB所选train媒体网络改锁外、每成员解压锁内，保留已完成文件；CLEVRER下载/CRC收尾。具体PID/路径以STATUS与真实cmdline为准，不重复启动队列。

runtime代理正在补全阶段锁依赖及root wheelhouse，不接受“未来尚未import所以无需准备”的循环理由；已有env不等于offline重建证明。商业媒体权利和已批准研究评测仍分池。

Chrome扩展连接失败，native间歇无窗口/正文空白；工具故障不等于用户没授权。不读cookie/浏览器私密文件、不绕过安全警告、不重复索要MDC/HLE。Android协议须实际具体接受记录；尚未完成。

## 验证与建议技能

最近主树289测试通过；GUI修正整合后Ruff/format184文件、mypy47源文件通过。新适配提交后跑定向行为测试再批量整合验证。

建议技能：research用于精确官方来源/协议，handoff用于闭环交接；需要调试时用diagnosing-bugs。继续轻量直接执行与独立worktree原生子代理，不启用superpowers/重型OMX流程。
