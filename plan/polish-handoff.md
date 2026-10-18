# 打磨协作交接(主会话 → codex"打磨")

日期:2026-10-18。数据已全部采集完毕;本文件是给打磨会话的素材清单与边界。
打磨范围:demo 观感、图表视觉、README/HF 叙事与宣传性、整体无不利呈现。**不得改动数字与事实。**

## 产品事实(打磨的素材边界,全部实测)

- 模型:Gemma-4 E4B + 决策 LoRA(D3P-600,600 样本),单次前向 = 全选项概率 + p(U)。
- Race demo 四段(真实网关录屏,`docs/demo/demo-{text,image,audio,video}.webm`):
  左侧并读概率 vs 右侧通用 LLM 自回归;文本 8 题。
- 实测时延(网关 Server-Timing):文本 8 题 0.13 s / 图像 0.45 s(3 题口径)/ 音频 0.89 s(3 题)/
  视频 16 帧 2.2 s(3 题);demo 内 8 题口径 0.13/3.2/3.3/8.8 s(两套口径都真实,页面展示取实际值)。
- 对照 LLM(公开 API 实测):文本 8 题 2.53 s;图像单题 1.23 s。**叙事不点名具体厂商。**
- 图表:`docs/charts/*.svg`(light/dark 成对):readout-diagram / latency-text / mmau-routing / parity-tile。
- 榜单(诚实呈现,JevBench −9.5 与 MMAU +2.8 并列):见 README 表;MMAU 路由形态 65.9>教师 63.10 是主打声明。
- 量化:GGUF F16/Q8_0/Q4_K_M 文本一致性全 20/20 零退化;Q4_K_M 5.34 GB 端侧首选;MLX 4bit 6.2G 已转换。
- 工程可信:parity 5.75e-6<1e-3、两遍 Δ=0、网关逐位一致、浏览器 live 通过(回执在 HF gateway/receipts/)。
- 许可:CC-BY-NC-SA-4.0(代码+权重);基座 Gemma 各自许可。

## 已知边界(如实,不得隐藏也不必放大)

- JevBench 距教师 −9.5(训练规模差);Decision Index 官方套件上游不可得,打分 pending。
- 音频 p(U) 原始读出偏低(校准层覆盖封存池);llama.cpp 音视频运行验证 pending;截图输入在
  llama.cpp 全档位空输出(上游运行时问题,与量化无关,不进宣传物料)。

## 打磨的具体任务建议

1. demo 页面/录屏的视觉包装:封面帧、标题排版、时钟字色 race 观感(源页面
   `.scratch/s6-release-serving/demo/index.html`,录制器 `record_demo.py` 可重录)。
2. 四张 SVG 的视觉升级(配色已按无障碍验证,改动需重跑 validator;保持"仅实测数字")。
3. README(GitHub)/HF 模型卡的叙事终审:首屏三句话、宣传性标题、结构;删除一切"进行中"噪声
   (DI pending 一处如实保留)。
4. HF 仓 README = `.scratch/s6-release-serving/hf-readme.md`(推送落地后可直接替换提交)。

## 硬约束

- 数字只出不进:任何图表/文案的数值必须能在 `reports/` 或 HF `gateway/receipts/` 找到出处。
- 不点名对照厂商;不利数据不展示但也不否认(JevBench 缺口在表内)。
- 许可与"非商业"表述不得弱化。
