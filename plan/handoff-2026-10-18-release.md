# 交接:发布收口(2026-10-18 晨 → 新 session)

> 接续入口仍 `plan/STATUS.md`;本文件只覆盖**发布收口的剩余机械步骤**。所有决策已定(榜单口径 ADR0028、serving 形态、发布内容清单),无需再做判断;照做即可。

## 状态总览

- **已完成**:③ serving/网关/浏览器全验收(票 issues/02);④ 量化数据全齐(文本 20/20×3 概率级;音视频/截图矩阵 13/13 EXIT=0,screenshot 需 RGBA→RGB+长边≤1344 预处理,runbook=docs/runbooks/gguf-multimodal.md);⑤ GitHub main=0088b46 已上线(README+LICENSE+4 段 GIF+图表);HF 模型卡/图表/demo 素材已在盒 `s6-hf-repo`(hf 会话 commit e3cfd96/0e847ea 在 release-final 分支待推)。
- **唯一进行中**:盒 screen `s6-remap` 跑 `flat_remap_safetensors.py`(compact→平面分片,日志 `/root/autodl-tmp/s6-hf-repo/remap.log`,成功标志 `FLAT_REMAP_DONE N shards, 2130 tensors`;N 预计 8-9)。**它自己在 hf-repo 写产物,不要动它,不要跑其它含 kill 的脚本。**

## 剩余步骤(按序,全部命令可直接复制)

### 1. 等 remap 完成(~20 分钟)

```bash
ssh seeta 'grep REMAP_EXIT /root/autodl-tmp/s6-hf-repo/remap.log'   # 0 = 成功
```
若 EXIT 非 0:日志在 remap.log,把错误原样排查;**大概率无需改脚本**(已在本地验证逻辑)。

### 2. 重建 serving view 并起栈

```bash
ssh seeta 'screen -wipe >/dev/null 2>&1; screen -dmS s6-rv bash /root/autodl-tmp/s6-release-serving-001/rv.sh'
```
rv.sh = 杀 serve 残留 → 重建 serving-view(硬链接 release 分片+基座 metadata)→ 起 vLLM+网关。
**坑**:rv/任何 kill 脚本必须以"文件+screen"方式跑;ssh 命令行里出现 kill 模式会自匹配自杀(本窗已栽三次)。

验证(约 6-8 分钟后):
```bash
ssh seeta 'curl -s -m 2 http://127.0.0.1:8001/health; curl -s -m 2 http://127.0.0.1:8000/health | head -c 120'
```
8000 应返回 `{"status":"ok","inference_configured":true,"model_version":"767be09b0919235d",...}`。

### 3. 录制最终 demo(4 段,Mac 侧)

```bash
cd /Users/bamboo/Githubs/Lite-OmniJev
ZHIPU_API_KEY='<用最新 key>' .venv/bin/python .scratch/s6-release-serving/demo/record_demo.py
# 产物 .scratch/s6-release-serving/demo/videos/demo-*.webm
# 转 GIF:
cd .scratch/s6-release-serving/demo/videos
for f in text image audio video; do ffmpeg -y -v error -i demo-$f.webm -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer" demo-$f.gif; done
```
**肉眼验收**(必做):每段 = 输入区素材各就各位(image=screenshot.png/audio=可播放控件/video=自动播放)+ 题干选项完整 + 左侧概率条 + 右侧 LLM 流式;text/image 段右侧有真回答,audio/video 段右侧为"无原生通道"说明。
失败排查:页面 JS 状态用 record_demo.py 的 wait_for_function 超时即挂;常见根因=网关没起/新 key 失效/页面 payload 与录制器路由不匹配(录制器 zhipu 路由已适配 anthropic 端点+glm-5.3-flash+content[].type=="text" 解析)。

### 4. 同步 GitHub 与 HF 仓库

GitHub(本地已改好 README 量化矩阵/接入段,提交推送):
```bash
cd /Users/bamboo/Githubs/Lite-OmniJev
git add README.md docs/demo docs/charts && git commit -m "Final release content" && git push origin main
```
HF(把最终素材并入盒上 hf-repo 后 `hf upload` 增量):
```bash
# 本地 GIF → 盒:
rsync -a docs/demo/demo-*.gif seeta:/root/autodl-tmp/s6-hf-repo/docs/demo/
rsync -a .scratch/s6-release-serving/hf-readme.md seeta:/root/autodl-tmp/s6-hf-repo/README.md
# 盒上(hf 会话的 e3cfd96/0e847ea 已在 release-final 分支,把工作树对齐后再传):
ssh seeta 'cd /root/autodl-tmp/s6-hf-repo && screen -dmS s6-hfup bash -c "source /etc/network_turbo >/dev/null 2>&1; export HF_TOKEN=<用户的 write token>; export HF_HUB_ENABLE_HF_TRANSFER=1; /root/autodl-tmp/s6-index-work/env/bin/hf upload Chinzhu/Lite-OmniJev . . --repo-type model > hfup4.log 2>&1; echo EXIT:\$? >> hfup4.log"'
```
注意:hf-repo 里的分片此时应为 remap 产出的平面分片(remap 直接写在 hf-repo);upload 按 sha 去重,已传的 compact 会被新平面文件替换。**token 问用户要,不入仓。**

### 5. 终审 + 关卡

- HF 页面 `https://huggingface.co/Chinzhu/Lite-OmniJev` 肉眼审(README 渲染/文件树/License)。
- GitHub 页面 light+dark 肉眼审(此前已验过一轮,改 demo 后复核 GIF 帧即可)。
- 数字对账:README/HF 卡每个数字能在 reports/ 或 gateway/receipts/ 找到出处。
- 全过后:通知用户关卡;盒上清理(stash/临时 view/日志归档)。

## 硬约束(传承)

- kill 类操作一律"写脚本文件 + screen 执行";ssh 命令行出现进程名模式 = 自杀。
- 盒上权重文件:merged-fp32(6 平面分片,含 24-41 层死名填真源值)是 serving 与分发身份;**不要被任何清理脚本删掉**。
- 数字只出不进;对照不点名厂商;JevBench −9.5 如实在表;DI 上游不可得(公开表不出现该行)。
- token/key 只走 env,不入仓。
