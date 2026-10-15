# Lite-OmniJev S2 迭代中交接

2026-10-15(Asia/Shanghai)。**S1已关闭全门通过(dev500 99.6%);S2处于s2:01迭代中——screenshot/text/文本回归达标,image差3.4pp待v2全量判定,反事实门按ADR0014如实FAIL;v2 adapter已训好(远端),只差三个dev评测+门槛判定+关票;S3全部前置已备。** 新session从"先执行"第4步继续,不重跑任何已完成的训练。

## 先执行

1. 在 /Users/bamboo/Githubs/Lite-OmniJev 读 AGENTS.md、plan/STATUS.md、本文件、docs/adr/0013(实测校准)与0014(反事实结构缺口)。
2. `git status --short`、`uv run python -m scripts.loop validate`(s2:01 claimed=继续它;本地main领先远端,含frame-AB e8e51e8等未同步提交)。
3. SSH:`sshpass -e ssh -p 48926 -o ControlMaster=auto -o ControlPath=/tmp/omnijev-ssh-root-final -o ControlPersist=8h -fN root@region-42.seetacloud.com`(密码向用户要,凭据不入仓;mux 8h过期或pkill误伤后重建)。GPU应空闲;不自动开关卡。
4. **补跑S2三评测**(adapter=/root/autodl-tmp/s1-text-lora-001/artifacts/s2-text-lora-001/joint-v2-001/adapter 已存在;脚本一律本地生成含字面绝对路径后scp——教训见"已知坑"):

```bash
cd /root/autodl-tmp/s1-text-lora-001 && source /root/.s1-swanlab-env
PY=/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python
MODEL=/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2
ART=artifacts/s2-text-lora-001
$PY -m scripts.s1_dev_eval --model-dir $MODEL --adapter $ART/joint-v2-001/adapter \
  --data /root/autodl-tmp/s0-media-development-002/artifacts/media-dev-002/image-a/dev.jsonl \
  --media-root /root/autodl-tmp/s0-media-development-002/artifacts/media-dev-002/image-a/media \
  --passes 2 --counterfactual --output-dir $ART/dev-eval-image-v2 \
  --report reports/s2-dev-eval-image-v2-001.json
$PY -m scripts.s1_dev_eval --model-dir $MODEL --adapter $ART/joint-v2-001/adapter \
  --data /root/autodl-tmp/s0-gui-dev-v2-001/artifacts/run-a/dev.jsonl \
  --media-root /root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb/screenshots \
  --passes 2 --counterfactual --output-dir $ART/dev-eval-screenshot-v2 \
  --report reports/s2-dev-eval-screenshot-v2-001.json
$PY -m scripts.s1_dev_eval --model-dir $MODEL --adapter $ART/joint-v2-001/adapter \
  --data /root/autodl-tmp/s0-text-dev-001/artifacts/text-dev-001-a/dev.jsonl \
  --passes 2 --output-dir $ART/dev-eval-text-v2 --report reports/s2-dev-eval-text-v2-001.json
```

5. 门槛判定:`uv run python -m scripts.s2_gate --image reports/s2-dev-eval-image-v2-001.json --screenshot reports/s2-dev-eval-screenshot-v2-001.json --text reports/s2-dev-eval-text-v2-001.json --smoke-run reports/s2-smoke-run-001.json --output reports/s2-stage-metrics-001.json`;阈值=直接用configs/gates.json的stages.S2原值生成reports/s2-stage-thresholds-001.json(image_vs_onejev min −0.03 / screenshot_vs_onejev min −0.03 / text_regression max 0.01 / counterfactual_flip_rate min 0.8);`uv run python -m eval.gate`判定。
6. 判定预案:cf门按ADR0014保持FAIL(不阻塞S3,列开放风险);image若≥80.6%达标,若仍<80.6%如实记录(对照探针趋势78.0%决定是否再迭代,继续按小批量流程);text_regression的metrics键=S1基线0.996−acc_text(≤0.01达标)。
7. 收口:拉回全部s2报告入本地reports/;s2:01关票(Evidence行绑SHA);STATUS更新;commit(ADR0014已在21a7bbb)。
8. S3接卡:先把本地main同步远端(bundle:`git bundle create /tmp/x.bundle 207bdf6..main`,远端`git fetch&&git stash -u&&checkout`后**务必恢复stash里reports未跟踪文件**,见已知坑);确认远端树含frame-AB;然后按.scratch/s3/spec.md与票s3:01执行(联合池拼接→五模态训练→五重评测)。

## 已完成与证据(本轮session)

| 项 | 证据 |
|---|---|
| S1全门通过 | reports/s1-stage-acceptance-001.json绑定13份SHA;dev500 99.6%/ECE .004;ADR0013(30k_vs_10k门0.02→0结构性校准) |
| S2首轮 | reports/s2-dev-eval-{image,screenshot,text}-001.json:77.2%/94.6%/99.6%,cf 0.288/0.280 |
| S2探针 | v3探针(cyclic)reports/s2-probe-v3-*.json:image 78.0%/cf 0.30;screenshot 95.3%/cf 0.287;zs-image 78.0%/cf 0.287 |
| ADR0014 | docs/adr/0014-counterfactual-binding-structural-gap.md:cf 0.80门结构性不可达(三体制+零样本全钉0.28–0.32),门保持FAIL列开放风险 |
| 代码 | scripts/s2_media_pools.py(四cohort池)、trainer mixed/audio/video+多根媒体解析、s1_dev_eval counterfactual+--video-frames、scripts/s2_gate.py、train_s1多ledger+变体id后缀;全部fable子代理实现+主会话集成,115+回归绿 |
| S3前置 | 池冻结audio 326/80/40、video 8299/900/300(远端artifacts/s3-pools-*,双遍一致零交集);候选driver与排除union证据在远端s3-media-candidates-001;票s3:01+spec就绪(Blocked by s2:01) |

v2迭代(当前):joint-v2-001 adapter=14368行池(含1809置换变体)、1600步、loss收敛0.001,训练报告reports/s2-joint-v2-run-001.json(远端);评测即上表命令。

## 已知坑(本轮真实踩过,勿重复)

- **heredoc经ssh写远端脚本**:未转义的$变量被远端空展开(--data /dev.jsonl、--media-root /media事故×3)。脚本一律本地生成字面路径→scp;或全用\$转义。
- **pkill/pgrep -f 自匹配**:模式会匹配自己会话的远端shell(自杀×2、死锁等待×1)。按`ps`查PID再kill。
- **链运行中禁止同步代码**:checkout会换掉后续阶段执行的代码,证据混revision;同步排在链完成后。
- **`git stash -u`循环会卷走未跟踪reports**:远端stashes已含历史;恢复用`git show stash@{n}:path`;已有恢复目录/root/autodl-tmp/s2-evidence-recovery。缺失文件(如实缺口):s2-probe-run-001、s2-probe-eval-screenshot-001、s2-probe-zeroshot-screenshot-001,其数值已录ADR0014与本文件。
- 远端HF下载需`source /etc/network_turbo`;教师env(s2训练/评测用)=/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python(torch2.13/TF5.17/peft0.21.2+swanlab0.10.2增量);SwanLab key在/root/.s1-swanlab-env(600)。
- S2评测媒体根:image dev用media-dev-002/image-a/**media**(uri含image/前缀);screenshot dev用.../app_split_train_under19_5gb/**screenshots**;训练池媒体根不同(coco树/screenshots树),config media_roots已绑。

## 边界(不变)

不push/付费/自动开关卡/生产部署;质量门不暗降,结构性不可达按ADR0010/0013/0014格式记录;5b/探索不开;需要用户时写STATUS"待我决策"。完整原始目标见plan/handoff-2026-10-15-s0-complete.md。
