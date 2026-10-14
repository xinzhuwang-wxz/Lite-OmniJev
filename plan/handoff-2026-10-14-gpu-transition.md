# 无卡准备完成 → 有卡模式接续

当前无卡环境能自主完成的必需准备已完成，可以由用户切换有卡模式。代理未开卡，未获代开卡授权；这不代表P0真实模型/S0或终局通过。用户明确要求：先完成当前无卡可做的所有准备，再交接；资源受限检查在切换后首先补验，不无限等待更大无卡实例。

## 新 session 的第一步

在 `/Users/bamboo/Githubs/Lite-OmniJev` 先读AGENTS、`plan/STATUS.md`、`docs/02-project.md`、`plan/LOOP.md`、本文件和 `.scratch/s0/issues/01-teacher-admission.md` / `03-multimodal-parity.md`。核对原生get_goal；旧完整目标实际为blocked，不能把本阶段完成标成终局complete。新session若没有原goal，按原目标创建，不声称跨会话迁移。模式切换准备验收唯一新汇总：`reports/project-before-gpu-readiness-002.json`；001是历史失败判定，保留不覆盖。

SSH：`ssh -p 48926 root@region-42.seetacloud.com`；复用socket `/tmp/omnijev-ssh-root-final` 仅当前会话便利，不保证新session有效。凭据不在仓库，不索要/输出原密码；若登录失效，记录账号阻塞，继续本地独立工作。

远端唯一最新执行源码目录：`/root/autodl-tmp/before-gpu-release-32fb559`。目录名不是版本；准确冻结HEAD及Git bundle SHA以本地 `reports/before-gpu-transition-sync-001.json` / 远端 `/root/autodl-tmp/before-gpu-transition-sync-001.json` 为准。该receipt是冻结提交之后的元数据侧车，不在冻结HEAD内；先读侧车再核对 `git rev-parse HEAD`、`git status --porcelain`。实测代码为655a838，之后仅文档/调度变化，560pass/1skip对应代码没有改变。不要覆盖用户原树 `/root/autodl-tmp/lite-omnijev`；旧4366197源码目录仅保留早期缓存。

## 可以直接复用

| 项目 | 位置 / 证据 |
|---|---|
| 全部7模型70文件121,094,724,397B | 原树 `artifacts/models/`；`reports/project-model-offline-acceptance-001.json` |
| 29来源下载/排除/研究隔离总账 | `reports/project-data-offline-acceptance-001.json`；原料不等于训练准入 |
| wheelhouse、前端/浏览器、量化/端侧工具 | 原树 `artifacts/runtime-builds/`；runtime-offline-acceptance001及子回执 |
| teacher/job完整训练环境 | `/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python`；fresh no-index重建224.9秒、pip check0 |
| 主服务026 | `/root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python`；Torch2.11/TF5.13，保持primary |
| 根开发/脚本 | `/root/autodl-tmp/envs/root-main-offline-check/bin/python` |
| 031 | `serve-next-vllm031-offline-check`，诊断备用，未经用户决定不提升primary |
| 固定公开100题 | 源码 `tests/fixtures/s0-prepare-public-001/manifest.json`；媒体在原树 `artifacts/s0-prepare-public-001` |
| E4B | 原树 `artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Jev-Omni教师 | 原树 `artifacts/models/akhilaaa3__Jev-Omni/5addda86ddee081a68fb067477ea100c221b8917`；安全head/source路径见teacher-inference runbook |

最后只读核查：上述7个关键路径全部存在，远端源码干净，无项目作业；cgroup2GiB，GPU查询No devices found，剩余420,558,426,112B。保留150GB checkpoint与18,164,208,659B模型临时空间，后续阶段按实测重估，不能当成无限预算。

不重下权重/媒体，不重建已验环境，不重复78分钟GUI全量解码、整套560测试、fresh依赖安装或已失败的Docker拉取。运行使用上表完整解释器，避免不带extra的uv run删除可选依赖；不借另一个环境site-packages。每次报告/输出用新路径，已有路径拒绝覆盖。

## 用户切换之后的首批前置

1. 先只读 `nvidia-smi -L`、GPU型号/显存/驱动、cgroup memory.max/cpu.max、持久盘空间，记录模式是否真的改变。未切换则交接保持待接续，不代开卡或重复2GiB失败配置。GPU可见但RAM仍不足也照实报告，不能假定开卡解决内存。
2. 先补以下资源受限CPU检查，仍设置CUDA_VISIBLE_DEVICES为空、离线、单线程和重锁，避免意外模型加载。固定运行时/同一Linux主机完成两遍HF100，比较每case摘要，冻结新的同环境参考。原Linux首遍100题exit0/94.092秒，第二遍90题后137；旧失败不覆盖。Mac/Linux有60条编码/张量摘要差异，不能借Mac参考冒充Linux一致。
3. primary026正常native processor六case成功后才full100；必须精确ID/shape/dtype/hash/帧数时间戳比较。正常ModelConfig初始化先跑；`--same-process-inspection`仅诊断，不能用它替代正常通过。031仅诊断，不要默认切换引擎。无卡两次137的killer未证实，不写成已证OOM/不支持CPU。
4. primary026真实官方processor+实际网关：先视频8/16，再六类及固定100请求。此处mock backend只验证输入传输，随后仍需真实模型浏览器/概率验收。视频batch历史137保持可见。
5. 任一前置失败：保存实际退出码/时间/配置hash/checkpoint及error，不把中断pending报告当成功；先诊断，不启动教师全量、长训练或反复重试。如果GPU闲置且短期无法恢复，落盘并通知用户关卡。

以下命令是切换后待执行接口，未在新资源下通过。运行前设唯一run名，不复制覆盖旧结果：

```bash
cd /root/autodl-tmp/before-gpu-release-32fb559
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PREP_ASSETS=/root/autodl-tmp/lite-omnijev/artifacts
PREP_MODEL="$PREP_ASSETS/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
PREP_PY=/root/autodl-tmp/envs/serve-vllm026-offline-check/bin/python
PREP_RUN=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "artifacts/gpu-transition-$PREP_RUN"
PREP_OUT="artifacts/gpu-transition-$PREP_RUN"
CUDA_VISIBLE_DEVICES='' flock /tmp/omnijev-cpu-heavy.lock "$PREP_PY" \
  -m scripts.s0_prepare \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --asset-root "$PREP_ASSETS/s0-prepare-public-001" --model-dir "$PREP_MODEL" \
  --profile canonical_http --output "$PREP_OUT/hf-a.json"
```

首遍成功后第二遍仅将output改为hf-b.json；两份complete_100_cases及进程exit0、cases摘要一致后，才能用hf-a作native参考：

```bash
CUDA_VISIBLE_DEVICES='' flock /tmp/omnijev-cpu-heavy.lock "$PREP_PY" \
  -m scripts.s0_vllm_processor \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --asset-root "$PREP_ASSETS/s0-prepare-public-001" --model-dir "$PREP_MODEL" \
  --hf-reference "$PREP_OUT/hf-a.json" --output "$PREP_OUT/native-six.json"
```

六case成功后同命令增加`--full`且output改为native-full100.json。正常/诊断结果分别命名；详细检查定义见 `docs/runbooks/s0-vllm-processor.md`。

网关pytest使用早先仅纯pytest工具target（已核对存在），不混入跨环境模型库：

```bash
export S0_LOCAL_MODEL_DIR="$PREP_MODEL"
export S0_CANONICAL_MEDIA_ROOT="$PREP_ASSETS/s0-prepare-public-001"
PREP_TEST_TOOLS=/root/autodl-tmp/before-gpu-release-4366197/artifacts/primary026-linux-gateway-cpu-001/test-tools
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PREP_TEST_TOOLS:$PWD" \
  flock /tmp/omnijev-cpu-heavy.lock "$PREP_PY" -m pytest \
  tests/test_gateway_media.py::test_real_five_categories_and_multiquestion_canonical_chat \
  tests/test_gateway_media.py::test_fixed_100_gateway_requests_match_canonical_reference \
  -q --tb=short </dev/null
```

已在远端实际collect-only确认六个node（0.48秒）；视频小批选完整node `tests/test_gateway_media.py::test_real_five_categories_and_multiquestion_canonical_chat[80-video_url]` 与 `[90-video_url]`，shell中用引号括住各node。保留log和进程退出码，全部选中项不得skip；上述完整组在视频小批成功后执行。命令3个bash块已bash -n通过，两CLI --help在固定026环境exit0；这些仅入口验证，不是资源补验通过。

## 真实GPU接续与停止点

- 用既有训练/教师与026服务环境各自检查CUDA/驱动小smoke，再最小真实模型forward。vLLM环境与HF环境不能只凭import成功声称同权重/模板/dtype输入一致。
- `docs/runbooks/teacher-inference.md` / `s0-teacher-eval.md`：teacher_forward是CUDA-gated readiness，不能当成真实forward结果；实际benchmark runner先小样本，再固定MMAU1000与九任务1800两遍，按原runner断点规则续跑。安全head的原格式受控参考与safetensors真实输出等价仍待验；不加载unsafe pickle或删head。
- `docs/runbooks/training.md`：核对P0现有实际证据，补真实10步/断点恢复缺口；仅LoRA，冻结主干，无分类头。不要重复已经实测的P0工程任务。
- `docs/runbooks/s0-parity.md`、`s0-gateway-parity.md`、`service-environment.md`：100题HF→原生chat→网关，完整合法答案+U/raw概率、精确expanded IDs、不截断/不二次采样/thinking关闭。历史maxTV .01129/.02292失败，严格<.001不变，D004未批准，不自动应用。
- `docs/runbooks/s0-prefix-cache.md`：独立专用on/off服务，真实KV/顺序10题；全部paired T10/T1≤2。mock结果不作性能证据。真实浏览器复用browser004流程与公开fixture，改用真实模型并存trace/截图；现有browser004明确fake。
- S0所有门槛通过再S1。独立gate身份、6000题/12000真人判断及κ≥.7仍须外部安排；开发agent只能读汇总。许可/来源未准入数据不因下载完成混入训练，见29源总账。S6真机/后续运营观测继续待条件，不因模式切换忽略。
- 每次记录代码/模型/配置/数据hash、seed、硬件/软件、耗时、退出码、实际成本（未知填null）、checkpoint和下一命令。用户未授权全项目300H100时或S4放量预算；先小smoke，不盲目启动长跑。
- 断线先查原作业/报告，禁止启动第二份；output已有则按runner显式resume，不覆盖。作业停止后checkpoint、adapter和报告全部落持久盘，记录续跑点并通知用户关卡；不代关卡/删盘。

## Suggested skills

具体完成主张用verify；交接用handoff并遵用户目录保存在plan。禁止重型OMX/superpowers流程、重复访谈、heartbeat、自动付费/开关卡/生产部署/push。新问题按最窄接口诊断，所有阶段门槛依原HTML、ADR与configs/gates.json。

## 新 session 启动词

> 继续Lite-OmniJev原终局目标。先读AGENTS、plan/STATUS.md、plan/handoff-2026-10-14-gpu-transition.md及readiness002。当前无卡可做准备已收口，先只读确认用户是否已切有卡、核对最新sync receipt和远端HEAD；未切换不代开卡。已切换则按交接先补同Linux HF重跑/native正常初始化/网关视频资源前置，再最小GPU smoke和S0实测。不重下缓存、不重建已验环境、不重复已完成CPU全量；许可与人工gate缺口显式保留。按原阈值推进，S0全绿前不进S1，禁止自动开关卡/付费/生产/push，及时写票据、STATUS和真实报告。
