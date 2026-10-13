# AutoDL 无卡实例接续

当前实例已实测：Ubuntu22.04 x86_64、0.5CPU/2GiB、无GPU；数据盘 `/root/autodl-tmp` 800GiB。1050GiB属于早期实例历史证据，不能用于当前空间预算。
SSH入口由用户提供，连接信息见 `plan/STATUS.md`；凭据禁止写到仓库。用户负责开/关卡。
远程项目 `/root/autodl-tmp/lite-omnijev`，训练 `.venv`，独立服务 `/root/autodl-tmp/envs/serve`。

## 安装与缓存

从项目根 `source infra/bootstrap/autodl-env.sh`。所有模型、Python、uv/HF缓存、环境、实验产物放数据盘。
根 `uv.lock` 固定训练及monitoring extras；服务使用 `infra/serve/uv.lock`。不得将服务依赖安装进训练环境。
国内 simple 索引可能缺新版本，但原 wheel 路径仍可用。本次先 `uv export --frozen --no-emit-project`，再用 `infra/bootstrap/locked_wheel_urls.py` 选 Linux x86_64/Python3.11 wheel，保留上游SHA256，镜像仅替换host，不改版本。
`uv pip sync --no-index --require-hashes` 安装直链锁；监控的 `pwinput==1.0.3` 只有sdist，原sha锁定后用已安装setuptools构建，不能伪装全部是预编译wheel。
实际安装与导入证据：`reports/linux-training-environment.json`、`reports/linux-service-environment.json`、`reports/linux-tests.txt`；文件缺失/blocked就未通过。
准备阶段直接使用环境内的Python执行模块，避免裸 `uv run` 同步掉可选training/monitoring依赖。

## 模型与数据

E4B固定 `ee0ef6023621cff504d758262d4e04895a5af4a2`，目录 `artifacts/models/google__gemma-4-E4B-it/<revision>`。
下载使用官方manifest的尺寸/SHA，镜像只传输文件；断点缓存保留。当前作业在 `reports/e4b-transfer.log`、`reports/e4b-download-progress.json` 记录状态。
Hugging Face官方直连超时，可使用[AutoDL学术加速](https://www.autodl.com/docs/network_turbo/)或公开镜像；不关闭TLS验证、不修改官方哈希。
模型已下载不等于可开卡：先全文件验证、token/template与数据编译，检查训练和服务环境。
P0数据 `artifacts/data/p0-train-100.jsonl` 为100条程序真值，非质量benchmark；来源/哈希见 `reports/p0-data-manifest.json`。
CLINC公开train仅下载到 `artifacts/data/raw/`，保留CC-BY3许可与归属；151类题型映射尚未准入，不混入P0。来源审计见 `configs/data-admission.json`，不下载validation/test/holdout。

## SwanLab

凭据在 `/root/.config/lite-omnijev/swanlab.key`，目录0700、文件0600；只在所需进程读取，不输出。
真实训练前在该进程环境设置 `SWANLAB_API_KEY`、`LITE_OMNIJEV_SWANLAB_MODE=online`，项目固定Lite-OmniJev、workspace固定maxen。
`reports/swanlab-auth.json` 是真实鉴权；`reports/swanlab-preparation.json` 是仅准备阶段的云端记录，training_steps=0，不代表模型训练。
训练loss/LR/grad_norm/step自动上报，完整本地JSON和checkpoint仍是验收依据；详见monitoring.md。

## 开卡门槛

当前全项目开卡门槛以 `.scratch/offline-prep/issues/04-integration.md` 为准；`reports/no-gpu-readiness.json` 仅为P0历史准备证据，不可凭其旧ready结论开卡。全部必需外部资产、完整离线fresh安装、媒体CPU验收及可在无卡执行的管线准备通过后才通知用户；缺权重、损坏文件、环境导入失败或数据校验失败都保持blocked。
无卡阶段不做权重前向或GPU训练；硬件/内核、BF16、实际峰值显存、基座与LoRA重载、HF/vLLM概率对齐及真实API仍在P0 GPU票验收。
先预留约1小时进行最小P0试跑（估算，吞吐未测），出现阻塞及时通知；不能将设计中的H100时直接折算H20实际时间。

## GPU命令预备
`bash infra/bootstrap/run-p0.sh` 是保留的P0复现入口，先验证CUDA/BF16再训练，单次初试硬超时1小时。P0工程冒烟已有真实通过证据，下一次开卡按S0当前作业清单执行，不默认重复训练。历史价格不作为本次预算承诺，实际以用户账单为准。严格概率、教师质量与阶段门槛仍须独立验收。
外部Parquet结构检查用`infra/data`独立PyArrow锁，远端环境`/root/autodl-tmp/envs/data`；不修改训练/服务包。
