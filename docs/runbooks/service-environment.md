# P0 独立 vLLM 服务环境

核查日期：2026-10-13。目标 Ubuntu 22.04 / x86_64 / Python 3.11，后续 H20 96GB、驱动 580.82.07。
本目录是独立 uv 项目，不安装到训练环境，不使用镜像自带 Python 3.12/PyTorch。
无卡机 0.5 CPU / 2GiB 仅串行安装、资产核验和轻量导入；不能启动 E4B 推理。

## 固定选择与官方证据

- `vllm==0.26.0`（Apache-2.0）：官方 [Gemma 4 配方](https://recipes.vllm.ai/Google/gemma-4-E4B-it) 给出 E4B 基座支持；但配方最低 0.19.1 不足以满足本项目接口。已直接核对 [0.26.0 CompletionRequest](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/completion/protocol.py) 的 `logprob_token_ids`、[Gemma4 多模态包装类](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/gemma4_mm.py) 的 `SupportsLoRA`，以及 [raw_logprobs 配置](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/model.py)。检查的 0.25.0 CompletionRequest 尚无指定答案 token 接口，不能直接复用。
- `torch==2.11.0`、`torchaudio==2.11.0`、`torchvision==0.26.0`：沿用该版 [CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/cuda.txt)，BSD-style 上游许可随 wheel 保留。Transformers服务环境固定 `5.13.0`（Apache-2.0），训练仍为5.19.0；共享固定的模型/tokenizer/template文件，实际输入须探针对齐。
- CUDA 采用该发行版默认构建；[tagged Dockerfile](https://github.com/vllm-project/vllm/blob/v0.26.0/docker/Dockerfile) 默认 `13.0.2`，CUDA 依赖包含 `cu13`。该版安装网页仍有默认 12.9 的旧文字，不能据此混装 cu129 Torch。以锁定 wheel、安装后 `torch.version.cuda` 和 `vllm.collect_env` 实测为准。
- [NVIDIA 兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) 给出 CUDA 13.x 最低驱动分支 580；580.82.07 满足分支条件。这仅是兼容性候选依据，PTX/JIT、vLLM ABI、H20 内核启动仍须开卡验证。
- 只引入 P0 文本服务及 vLLM 必需依赖；未安装 `vllm[audio]`，音频扩展/多模态验收留到对应票。传递依赖的确切版本、下载地址和哈希见 `infra/serve/uv.lock`。NVIDIA 组件适用 NVIDIA 自身许可，不能把整个依赖栈统称 Apache-2.0。

## 无卡准备（目标机待执行）

以下示例以实际挂载确认后的 `/root/autodl-tmp` 为数据盘。先核实 `findmnt -T /root/autodl-tmp`、`df -h`；不要把 30GB 系统盘当权重或 wheel 缓存盘。仓库也放数据盘，按实际 checkout 调整 `REPO`。

```bash
export DATA_ROOT=/root/autodl-tmp/lite-omnijev
export REPO="$DATA_ROOT/repo"
export UV_CACHE_DIR="$DATA_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$DATA_ROOT/python"
export UV_PROJECT_ENVIRONMENT="$DATA_ROOT/envs/serve"
export HF_HOME="$DATA_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$DATA_ROOT/cache"
export TMPDIR="$DATA_ROOT/tmp"
export VLLM_CACHE_ROOT="$DATA_ROOT/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$DATA_ROOT/cache/triton"
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR" "$HF_HOME" "$DATA_ROOT/reports"
cd "$REPO"
uv python install 3.11.14
uv sync --project infra/serve --frozen --no-dev --no-build
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
```

`--no-build` 阻止在小内存实例意外编译；安装失败保留日志与退出码，不移除约束硬装。
系统 `libnuma.so.1`/`libgomp.so.1` 等依赖若导入报缺失，先记录镜像和具体错误，再安装对应 Ubuntu 包；不要改写 NVIDIA 驱动。
导入检查串行执行，记录 peak RSS，OOM 属资源阻塞，不能记为通过：

```bash
/usr/bin/time -v "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import importlib.metadata as md
import platform
import torch
import transformers
import vllm
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
print(platform.platform(), platform.machine(), platform.python_version())
print({p: md.version(p) for p in ('vllm', 'torch', 'transformers', 'torchaudio', 'torchvision')})
print('compiled_cuda', torch.version.cuda, 'gpu_visible', torch.cuda.is_available())
assert 'logprob_token_ids' in CompletionRequest.model_fields
assert 'return_tokens_as_token_ids' in CompletionRequest.model_fields
PY
```

无卡 `gpu_visible=False` 是预期；导入、依赖检查成功也不是 GPU ready。另存 `uv --version`、`uv pip freeze --python ...`、锁文件 SHA256、代码 commit、模型 manifest/hash、命令、UTC 时间和退出码。初始化/安装日志不要含 HF token。

## 开卡后的检查与启动（GPU 待验收命令）

用户完成开卡后，复用以上数据盘环境变量。先 `nvidia-smi`、`"$UV_PROJECT_ENVIRONMENT/bin/python" -m vllm.collect_env`，再检查 GPU：

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'GPU unavailable'
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
PY
```

`MODEL_DIR` 必须指向资产工具已校验的固定 revision 完整本地 snapshot；`ADAPTER_DIR` 指向训练十步后保存并记录哈希的 PEFT adapter。先以无 LoRA 基座进行 HF/vLLM 对照，再用同一启动配置加 adapter 对照。停止训练进程、释放 GPU 后启动，二者不要争抢单卡。

```bash
export MODEL_DIR="$DATA_ROOT/models/gemma-4-E4B-it"
export ADAPTER_DIR="$DATA_ROOT/runs/p0/adapter"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"  # FlashInfer JIT needs the environment ninja executable
"$UV_PROJECT_ENVIRONMENT/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 2048 --max-num-seqs 1 \
  --gpu-memory-utilization 0.75 --seed 42 \
  --enforce-eager --generation-config vllm \
  --logprobs-mode raw_logprobs --max-logprobs 32 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules "decision=$ADAPTER_DIR"
```

基座首次启动删去最后两行 LoRA 参数；之后确认 adapter 实际 `r<=16`（若更大按 adapter 调整允许 rank），目标模块仅来自语言决策 LoRA。`2048`/单并发/eager 是 P0 小样本可复现配置，未声称满足生产延迟；所有 prompt 必须计数并留出一个生成 token。
不要换用工具调用模板或开启 reasoning parser。网关 `train.template.render_prompt` 以固定 snapshot 的官方 chat template、`enable_thinking=False` 生成 `/v1/completions` prompt；若独立调试 chat 请求，显式 `chat_template_kwargs={"enable_thinking":false}`。

`curl -f http://127.0.0.1:8001/health` 和 `/v1/models` 仅验证存活/adapter 注册。实际请求必须包含所有答案及 U 的 `logprob_token_ids`、`return_tokens_as_token_ids=true`、`temperature=1`、`max_tokens=1`；`logprobs=1` 是现有客户端协议，不能把默认 top-1 结果当答案全集。缺 token、非有限值、正 logprob、≤-9999 哨兵都拒绝。

完成真实 100 题 HF/vLLM 最大概率差 `<1e-3`（基座与 adapter 分别检查）、相同模板/token IDs/权重 dtype 和原始概率模式核验后，才设置网关：

```bash
export LITE_MODEL_DIR="$MODEL_DIR"
export LITE_UPSTREAM_URL=http://127.0.0.1:8001
export LITE_UPSTREAM_MODEL=decision
export LITE_MODEL_VERSION=p0-REPLACE_WITH_VERIFIED_ADAPTER_HASH
export LITE_RAW_LOGPROBS_CONFIRMED=1
```

网关使用根项目环境单独启动；不要把此服务项目的 `UV_PROJECT_ENVIRONMENT` 继续用于根项目 `uv run`。本票不提供或宣称已有真实 parity 报告；若模型映射/adapter 加载/指定 token 接口失败，保持 blocked，保存失败证据，不合并权重或改推理引擎绕过验收。

## 验证状态

本地 macOS 上 uv 0.9.27 已完成 197 包锁解析、离线 `uv lock --check`，并将锁导出后按 Python 3.11 / manylinux_2_35 x86_64 / only-binary 重新核验（196 个第三方包，exit 0）；未下载完整 wheel 或安装模型。Linux 安装、导入、CUDA BF16、模型加载、LoRA 加载、100 题概率一致性、API 集成全部待目标机实测。锁解析成功不能替代上述任一项。

## 2026-10-13 GPU启动兼容性修正
实际启动vLLM0.26.0+Transformers5.19.0，在Gemma4 config.head_dim读取时抛AmbiguousGlobalPerLayerAttributeError；无卡时只import不能发现此错误。最小复现为AutoConfig.from_pretrained(...).text_config.head_dim。
按[vLLM0.26.0官方release](https://github.com/vllm-project/vllm/releases/tag/v0.26.0)所用Transformers5.13.0固定服务栈，同时锁定其兼容tokenizers0.22.2/huggingface-hub1.33.0；训练栈、模型文件不变。上游同类问题见[issue52768](https://github.com/vllm-project/vllm/issues/52768)。
`infra/serve/check_model_config.py MODEL_DIR`现于目标环境通过（head_dim=256）；uv pip check通过。必须继续真实服务启动和100题概率对照，不能将配置读取通过当作模型正确性通过。
P0文本启动增加三模态limit=0和skip-mm-profiling，避免为不输入的媒体分配显存；多模态阶段必须恢复相应预算并独立验证。

第二个GPU启动问题为FlashInfer JIT找不到ninja：ninja已安装在服务venv/bin，但绝对路径执行vllm不会自动激活该PATH。将服务bin加入PATH后继续验证；无需修改引擎或重新下载模型。
