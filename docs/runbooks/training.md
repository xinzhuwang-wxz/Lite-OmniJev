# P0 十步训练入口（GPU 路径尚未实测）

此入口执行真实 Gemma 4 E4B 的 **text-only 单序列 LoRA 工程冒烟**。CPU 梯度测试、100 条程序真值、预检成功均不等于 P0 或模型质量门槛通过。多模态、HF/vLLM 对齐与完整断点续训仍是 P0 后续工作。

## 环境与 API 来源

依赖由根 `pyproject.toml` 的 training extra 和 `uv.lock` 固定；在 CUDA 主机执行 `uv sync --frozen --extra training --group dev`，并验证 PyTorch CUDA 可用。Apple Silicon 的 CPU wheel 仅供数学/冻结行为测试。

2026-10-13 核对：[Transformers Gemma4 文档](https://huggingface.co/docs/transformers/model_doc/gemma4) 和[官方实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)提供 `Gemma4ForConditionalGeneration`、`logits_to_keep=1` 与 `model.language_model.layers`。显式采用模型 config 声明的类，避免 AutoModel 别名变化；不加载 remote code。PEFT [LoRA 配置](https://huggingface.co/docs/peft/package_reference/lora)和 [save_pretrained](https://huggingface.co/docs/peft/package_reference/peft_model)提供适配器注入与 safetensors 保存。

## 无卡预检

先用资产工具下载并校验完整的 pinned E4B。metadata-only 目录可以检查题面/token，但预检会因权重/CUDA 不足以非零退出；这是预期阻塞，不是训练通过。

```bash
uv run python -m scripts.train_smoke preflight \
  --model-dir artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --asset-manifest artifacts/manifests/e4b.json \
  --report reports/train-preflight.json
```

输出 JSON 含 `blockers`、退出码、配置、数据哈希、代码 revision、时长。全部权重按 manifest 哈希重验，模型目录必须以 revision 命名；只读取本地文件。预检不要求 import torch 成功才能输出结构化失败。

## 用户开卡后运行

```bash
uv run python -m scripts.train_smoke run \
  --model-dir artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --asset-manifest artifacts/manifests/e4b.json \
  --output-dir artifacts/training/p0-smoke-001 \
  --report reports/train-smoke-001.json
```

默认生成固定 seed 的 100 条程序真值样本，每条一个问题，轮换 Choice/Noul/Score/U。`--data FILE.jsonl` 可替换为严格校验的 100 条 text/train 样本；每条必须一个问题。输入超过 max_length 即失败，禁止截断监督条件。

训练共 10 个优化步，每步累积 10 个单序列；使用共享官方模板、关闭 thinking，逐题验证答案在实际 generation boundary 为单 token。读取最后位置的所有合法答案 logits（包括 U），先 FP32 log-softmax 再对 hard/soft/U 目标求交叉熵；没有温度、采样或 top-k。仅 decoder q/k/v/o 与 gate/up/down 投影添加 r16 LoRA，主干/编码器/词表头冻结，不新增分类头。

已有 output 目录直接拒绝，防止覆盖实验。成功生成 `adapter/adapter_model.safetensors`、`data.jsonl`、`run.json`。报告含版本、资产/数据/适配器哈希、seed、逐步 loss、步数、时长和显存。`cost: null` 表示未接入租赁账单，不能当作零成本。异常保留已完成步数与失败原因。

每个完成的优化步会原子写入 `checkpoints/step-000N/`。checkpoint 由 JSON provenance、LoRA 可训练张量 safetensors、optimizer tensor safetensors 和 RNG safetensors 组成；恢复不读取 pickle。恢复必须显式指定已有 run 目录里的 checkpoint，普通 `run` 仍拒绝已有 `--output-dir`：

```bash
uv run python -m scripts.train_smoke run \
  --model-dir artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --asset-manifest artifacts/manifests/e4b.json \
  --output-dir artifacts/training/p0-smoke-001 \
  --report reports/train-smoke-001-resume.json \
  --resume-from artifacts/training/p0-smoke-001/checkpoints/step-0004
```

恢复前会重新计算并比对 config、模型 revision、资产哈希、数据哈希和样本顺序；任一不一致都会拒绝继续。恢复只覆盖这个固定 10 步 smoke 的 optimizer、RNG、step、loss 和 LoRA 参数状态，不是通用训练框架。

## 验收与剩余缺口

- `uv run pytest tests/test_training.py`：无卡契约和失败路径；未安装 torch/peft 时相关梯度测试会明确 skip。
- `uv run --extra training pytest tests/test_training.py`：CPU 梯度及 LoRA 冻结测试；不下载真实模型、不证明 GPU 可运行。
- CUDA 主机上必须实际跑满 10 步、有限 loss、非零有限 LoRA 梯度并保存适配器后，才可登记“text smoke 已通过”。
- resume 的 CPU 等价性测试只覆盖小模型状态恢复；真实 E4B/GPU 仍须在 CUDA 主机实跑中验证。
- 单卡内存、真实 E4B BF16 加载、GPU 算子确定性、实测学习曲线、媒体路径和推理概率对齐均未在本机验证；失败时留报告并修复，不降低门槛。
