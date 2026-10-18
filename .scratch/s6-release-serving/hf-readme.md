---
license: cc-by-nc-sa-4.0
base_model: google/gemma-4-E4B-it
base_model_relation: merge
library_name: transformers
tags:
- decision-making
- multimodal
- probability-readout
- gemma
language:
- en
- zh
---

# Lite-OmniJev · 单步多模态决策

冻结 Gemma 4 E4B 基座 + 决策 LoRA，每题一次前向读取完整答案集合的概率。
支持 Choice（2–20项）、Score（2–10级）和 yes/no；覆盖文本、图像、截图、音频与视频。

<img src="docs/charts/readout-diagram.svg" width="720" alt="每题一次前向读取选项概率的示意图">

**当前状态：发布候选。** vLLM FP32 配方已通过数值验证，返回原始概率
`calibration: unfitted`。本仓不是已经完成全部端侧、生产及运营验收的产品。
[工程仓与使用说明](https://github.com/xinzhuwang-wxz/Lite-OmniJev) ·
[训练过程](TRAINING.md) · [数据声明](DATA_CARD.md) · [验收边界](RELEASE_STATUS.md)

## 已测表现

| 榜单 | 本模型/评测方案 | 基座 | 对照公开数 |
|---|---|---|---|
| JevBench 231 | 76.62% | 76.19% | Jev-Omni-12B 86.15% |
| MMAU test-mini 1000 | adapter全挂载60.6%；音频路由基座65.9% | 65.9% | Jev-Omni-12B 63.10% |
| MVBench 九任务1800（参照） | 48.5% | 47.2% | 教师53.09%（14任务/2786，口径不同） |

JevBench仍低于教师公开结果。音频路由为评测方案；当前启动脚本全模态挂载adapter，不自动路由。
Decision Index上游套件/来源不可得，缺测如实保留；不宣称SOTA。
最终D3P600校准在封存汇总上ECE门3/6通过，audio/video/crossmodal未达；不能跨运行时激活温度。

## 快速开始：网关与Playground

权重通过 `hf download Chinzhu/Lite-OmniJev --local-dir lite-omnijev-model` 下载。
工程代码从上方GitHub获取。服务环境按工程仓 `docs/runbooks/service-environment.md` 的锁定配方安装，
网关环境按工程仓 `uv.lock` 安装。以下变量都是绝对路径，不是账号凭据：

```bash
# 终端A，先进入一个新的运行目录；H20 96GB已测，FP32验收配方显存约39GB
SERVE_ENV=/absolute/path/to/serve-env \
MODEL_DIR=/absolute/path/to/lite-omnijev-model \
BASE_DIR=/absolute/path/to/lite-omnijev-model \
bash /absolute/path/to/lite-omnijev-model/gateway/launch-vllm.sh

# 终端B
GATEWAY_PYTHON=/absolute/path/to/gateway-env/bin/python \
REPO=/absolute/path/to/Lite-OmniJev \
LITE_MODEL_DIR=/absolute/path/to/lite-omnijev-model \
LITE_MEDIA_ROOT=/absolute/path/to/trusted-media \
bash /absolute/path/to/lite-omnijev-model/gateway/launch-gateway.sh
```

打开 `http://127.0.0.1:8000/` 使用Playground，或直接请求：

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"订单已到达末端站点，尚无签收记录。","questions":{"status":{"type":"choice","instructions":"最可能的状态？","criteria":{"A":"已签收","B":"待配送"}},"progress":{"type":"score","instructions":"配送进度","criteria":["未发货","运输中","已签收"]},"signed":{"type":"noul","instructions":"订单是否已经签收？"}}}'
```

返回 `answers`、版本与计时。每题包含 `probabilities`、`confidence` 和相应的 `choice`、`score`
或 `noul`；Score为从0开始的期望等级，Noul为yes的概率。U仅保留在内部数值诊断，不作为产品输出。
媒体使用 `media: [{"type":"image","uri":"shot.png","sha256":"<文件SHA256>"}]`；
也支持audio/video，视频显式选择 `video_frames: 8` 或 `16`。URI必须位于服务端受信媒体根。
当前是本地开发网关，不提供公网鉴权或上传服务。

## 其他工件

| 工件 | 实测范围 |
|---|---|
| FP32平面分片（5片，约31.98GB） | 2130张量；与compact源逐字节核验；HF/vLLM原始概率最大偏差5.75e-6，小于1e-3 |
| `adapter-d3p600/` | rank16/alpha32，adapter SHA256前缀767be09b；加载时显式指定官方冻结基座 |
| GGUF F16 / Q8_0 / Q4_K_M | 文件15.05 / 8.03 / 5.34GB；20个文本案例各20/20选答一致；不是概率/ECE等价证明 |
| GGUF mmproj | 图像/音频/视频生成式冒烟；截图需RGBA→RGB，长边≤1344；不是完整决策基准 |
| `mlx-4bit/` | 转换完成约6.2GB；Apple Silicon运行仍待验证 |
| `gateway/receipts/` | 数值、浏览器、量化与分片恢复回执；`SHA256SUMS`核验分发文件 |

llama.cpp为独立生成式运行路径，不等价于网关的原始答案概率接口。
按 `gguf/` 中模型与mmproj路径使用上游CLI；音视频模板、截图预处理和后台stdin要求见工程仓
`docs/runbooks/gguf-multimodal.md`。不提供未经实跑的MLX推理示例。
PEFT可用官方 `PeftModel.from_pretrained(base, ..., subfolder="adapter-d3p600")` 加载adapter；
标准Transformers/PEFT模型没有 `model.decide()` 方法，结构化判断使用本项目网关。

## 数值与校准边界

数值参考：float32、TF32关闭、IEEE、eager、前缀缓存关闭、单序列。
HF/vLLM对照为100个公开案例各两遍；不是200个独立案例，也不是三运行时200题验收。
现有温度绑定Transformers BF16/native-LoRA，与本次vLLM FP32/merged配方不同；
网关默认返回`unfitted`，要求强制校准时明确拒绝，不挪用温度。
生成token计入`backend_usage`，产品不返回生成文本；多题分别前向，不把并发请求称为一次前向。

## 许可与限制

项目新增代码与权重按CC-BY-NC-SA-4.0（非商业）发布；基座另受Google Gemma原始条款约束，
第三方依赖与数据保留各自许可。训练/评测来源与角色见DATA_CARD；封存/校准原始样本不分发。
E2B蒸馏端侧、完整量化概率与ECE、真机内存/时延、生产SLO尚未验收，不以冒烟代替。
