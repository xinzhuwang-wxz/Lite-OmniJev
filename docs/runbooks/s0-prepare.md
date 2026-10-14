# S0 公开输入 CPU 准备

`train.multimodal.prepare_case` 是共享入口。它复用 `render_payload` 和
`check_answer_tokens`，用官方本地 `AutoProcessor` 关闭 thinking、不截断输入，构建
CPU 张量。`PreparedDecision` 的 `messages` 只含官方模板占位符；`prompt`、`inputs`、
`answer_token_ids` 仅留在内存。仅 `summary` 写报告，不落原文、token ID 或媒体内容。
占位符 messages **不是可直接提交的 vLLM Chat 请求**。vLLM processor/transport、
视频 JPEG 重解码像素及时间戳一致性仍待下游验收；本入口没有 forward、模型质量或概率结论。

公开 fixture：`tests/fixtures/s0-prepare-public-001/manifest.json`，每类20题，涵盖
text/image/screenshot/audio/video、choice/score/noul，所有合法答案集都含 U。
这些是程序自写问题，无真值、无教师标签；100条输入来自4个来源族（CLEVRER 视频与
其首帧同族、GUI train 截图、Clotho CC0 音频、程序文本），每类20个问题变体；20个视频问题分为10×8帧和10×16帧。
不代表100独立样本、准确率或泛化覆盖。来源与派生方法在 `provenance.json`。
原始媒体不进 Git，受控远端根为
`/root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001`。

源 split 与项目 pool 独立：此 fixture 全部项目 dev。未来训练构建器必须消费
`excluded-training-families.json`，排除对应源 family 及媒体 hash；不得因原来源是 train
再次入训练池。开发入口拒绝 holdout 和跨池同族。

```sh
uv run python -m scripts.s0_prepare \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --asset-root /path/to/verified/public/media \
  --model-dir /path/to/pinned/gemma4/local/files \
  --output reports/s0-prepare.json
```

需已有 training CPU 依赖、Pillow/numpy、ffmpeg/ffprobe；不新增依赖，不自动下载。
必须使用固定 revision 对应且已验哈希的 metadata 文件；目录名不构成模型身份。
本次两份 CPU 报告的 `model_files` 已逐文件与固定 revision 的项目下载回执及资产清单
独立交叉核对，证据见 `reports/s0-prepare-model-metadata-crosscheck-001.json`；
这只绑定 processor/tokenizer metadata 来源，不构成权重或整条 API 准入。
加载设置 `local_files_only=True` 和
`trust_remote_code=False`，不加载模型权重。完整运行要求100题、五类各20题、三primitive。
已有 `--output` 路径（含目录、正常或悬空软链接）会在写入前拒绝；每次运行使用新报告路径。
`--allow-partial` 仅作显式调试，成功也写 partial，不写 complete_100_cases。

无卡远端运行整个 Python 进程时使用 `flock /tmp/omnijev-cpu-heavy.lock`、
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false`。
不要与高 RSS 下载/安装并行挤占2GiB内存。报告先写 pending，实例化前与每题前写
checkpoint；外部 SIGKILL 后 checkpoint 保留，但需监督者另记实际退出码，不能当完成报告。
失败记录：`reports/s0-prepare-resource-failure-001.json`。原137失败证据不可覆盖。

输入只接受显式根下的相对常规文件，拒绝远程 URI、逃逸、软链接、空文件、坏 hash；
单文件默认64MiB，图片40MP，音频30秒、视频60秒。视频仅8/16帧，由manifest固定，
本CPU probe按源帧索引均匀取样（含首尾）与最长边640像素解码；
显式传入源fps、total_num_frames、frames_indices、duration，关闭processor二次采样，
不依赖默认fps24；不声称复现任何benchmark视频协议。
每题摘要保存实际配置、来源、媒体/payload/prompt/答案映射 hash，以及每个张量
shape/dtype/hash。generation boundary 检查合法答案单token、非特殊/未知、可逆与
prompt续接；同时校验媒体展开后的最后8个token仍匹配模板生成边界。

两遍运行只比较 `cases` 的摘要和 manifest/model/source hash，忽略运行耗时。
CPU实际成功与两遍逐题一致前保持 pending；即使通过，也不解锁 GPU 概率门槛。

本机共享只读target（唯一安装者为此票所有者）：
`/Users/bamboo/.codex/worktrees/s0-multimodal-cpu/Lite-OmniJev/artifacts/mac-media-libs`。
固定Pillow11.3.0/TorchVision0.29.0 Mac arm64 wheels以`uv pip install --target ... --no-deps`
安装；wheels、PyPI SHA与上游许可metadata见`reports/s0-prepare-mac-media-wheels.json`。
以`PYTHONPATH=<target> /Users/bamboo/Githubs/Lite-OmniJev/.venv/bin/python ...`运行，
target优先，root既有site-packages随后；root Torch2.14.1/numpy2.4.6未改。
这是复用现有环境的macOS CPU验证，不是fresh离线安装、Linux/GPU兼容证明。
真实结果与两遍比较见`reports/s0-prepare-acceptance-001.json`；Linux137独立保留。

`tests/fixtures/s0-prepare-public-001/build_public_fixture.py`仅用于远端重建小媒体与固定
问题manifest，输入是已审计源资产；运行时也必须持有重CPU锁。它不读取题标签或
内部封存集，输出媒体留在artifacts；文本source_sha256绑定此生成器源码。

## Canonical HTTP 准备与 chat 客户端

默认 `--profile hf_only` 保留原 HF-only 输入路径与旧报告；该模式不会返回可发送的
HTTP messages。`--profile canonical_http` 使用独立版本 `canonical-http-v1`，返回
`PreparedDecision.http_messages` 与 `http_media_io_kwargs`，不能与 `messages` 的 HF
占位符混用。共享 API 核心是 `prepare_decision(prefix, question, processor, root, ...)`，
不要求或杜撰训练用 Metadata；`prepare_case` 只为 fixture 追加真实 id/category/provenance。

canonical 的实际输入约定：

- 图像/截图：RGB PNG，compress_level=6、optimize=false；HF 从同一 PNG 字节重解码。
- 音频：ffmpeg 转单声道16kHz float32，再以 `round(clip(x,-1,1)*32767)` 编码 PCM16
  WAV。HF 从同一 WAV 的 PCM16 以 `/32768.0` 还原 float32，不使用编码前 float32。
- 视频：按已冻结源帧索引提帧，每帧 JPEG quality=95、subsampling=0、optimize=false、
  progressive=false；HF 使用同一 JPEG 字节重新解码的 RGB。HTTP 用
  `data:video/jpeg;base64,JPEG0,JPEG1,...`，`media_io_kwargs.video` 带源 fps、indices、
  total_num_frames、duration、do_sample_frames=false；HF 保持对应 metadata，禁止二次采样。
- canonical HF 与 HTTP 都显式 add_special_tokens=false、关闭 thinking、不截断。

摘要分别保留原始媒体 SHA、编码前 RGB/float32 SHA、canonical PNG/WAV/JPEG 字节
与重解码 RGB/PCM/float32 SHA、HTTP messages/视频 IO kwargs hash、实际张量摘要。
原文、base64、像素、token IDs 只在内存；仅 `summary` 可落盘。

| 当前 S0 profile 组合 | CPU 与 mock HTTP | 真实 vLLM/GPU |
| --- | --- | --- |
| text/image/screenshot/一段audio/一个video的8或16帧 | 真实公开媒体重解码与HF张量比对 | pending |
| 多图、图+单音频、单视频+单音频 | Metadata-free核心与真实媒体重解码/HF张量比对 | pending |
| 图+视频、多视频、多音频 | 当前profile保守拒绝，后续兼容性CPU验证pending | pending |

这些限制只描述当前 S0 传输 profile 的已验范围，不改变最终产品范围，也不声称引擎
不能处理混合输入。尤其多视频不可把不同 metadata 塞进同一个全局 video kwargs，
因此先拒绝；图+视频尚需对不同固定 vLLM 版本的处理器配置做独立 CPU 验证。

`serve.vllm_chat.VllmChatClient(base_url, model, raw_logprobs_confirmed=True, transport=...)`
支持 `async with`；`await client.read_logprobs(prepared)` 返回 `ChatLogprobs` 的完整
`raw_logprobs`、内存 prompt IDs、usage 与安全摘要，供共用 readout 使用，客户端不
自行归一化。`raw_logprobs_confirmed` 必须来自调用者核验的固定启动记录；布尔值和
response model alias 都不能自证实际权重或引擎配置。响应 model alias 必须精确相等，
不发明服务端 revision 字段。

客户端使用冻结原生 chat 字段：一位置生成、全部答案与U的指定 IDs、top_logprobs=0、
无采样变换、thinking=false、return_token_ids=true。服务响应顶层最终展开
prompt_token_ids 必须逐位等于 HF input_ids，并与 usage 完整匹配。缺 token、重复/非法
ID、NaN/正值/裁剪哨兵、多位置/多choice、模型声明错配、canonical绑定变化均拒绝；
额外 sampled token 可忽略，不以 top-k 回退补答案。错误只返回静态码。

```sh
PYTHONPATH=/path/to/mac-media-libs OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false /path/to/existing/python -m scripts.s0_prepare \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --asset-root /path/to/verified/public/media --model-dir /path/to/pinned/metadata \
  --profile canonical_http --output reports/new-canonical-http-cpu.json
```

真实媒体/mock HTTP 回归需显式设置 `S0_LOCAL_MODEL_DIR`、`S0_CANONICAL_MEDIA_ROOT`；
未提供时对应集成测试明确 skip，不能作为真实证据。canonical 100题两遍结果与限定
兼容矩阵见专属 `reports/s0-canonical-http-acceptance-001.json`。mock返回的logprobs
没有模型含义，不证明 vLLM processor、真实 API、GPU forward 或概率门槛通过。

## 固定 processor 版本的视频字段差异

固定 Linux Transformers5.17 的官方Gemma4 video processor只声明并返回
`pixel_values_videos` 和 `video_position_ids`；5.19另外返回
`num_frames_per_video`。共享准备要求前两个视频张量，后一字段若存在原样保留，
不合成、不转换；实际已选帧数仍由冻结media配置/解码帧摘要记录。
该兼容修复消除5.17误报“omitted required modality tensors”的422，不表示
跨processor版本张量/概率等价；不同版本的完整输入摘要仍必须分别核对。

证据见 `reports/linux-video-processor-compat-001.json`：Mac5.19真实8/16帧网关CPU通过；
Linux5.17修复前诊断定位明确，修复后有界网关batch仍exit137，Linux真实网关CPU验收
保持pending。未发现FFmpeg subprocess失败，未据猜测调整线程参数，未重跑已知OOM。

## 主服务026的Transformers5.13诊断

`reports/primary-transformers513-cpu-diagnostic-001.json`记录一次独立Mac target检查：
既有026官方pure-Python wheel的SHA与wheelhouse manifest一致；wheel加解包58.15MB。
`PYTHONPATH`先放该target、再放只读媒体target、最后由现有Python使用原site-packages，
没有安装包或覆盖环境。六个公开case选text/image/screenshot/audio/video8/video16。

初次运行在processor加载前阻断：官方5.13要求tokenizers>=0.22.0且<=0.23.0，
Mac现有0.23.3不满足。保留CLI失败报告`primary-transformers513-cpu-001.json`至
`primary-transformers513-cpu-003.json`，未绕过官方依赖校验。

后续通过既有SSH mux中转取得官方macOS arm64 `tokenizers-0.22.2` wheel，
大小2,981,472字节，SHA256 `1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001`，
解包进同一隔离target；root环境未改。`reports/primary-transformers513-cpu-004.json`
是真实CPU canonical_http六case结果，状态为partial（显式六case，不是100题验收），exit 0。
`reports/primary-transformers513-cpu-comparison-001.json`与旧5.19参考逐case比较：答案映射、HTTP
messages hash与input_ids hash全部一致；两条视频case唯一字段差异是5.13缺少`num_frames_per_video`。
这与5.17诊断一致：视频必要字段仍是`pixel_values_videos`和`video_position_ids`，实际帧数
由冻结`frames_indices`/解码摘要约束。

该结果只证明Mac CPU官方processor 5.13在六个公开canonical_http输入上可加载并与5.19参考的
IDs/HTTP绑定一致；不声称native vLLM processor、Linux Torch2.11、GPU forward、概率门槛或生产服务通过。
