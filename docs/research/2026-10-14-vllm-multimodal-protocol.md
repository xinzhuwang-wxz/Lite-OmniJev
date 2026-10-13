# S0 多模态原始概率传输：固定版本源码核验

2026-10-14只读核验vLLM官方v0.26.0/v0.31.0 tag；不启动服务、不修改引擎，不构成真实概率或GPU兼容验收。P0文本completions路径保留；S0可以增加原生chat适配器，实际集成仍待实现和验收。

两个版本的ChatCompletionRequest均支持`logprob_token_ids`，要求`logprobs=true`。指定IDs非空时，chat serving序列化使用`return_all=True`，不依赖`top_logprobs`截取答案。`return_tokens_as_token_ids=true`使结果可按`token_id:N`提取；额外采样token可忽略，缺失、重复、非有限、正logprob或`-9999`哨兵继续拒绝。completions没有原生媒体字段，不能给既有文本请求拼media后就声称接入完成。

来源：[0.26 chat协议](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/chat_completion/protocol.py)、[0.31 chat协议](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/protocol.py)、[0.26 chat serving](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/chat_completion/serving.py)、[0.31 chat serving](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/serving.py)。

## 固定请求与响应

使用`max_completion_tokens=1`、`stream=false`、`n=1`、temperature/top_p为1、`top_k=-1`、`min_p=0`、presence/frequency penalty为0、repetition penalty为1、seed42、`logprobs=true`、`top_logprobs=0`、完整答案加U的`logprob_token_ids`、`return_tokens_as_token_ids=true`、`return_token_ids=true`、`add_generation_prompt=true`、`add_special_tokens=false`及`chat_template_kwargs.enable_thinking=false`。不请求或持久化原prompt文本；最终prompt token IDs在内存比较并写哈希。

响应读取`choices[0].logprobs.content[0].top_logprobs`，按`token_id:N`映射，不依赖数组顺序。服务端显式`--logprobs-mode raw_logprobs`；指定IDs从sampler输入的全词表FP32 log_softmax中gather，不能对答案子集softmax冒充全词表概率。原始概率、合法K+1联合分布、去U条件分布、p(U)分别保存，门槛不变。

来源：[0.26 sampler](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/sample/sampler.py)、[0.31 sampler](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/sample/sampler.py)。

## 媒体与输入一致性

- 图像：chat内容使用`image_url.url=data:image/png;base64,...`；截图同样传image，fixture保留screenshot类别。
- 音频：`type=input_audio`，`input_audio.data`为完整WAV base64、`format=wav`；采样率、通道、时长与HF一致。
- 视频：预提帧协议为`video_url.url=data:video/jpeg;base64,<JPEG0>,<JPEG1>,...`，同时提供真实`media_io_kwargs.video`：fps、frames_indices、total_num_frames、duration、do_sample_frames=false；indices长度等于帧数。Gemma4使用indices/fps产生时间戳，不能用无时间信息的image序列冒充同一视频路径。

JPEG有损，HF必须消费同一JPEG字节重新解码的像素；编码前RGB hash只证明源抽帧。若改用原生视频容器路径，也必须实测采样元数据和最终tensor，不凭相同文件hash推定processor相同。

来源：[0.26视频IO](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/multimodal/media/video.py)、[0.31视频IO](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/multimodal/media/video.py)、[官方多模态输入文档](https://docs.vllm.ai/en/v0.31.0/features/multimodal_inputs/)。

0.31的Gemma4视频分支额外清空images_kwargs，0.26只覆盖视频max_soft_tokens配置。跨版本须比较最终prompt IDs、placeholder范围、像素/position IDs、音频特征/mask、视频时间戳及缓存冷热；图像position key有重命名，比较前明确映射。官方processor测试不需加载主干，但项目2GiB无卡环境能否运行仍须实测。

来源：[0.26 Gemma4](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/gemma4_mm.py)、[0.31 Gemma4](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/gemma4_mm.py)、[官方processor测试](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/models/multimodal/processing/test_gemma4.py)。

## raw边界与当前快照

Gemma4模型compute_logits可按generation_config.suppress_tokens在sampler之前抑制token。`--generation-config vllm`不能单独证明此分支关闭。当前固定E4B `ee0ef6023621cff504d758262d4e04895a5af4a2/generation_config.json`实际读取208B，SHA256 `d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de`，没有suppress_tokens；默认top_k64/top_p.95由请求显式覆盖。不能推断这就是现有parity失败原因。

来源：上列Gemma4实现及[0.31配置加载](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/model.py)；文件身份见`reports/cpu-media-probe-offline-runtime.json`。

## 验收分界

无卡先完成真实媒体fixture/官方processor、chat序列化、指定IDs响应负测和重复输入摘要；源码支持与mock只能证明协议准备。实际服务启动、展开后输入对齐、独立LoRA forward、100题严格<1e-3、两遍完全一致、网关/浏览器真实成功和性能仍须GPU实测。缺失适配代码不能列作GPU-only，应在开卡前实现。
