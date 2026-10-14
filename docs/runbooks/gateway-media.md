# 受信本地媒体网关 CPU 接入

`POST /v1/systemone` 保持原 state/model/questions 和概率读出。新增 `media`（`data.schema.Media`
引用数组）与 `video_frames`（仅8/16，默认16）。引用必须是运维配置根内的相对常规文件，
携带SHA256；当前本地受控资源入口最多8个引用（含audio/video），超量在准备前422。
这是S0资源limit，后续产品多媒体组合支持仍需单独验收。HTTP/文件URL、绝对路径、逃逸、软链接、坏hash、空/超限/坏媒体均拒绝。
本接口没有上传或任意 URL 下载能力，也不接受调用者设置媒体根。

配置：`LITE_MEDIA_ROOT`、固定 metadata 的 `LITE_MODEL_DIR`、`LITE_UPSTREAM_URL`、
`LITE_UPSTREAM_MODEL`、`LITE_MODEL_VERSION`，以及**独立核验启动记录后**设置的
`LITE_RAW_LOGPROBS_CONFIRMED=1`。这一标志不能自证服务实际raw模式或权重身份。
processor lifespan仅以local_files_only=true、trust_remote_code=false从本地metadata加载，
不加载权重。CPU测试允许 `create_app(..., processor=official_processor, transport=...)` 注入。

显式配置媒体根后，文本与媒体请求全部走canonical chat，避免文本绕过共同输入准备。
processor/根/后端未配置或加载失败返回503，不暗降为legacy。未配置媒体根的旧文本设置
继续原 `/v1/completions` 路径，且不会强制加载processor；默认文档标题保留以兼容已有
browser smoke。`/health.mode` 明确返回 `canonical-chat` 或 `legacy-text-completions`。
`inference_configured` 只是配置状态；`model_quality_verified` 始终false。

请求示意（替换实际已验sha，不使用虚构摘要）：

```json
{
  "state": "Read the approved local evidence.",
  "media": [{"type": "image", "uri": "image.png", "sha256": "<verified-file-sha256>"}],
  "video_frames": 16,
  "questions": {
    "choice": {"type": "choice", "instructions": "Choose the description.",
               "criteria": {"first": "First", "second": "Second"}},
    "score": {"type": "score", "instructions": "Rate visibility.",
              "criteria": ["Low", "Middle", "High"]},
    "noul": {"type": "noul", "instructions": "Is evidence sufficient?"}
  }
}
```

截图仍引用type=image。canonical PNG、PCM16 WAV、JPEG8/16帧及真实时间戳约定见
`s0-prepare.md`。当前profile一段audio/一个video；图+视频、多audio、多video兼容验证pending，
会显式422，不丢弃后面的媒体。这是当前S0 profile限制，不是最终产品范围变更。
音频<=30秒、视频<=60秒、单文件<=64MiB、图片<=40MP；固定根应只放已验开发媒体，
不放封存集、私钥或其它业务文件。共享decoder协议/format白名单依赖主树42325d7，
该修复不由本网关切片重复提交。

canonical请求先校验全部问题契约，再逐题串行CPU prepare→HTTP→读出，避免多题同时
保留大张量；CPU prepare由app级Semaphore(1)串行并放入线程。10题不会使用legacy的
asyncio.gather。CPU锁由受追踪后台准备任务持有到真实线程结束；HTTP等待者取消不会提前释放锁。
任务完成后收集异常且不记录第三方原文，lifespan关闭前等待已提交准备任务收口。
不同HTTP请求的上游调用可以重叠，本票没有性能或生产并发验收结论。
每题上游完整指定答案加U，要求展开prompt IDs与HF逐位相同；条件答案分布归一化、
p_insufficient独立，calibration=unfitted。usage.output_tokens=0保持产品语义，
backend_usage.output_tokens记录真实上游生成计量，一题应为1。

错误边界：坏输入/媒体/当前未验组合422，未配置503，上游完整概率/ID对齐协议失败502，
超时504。错误详情静态化，不返回源文、媒体URI/base64或第三方异常；Pydantic验证保留
422 detail数组外形，但移除原input/context以及用户字段key。没有任何静态假概率后备。

## 计时边界

所有HTTP响应带标准 `Server-Timing`，固定 `preprocess`、`upstream`、`total`，另有`queue`，
单位毫秒；只含数字和固定label，不含源文/媒体内容。

- preprocess：各题实际CPU prepare的monotonic时间之和，包含媒体读取/校验/解码/
  canonical编码/官方processor与输入摘要，排除CPU锁与线程排队；不含lifespan加载。
  legacy文本对应token/template准备之和。
- queue：各题CPU锁和线程开始前等待时间之和。
- upstream：各题HTTP await及完整响应读取/协议解析的时间之和，包含网络、服务端
  预处理、调度、GPU/采样等；不是GPU纯推理，也不是可直接冒充的TTFT。
- total：middleware接收请求到响应头可返回的整个网关墙钟，含请求解析/读出/序列化，
  不含响应体网络传输完成。canonical题目串行；legacy并发的各题sum不能当whole wall。

测试只检查计时label、非负有限数与隐私，不设置合成性能阈值。顺序测试的mock等待仅
用于制造调度交错，不是时延或cache效果证据。真实GPU时延、缓存收益、纯推理时间和
浏览器操作成功仍pending。

## CPU 证据入口

显式设置 `S0_LOCAL_MODEL_DIR`、`S0_CANONICAL_MEDIA_ROOT`，使用共享只读Mac媒体target
与既有root Python运行 `pytest tests/test_gateway.py tests/test_gateway_media.py`。
新测试以实际许可媒体、官方HF processor及ASGI FastAPI请求验证五类输入/8和16帧、
三Primitive、多题与10题顺序、固定100题同canonical参考的HTTP/tensor hash绑定、真实
local-only lifespan及负测。上游由MockTransport返回合成logprobs；未执行vLLM/GPU，
不能把200响应称为真实模型/API产品验收。缺显式媒体环境时对应集成项会skip。
小型汇总保存在专属 `reports/gateway-media-cpu-001.json`，旧报告不覆盖。

取消回归使用ASGI请求与受控阻塞的真实线程：取消首请求后第二请求不能并行准备；
释放线程后第二请求静态422，shutdown等待被取消请求遗留线程收口。此测试只证明
取消下的CPU串行边界与线程清理，不代表强制终止processor或生产shutdown时限验收。
