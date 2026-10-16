# 网关温度registry：CPU准备接口

默认`create_app()`不加载registry，仍返回`calibration: unfitted`及既有概率；没有新增env开关，没有启用实际校准。显式入口供受信任本地bootstrap调用：

```python
registry = load_registry(
    registry_path, serving_receipt=live_native_receipt,
    asset_manifest=pinned_model_manifest, model_dir=official_snapshot,
    upstream_url="http://127.0.0.1:8014", upstream_model="decision",
)
app = create_app(settings, calibration_registry=registry)
```

实际launcher的同版本接线、真实校准导出/来源准入、正式S5激活与真实后端验证仍pending。CPU fixture只验证文件/进程身份/HTTP概率逻辑；不证明GPU模型或质量门。

## 可接受的普通本地信任边界

复用现有`launch-native-routing.py`的`argv/server_pid/ready`结构，loader核当前Linux`/proc/<pid>/cmdline`与receipt一致（允许vllm shebang添加Python前缀），再从实际argv解析原始模型目录、loopback host/port、raw_logprobs模式和指定LoRA的alias=absolute_path；远程URL、只有自报alias/布尔flag/hash的配置、已停止receipt均拒绝。调用方拥有受信任本地bootstrap与固定model asset manifest；不防root篡改，也不新加远程密码学证明门槛。

启动时复用`verified_assets`全文件hash核实际基座、核实际adapter safetensors/config、运行官方tokenizer/template probe，再检查registry每个artifact SHA和严格完整Binding。仅支持现有标准native base+LoRA、同官方tokenizer模板的recipe；merged、quantized、hf-config/chat-template override需对应真实派生/量化身份绑定，当前明确拒绝，不静默冒充原基座。

现有native lifecycle报告在停止后才落盘，不能直接拿历史报告启用。现`compare-merged-reference.py`另有观测live PID的merged报告，但本slice未扩merged链。正式bootstrap需要在实际native process ready期间保存已观察的argv/PID/ready并调用上述接口；不用修改Model/Media schema，不会自动启动任何服务。Mac没有Linux/proc时此loader也不会捏造身份通过。

进程/文件身份在加载结束再次检查；请求只做cmdline和文件dev/inode/size/mtime_ns检查、registry快照绑定及实际prepared答案token IDs比较，**不重复hash基座/adapter**。停止/换进程、换文件/参数、温度artifact修改均failclosed；没有质量通过声明。固定文件应由部署账户只读挂载，文件stat不是抵抗root刻意恢复元数据的认证。

## Registry与映射契约

JSON字段：`schema_version:1`、`model_version`、`calibration_version`、`groups`。每组为`path`（registry根内相对路径）、`sha256`和完整`train.calibration.Binding`。Binding必须与实际base revision/adapter weights/template/token map一致，其余source receipt/data/seed/code字段必须与对应artifact严格相同。artifact路径逃逸、checksum/身份不符、未知字段、重复组、缺组均拒绝；版本必须与gateway settings一致。

**要求六cohort×三primitive完整18组配置**：text/image/screenshot/audio/video/crossmodal × choice/score/noul。当前Media只有image/audio/video，推断规则是无media→text、单一media种类→该种类、两种以上media→crossmodal；多张image仍image，所有请求都有文本state所以不因此算crossmodal。当前只有15组可路由；screenshot三组虽可配置和校验，仍待可信来源/分类入口，不能让请求自报screenshot绕过，也不改Media schema。18组加载成功不代表screenshot链或S5完成。

`answer_token_map_sha256`在此明确采用官方probe的全部33个答案token（A–T、U、0–9、yes/no）到ID字典的`sha256(json.dumps(mapping, sort_keys=True).encode())`。这是部署全局映射；逐题子集mapping hash不能替代它。正式calibration producer需使用同一约定。启动另核实际gateway renderer模板字节和tokenizer编码；请求核实际prepared子集IDs，避免同名字母被错贴概率。

raw backend仍必须返回完整未采样/未截断的legal答案+U。完整原始logprobs相对logits只差同位置公共常数，除以T后在legal集合归一化等价；不改变backend采样temperature=1。应用复用已有readout，U先joint、答案去U条件归一化。返回`scalar_temperature`、calibration_version、registry/artifact SHA和实际cohort，`model_quality_verified`仍false，stage quality仍pending。

## 验证及缺口

新CPU测试使用明确合成权重/模板/温度和mock `/proc`/official probe，不冒充真实模型。覆盖实际文件篡改、版本/四项部署绑定、缺/重复组、完整Binding失配、artifact checksum/path拒绝、18配置/15路由、runtime tokenizer/template失配、joint U/conditional温度变化、HTTP默认/显式校准与停止进程503、请求不重hash权重。既有gateway/scheduling回归一并运行。

本机受影响真实media fixtures依赖`S0_LOCAL_*`许可媒体/metadata路径；当前未提供，已有测试明确skip，不能称真实processor/媒体/GPU校准已验收。没有读取真实calibration/holdout、拟合、GPU/服务启动、依赖变更或质量门调整。

本片实际：`pytest tests/test_gateway_calibration.py tests/test_gateway.py tests/test_gateway_media.py tests/test_gateway_scheduling.py -q --tb=short`：53 passed、38个既有media fixture skip；新registry/HTTP测试全部执行。`ruff check`、`ruff format --check`及`mypy serve/gateway/app.py serve/gateway/calibration.py`均exit0。初轮默认路径额外读取prepared mapping导致5个既有调度fixture失败，已将该读取严格限制为显式registry模式；上述为修复后fresh结果，未修改原测试掩盖回归。
