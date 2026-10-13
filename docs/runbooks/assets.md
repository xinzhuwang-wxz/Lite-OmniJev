# 无卡资产准备与 GPU 交接

所有命令从仓库根运行。这里不租卡、不接受 gated 模型条款、不载入权重、不执行远端 Python。
`huggingface_hub` 负责下载和断点续传，`tokenizers` + Jinja2 sandbox 仅验证文本分词/官方模板。
先 `uv sync --frozen --group dev`；完整模型不要下载到空间不足的开发机。

## 1. 冻结清单

```sh
uv run python -m scripts.assets plan --config configs/assets.json --model e4b --output artifacts/manifests/e4b.json
```

每个模型只请求一次 `HfApi.model_info(files_metadata=True)`，将分支解析为不可变 commit SHA，记录模型卡许可、来源、每文件大小、LFS SHA256 或 Git blob SHA1。默认仅 E4B；`--model e2b` 或重复 `--model` 可选择额外模型。
`teacher` 配置故意未启用：先核实原方案允许的官方教师仓库、revision 和许可，再填配置；未核实会失败，不下载未知教师。Jev-Omni 的 `head.pt` 还需经审计的 safetensors 转换及来源记录；不能跳过必需 head 却宣称教师完整。
清单只允许 safetensors 权重、配置、tokenizer、processor 和许可说明；排除 pickle、`.bin` 和自定义 Python。
`accepted_licenses` 表示项目审查过的模型卡许可，不代表脚本代替用户接受外部条款。

## 2. 本地仅下载元数据

```sh
uv run python -m scripts.assets download --manifest artifacts/manifests/e4b.json --root artifacts/models --metadata-only --workers 2
uv run python -m scripts.assets verify --manifest artifacts/manifests/e4b.json --root artifacts/models --metadata-only
```

默认报告为 `reports/assets-last-run.json`。保存独立报告时，把全局选项放子命令之前：
`python -m scripts.assets --report reports/assets-metadata.json download ...`。
`metadata_verified` 不表示权重就绪。下载后逐文件校验上游哈希，并记录实际 SHA256；已有文件先校验，损坏文件不覆盖。
遇到损坏先留证据、定位文件，在得到删除授权或移动到隔离位置后重新下载；不要篡改 manifest 哈希来“修好”验证。
下载保留模型目录内 `.cache/huggingface/` 元数据，重跑同一清单可继续；网络异常返回非零并记录异常类型，避免把签名 URL/凭证写日志。
磁盘预检使用未完成文件完整大小 + `max(64 MiB, 10%)` 余量，保守地不抵扣部分下载；线程限制 1–4。

## 3. 真 tokenizer 验证

将 `<SHA>` 换为 manifest 内的 `revision`，不要填写 `main`：

```sh
uv run python -m scripts.check_tokens --model-dir artifacts/models/google__gemma-4-E4B-it/<SHA> --output reports/e4b-token-probe.json
```

验证 A–T、U、0–9、yes/no 共 33 个答案：单 token、ID 无碰撞、实际生成位置拼接后仍是同一个 token；官方模板显式传 `enable_thinking=false`，并以开启 thinking 的渲染作对照；记录公共前缀长度、token 映射及模板/tokenizer SHA256。
使用 `scripts.check_tokens.OfficialTokenizerAdapter(Path(...))` 可共享此 CPU 文本渲染入口（`encode` / `apply_chat_template`）；正式 train/template 与服务必须使用同一固定版本模板，并另证 HF/vLLM 读出等价。
这个探针仅覆盖文本模板，不覆盖多模态 processor、模型前向、GPU、概率或推理引擎；这些门槛保持 pending。
缺文件、鉴权、模板不支持或分词失败都返回非零；不能用 mock 测试代替模型实测。

## 4. 用户租无卡主机后

1. 主会话核实主机目录/磁盘/连接方式，用户提供必要账号；把代码、锁文件、固定 manifest 带到主机。
2. 按 manifest 的 `total_bytes` 总量预留额外数据、缓存、环境空间；不能按 E4B“有效参数”估算磁盘。
3. 在主机执行同样的 `download`，去掉 `--metadata-only`；默认 2 线程。不要在开发机误跑。
4. 全文件 `verify`，重复 token probe，保存固定清单、实际 SHA、退出码与主机环境证据。
5. 数据资产由独立的数据来源/许可/拆分流程处理；此脚本不宣称已经下载或验证数据集。
6. 只有代码环境、模型、阶段 0 数据及校验全部就绪，才通知用户开卡；GPU 作业完成后主会话保存产物并通知用户关卡。

## 官方依据

- [HfApi.model_info 与 RepoSibling 元数据](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
- [hf_hub_download local_dir 与文件下载](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)
- [Gemma E4B 模型与官方模板](https://huggingface.co/google/gemma-4-E4B-it/tree/main)

模型分支会变化，以上链接用于核查方法；实际运行只认 manifest 固定 SHA。
