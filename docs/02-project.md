# Lite-OmniJev：交付与验收契约

原始方案：[最终版](Lite-OmniJev设计方案-最终版.html)。此文负责执行顺序、命令和验收入口，不改原方案指标。
范围：单步多模态判断的 API、单卡包、E2B 端侧包；阶段 5b 和第十一节探索默认不启用。
版本、成本、准确率、时延均需实测；P0工程闭环已完成（reports/p0-engineering-acceptance.md）。当前S0，跨引擎概率门槛尚未通过，S1禁止启动，不能宣称模型质量达标。

## 当前基线与栈

2026-10-13 初始本地只有 HTML，远端只有 Apache-2.0 LICENSE；不存在可复用应用或真实 build/test 脚本。
因此新建最小 Python 3.11/uv 项目：Pydantic 2、FastAPI、httpx、Transformers/PEFT（到 GPU 票才安装）、vLLM（Linux CUDA 独立环境）。
轻量依赖及选择理由见 `dependencies.md`。本机 macOS arm64，不能把本机测试当 CUDA 兼容证明。
实验管理按用户要求使用 SwanLab（Lite-OmniJev / maxen），仅上传真实训练指标；凭据置于项目外，见 ADR-0003。数据版本初期使用内容哈希 manifest。
独立原生前端到 Playground 票才引入 React/shadcn/ui/Animate UI；当前 API 文档页面不构成最终 Playground。

## 本地常用命令

下面是落地接口；最终运行状态与原始输出见 `reports/bootstrap-validation.txt`，缺失报告时视为未验证。

| 工作 | 命令 | 适用范围 |
|---|---|---|
| 安装 | `uv sync --frozen --group dev` | CPU 开发环境 |
| 测试 | `uv run pytest` | 外部行为与失败路径 |
| lint | `uv run ruff check .` | 静态问题 |
| format | `uv run ruff format --check .` | 格式 |
| typecheck | `uv run mypy data train eval serve scripts` | 严格类型 |
| build | `uv build` | Python wheel/sdist |
| dev | `uv run uvicorn serve.gateway.app:app --host 127.0.0.1 --port 8000` | 仅本机；真实推理需配置后端 |
| 队列 | `uv run python -m scripts.loop list` | 显示实际可认领票 |
| 校验规划 | `uv run python -m scripts.loop validate` | 引用、依赖环、状态与所有权 |
| CPU smoke | `uv run python -m scripts.smoke --output reports/cpu-smoke.json` | 合成数据→读出→指标→门槛，不是模型训练 |
| 模型资产 | 见 `runbooks/assets.md` | 锁 revision、下载、哈希校验、token probe |

每次改模板、读出、校准或数据过滤，都要跑对应行为回归；真实 HF/vLLM 100 题对照独立在 GPU 作业中执行，不能在 CPU CI 静默 skip 后声称通过。

## 里程碑与逐步拆票

| ID | 出口 | 何时细拆 |
|---|---|---|
| M0 | 根规则≤150行、技能配置、规格/票据/loop、锁环境、CPU 核心、资产工具与本地证据 | 当前 |
| P0 | 无卡机准备完成后，真实基座 100 样本→训练10步→评测→服务；配置/环境/权重/数据可追溯 | M0 后细拆 GPU 与环境票 |
| S0 | 基线和评测框架通过全部阶段0门槛；包含教师复现、100题概率一致性、缓存、标注一致性 | P0 就绪后，先做 tokenizer/模板先决探针以免循环依赖 |
| S1 | 2k/300步≥98%；文本≥OneJev-4B−3pp、较E4B零样本+5pp、ECE≤.06、30k较10k+2pp | S0通过 |
| S2 | 图像/截图≥OneJev-4B−3pp；文本下降≤1pp；反事实翻转≥.80；打包差<1e-3才启用 | S1通过 |
| S3 | 音频/视频≥教师90%；旧模态下降≤1pp；8/16帧对照；约24 H100时可行性复盘 | S2通过 |
| S4 | 各模态≥最强教师97%；封存/开发差≤3pp；外部榜不低于同尺寸基线 | S3通过且放量预算明确 |
| S5 | 置换≤.02；正反偏差≤.05；每模态ECE≤.03；coverage@5%≥.65；掉分≤.5pp；R1结论和封存门槛 | S4通过 |
| S6 | 量化掉分≤1pp/ECE≤.035；E2B各模态≥E4B92%；Q4内存≤3GB；三运行时/端侧/压测达标 | S5通过 |
| FINAL | 三档发布物完整、跨功能业务闭环、真实浏览器验收、无交付范围未解决票；公开发布另获授权 | 最终验收票现在建立，细项随阶段结果补齐 |

P0 包含 CPU 子集和 GPU 真模型闭环；M0 完成绝不自动把 P0/S0 标完成。先验证 token/template 的 P0 探针不代表 S0 通过。
训练阶段门槛参考原文第六节；机器可读阈值在 `configs/gates.json`，必须全量采集相关模态和每一条件，缺数据=blocked。
每阶段报告必须带样本数、数据 split/revision、基线同条件测量、硬件、种子、代码/配置哈希、费用；只有分数没有可复现证据不能放行。
阈值比例与百分点明确区分（90% 是乘数；−3分是绝对 .03）；不拿不同 split/帧数/提示的模型卡分数直接作匹配基线。
训练 0–3 约24 H100时、全项目约300 H100时均为原方案估计；用户未授权直接购买此预算。教师打标120–200时的上界会逼近/超过总预算，S4前重新核算并提交范围决策。

## 真实浏览器验收步骤

每里程碑先启动已实现服务，优先 Playwright MCP；当前工具无浏览器 MCP，允许 Playwright Chromium，报告明确工具版本。
1. 当前 M0/P0：打开 `/docs`，验证 OpenAPI 加载、health 请求成功；未配置模型时决策请求应清楚失败，禁止展示虚构概率。
2. Playground 实现后：打开页面，依次完成文本 Choice/Noul/Score；检查条件答案分布和 p(U)、版本、时延，复制 curl 重放并比较结果。
3. 逐模态上传固定测试媒体：图像/截图、16kHz音频≤30秒、视频≤60秒（8/16帧）；请求成功、概率和媒体引用正确。
4. 负向流程：非法选项、坏媒体/超限、无 key/越权、429、上游超时、空问题和重复qid；错误可理解且可恢复，无跨租户缓存/原文日志泄漏。
5. 操作链：上传→多题判断→证据不足→修改输入重试→复制 API→查看版本与反馈（默认不留原始内容）；检查窄屏/键盘/加载/错误态。
6. 保存 URL、命令、退出码、截图、trace、网络断言与服务日志。对照已实现范围；未实现步骤列为 pending，不能整张清单打勾。

## 集成票与终局判定

每阶段至少一张集成票：数据→训练/推理→评测→服务；单模块票关闭不能替代它。
最终票 `.scratch/delivery/issues/01-final-acceptance.md` 检查模型、校准、网关同版本；API、单卡、端侧同200题差异；模型卡/数据卡/许可证；部署与回滚；真实性能与用户操作。
API 服务端目标：文本<40ms、截图<60ms、16帧视频<200ms；含网关/预处理 p95 分别≤80/120/400ms，音频≤150ms；分别报告测量边界，不能混用。
可用性99.9%、5xx<.1%、两周SLO、20位外部用户等属于 P4 运营证据；没有生产授权与观测窗口就保持待验收，不用本地压测冒充。
产品范围改变（如端侧音频受限）记录兼容矩阵与待决策；支持的主功能不能靠标记 beta 或 wontfix 静默免除验收。

## 人机边界

常规 spec/拆票/triage、技术验证、里程碑推进由 agent 决定；用户负责租用/开关卡及必须本人完成的账号授权。
封存集人工标注不是 agent 自己生成的“人工真值”；有合规现成人工标注则复用，否则列阻塞，不能伪造 κ。
生产发布、付费、删除数据、重大预算/阈值/方向改变仍须用户决定；继续其它不受影响票。

## 可选训练准备命令

安装：`uv sync --frozen --extra training --group dev`；CPU损失/LoRA验证：`uv run --extra training pytest tests/test_training.py`。
十步真实训练的preflight/run入口见 `docs/runbooks/training.md`；`preflight` 在本机缺权重/CUDA时应返回blocked与非零退出，这个拒绝是预期行为，不能计为GPU通过。
本地跑通的CPU版本锁用于验证代码，实际租机仍须核验驱动/CUDA组合与vLLM的独立环境。

阶段3来源调整（用户批准）：AudioCaps首发省略，按ADR-0004使用Clotho-AQA及额外TTS程序化题补足原音频tranche；保持题量、原样本族隔离和质量门槛，不重复计数。
