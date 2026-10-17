# Lite-OmniJev 工作契约

## 开工与范围
- 每轮先读 `plan/STATUS.md`、`docs/02-project.md`、当前票；领域词与决策读 `CONTEXT.md` 和相关 `docs/adr/`。
- 原始需求是 `docs/Lite-OmniJev设计方案-最终版.html`；保留原件，冲突用 ADR 记录；性能/资源估值按实测校准，其余门槛变更须明确决策。
- 一票交付一个业务闭环，可跨session恢复；模型、来源、探针与普通修复放票内清单，不按小步骤拆票。先冻结验收、范围、依赖。
- 先读现有代码和依赖，估算改动，复用现有能力；不为假想需求抽象，不顺手修非阻断问题。
- 新发现的非阻断问题建票；阻断问题并入当前闭环或建明确依赖，所有交付范围内问题最终收口。
- 默认直接执行本地可逆工作；不启用 superpowers/OMX 重型流程，不重复做需求访谈。
- 优先 mattpocock 的 to-spec → to-tickets → triage；实测驱动优化可用 autoresearch，须先定指标和退出条件。

## 项目硬约束
- Python 3.11；Pydantic 数据契约，Transformers/PEFT 训练，FastAPI 网关，vLLM GPU 推理。
- 冻结 Gemma 4 E4B/E2B 主干，只训决策 LoRA；不新增分类头、不改推理引擎；偏离须 ADR 与用户决策。
- 训练/服务共用官方 chat template 和答案 token 映射；明确关闭 thinking；单 token 检查通过才训练。
- Choice 单组 2–20 个答案，另加 U；Score 2–10 个等级，Noul yes/no；单选项退化题拒绝，避免置信度除零。
- p(U) 单独返回；答案分布去 U 后归一化；不得把受采样变换或截断 top-k 的分布当原始概率。
- 样本真值优先于教师；不使用 Jev API 输出、JEV-27B/JEV-27B-VL 权重或其标签。
- 模型/数据入库记录来源、revision、许可与文件哈希；不明许可不进训练，研究数据与商用数据隔离。
- train/dev/calibration/holdout 按原始样本族隔离；变体和反事实不能跨池；封存集由自动 gate 独立作业读取、只回汇总；按ADR0012用现成可靠gold，不要求新增真人双标/κ或用户提供gate账号。
- 开发集可调试，封存集不得给 agent 看内容；阶段 4/5/6 仅接收汇总门槛报告。
- 固定种子；报告带代码版本、数据哈希、配置、运行环境、耗时、成本和退出码；失败也留证据。
- 进阶优化、组合创新与新想法按阶段/实测瓶颈自主实验，先小批量后扩量，预算与停止条件见ADR0015；5b多步与生产运营扩张仍须范围决策。
- "SOTA"只按ADR0025+0026+0027锁定的四公开榜（Decision Index/JevBench231/MMAU/MVBench九任务）判定：我方模型官方协议实测，对照用其公开已发布数（无公开数如实标注，不为对照烧卡）；CLEVRER、封存池、dev切片分数不得称SOTA。
- E4B/E2B 的有效参数不等于驻留权重大小；租卡与下载按实际文件清单估算。

## 命令与目录
- 当前可用命令及验证状态以 `docs/02-project.md` 为准；新增命令必须实跑后才能标为可用。
- `uv sync --frozen --group dev` 安装本地环境；`uv run pytest` 测外部行为。
- `uv run ruff check .`、`uv run ruff format --check .`；`uv run mypy data train eval serve scripts` 做类型检查。
- 训练入口先 `uv sync --frozen --extra training --group dev`；CPU验证 `make training-check`，GPU实跑按训练runbook。
- `uv build` 构建 Python 包；`uv run uvicorn serve.gateway.app:app --host 127.0.0.1 --port 8000` 本地开发。
- `data/` 数据契约/构建/去污染；`train/` 共享模板/训练/校准；`eval/` 指标和门槛。
- `serve/` 网关与部署适配；`configs/` 阶段配置；`infra/` 可复现环境；`scripts/` 操作入口；`tests/` 行为测试。
- `docs/agents/` 技能配置；`docs/adr/` 决策；`docs/runbooks/` 操作；`reports/` 小型证据；`plan/` 状态与 handoff。
- `.scratch/<feature>/spec.md` 和 `issues/NN-*.md` 是本地规格/票据，必须进 Git；大权重/数据/媒体放忽略的 `artifacts/` 或外部盘。

## 实现与验收
- 先领域模型、权限、状态和操作闭环，再视觉；Playground 基础控件优先 shadcn/ui、Animate UI。
- 新依赖写明用途、复用理由与许可；按当前阶段引入，不预装完整训练/云基础设施栈。
- 只测公共接口的可观察行为；选择最窄稳定接口；不要为低风险文案或文件改名写镜像测试。
- Web 交互在真实浏览器验收，优先 Playwright MCP；不可用时用 Playwright 实际 Chromium 并注明替代。
- mock/合成结果、CPU 管线测试不能代替真实模型/GPU/端侧验收；没有证据的门槛只能写 pending/blocked。
- 普通修复只跑受影响行为测试；一批代码合并后按需跑lint/typecheck/build。已通过且代码/输入未变的证据直接复用，不因新轮次重跑。
- 里程碑按 `docs/02-project.md` 跑真实验收，保存命令、退出码、报告/截图；全绿才进入下一里程碑。
- 独立review集中到里程碑集成；鉴权/隔离/迁移等高风险改动按风险复核。普通探针、数据来源调查、文档和小修复不另加review。

## 循环与并行
- 原生 goal 负责终局；执行算法见 `plan/LOOP.md`；STATUS 每轮先读后写，不以聊天记忆当状态。
- 每轮调用工具并产出代码、票据变化、验证结果或已证实阻塞；纯文字、重复检查不算进展。
- 有依赖完成的业务闭环直接执行；无frontier时在当前票内展开下一可执行步骤，不靠新票/文档制造进度。全项目粗粒度路线见 `plan/DELIVERY-PLAN.md`。
- 当前里程碑全部通过后验证、复盘 AGENTS、写 handoff，再展开下一批；终局验收票最后执行。
- 并行任务各用独立 git worktree，任务书写绝对路径、所有者和模块范围；禁止两个执行者同时写同一模块。
- 子代理用 `fork_turns="none"` 配完整任务书；每次 shell 显式 `workdir`，patch 用 worktree 绝对路径，禁止在主树提交或更新票据/STATUS。
- 主会话统一认领/整合/关票，子代理只交指定切片及证据，不回滚别人修改；完成后立即更新票和 STATUS。
- 批量检索/浏览器操作可派给当前工具实际提供的轻量模型；不可假称支持 Sonnet/Haiku。
- 无子代理或外部作业运行时不得空等；全部剩余工作依赖用户时写清阻塞并交接，不忙轮询。
- 维护精简STATUS与一个当前接续入口；只在里程碑出口或用户要求换session时写完整handoff，小切片不另造交接文件。
- 踩坑按“问题 → 原因 → 解法”写对应目录 AGENTS；先删过时条目，根文件 ≤150 行，子目录 ≤40 行。

## Git、资源与授权
- 本地小步提交，信息说明为什么，按 Lore trailers 记录约束、验证与缺口；不把每轮产出等同于远端推送。
- 默认不 push、不创建远端 issue/PR；有明确同步需求时按里程碑批量操作，禁止高频 gh 轮询和循环重试。
- 禁止 force push 主分支、提交密钥/权重/原始业务数据；不得覆盖或清理用户原有工作。
- 仅清理自己创建且已无用的进程、容器、worktree、合并分支；权重缓存除非明确过期不得擅删。
- 租卡/开卡/关卡、付费服务、生产部署、删数据、账号/密钥需求、产品方向/范围改变须用户决定。
- 整个方案按ADR0010自主用实测校准预估性能/资源目标，记录原值、证据、新标准与边界，限额优化后推进，不为未验证估值反复耗卡；质量/正确性门槛、大规模放量、突破预算仍须用户决定。
- 需要用户时记入 STATUS“待我决策”并通知；继续不依赖该决定的工作。
- 开卡前完成全项目必需外部模型/数据/依赖下载与哈希、CPU准备验收；未来训练产物标待生成，账号/许可缺口不算就绪。GPU 用完记录资源并通知用户关卡。

## Agent skills
### Issue tracker
本地 Markdown 是唯一票据源，远端仅按需镜像；见 `docs/agents/issue-tracker.md`。
### Triage labels
使用默认五状态标签；执行状态与标签分离；见 `docs/agents/triage-labels.md`。
### Domain docs
单 context：根 `CONTEXT.md` + `docs/adr/`；见 `docs/agents/domain.md`。
