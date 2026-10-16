# Decision Index 第三批：固定发布方数据许可

仅 catalog 1 BFCL、3 API-Bank、58 BBH。基准为已批准16项 manifest SHA `e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f`。**提案 BFCL、BBH 两项仅本地离线 evaluation；API-Bank交root判断固定subproject Work范围。未应用任何准入。**

本报告修正初查：后来的HF文件不相同不能否定旧Git发布方明确许可；未查看逐任务条款不能推断无许可。关键证据为固定Git数据目录README，而非未来HF双射。未修改远端manifest、未读题/答案行、未运行上游/pickle/GPU、未接受条款、未下载数据。现有输入仅流入hash计算，输出元数据。

| 项 | 固定发布方事实 | 提案及边界 |
| --- | --- | --- |
| BFCL | `916260dfc116bf06793a1af79b4ec8195b0453b6` 的 **data/README.md第2行直接标Apache-2.0**；正文明确本evaluation dataset、手写函数、企业/OSS贡献live数据和多轮构造来源。SHA `22234ad32a0bb921809278d01ed65e932b4e611690dfa17d1e0eb6ba74ab4e27`。固定树该模块仅module README与data README，无另列LICENSE/NOTICE文件。 | 可提交Apache-2.0 evaluation-only提案，保留license/attribution/change notices，不执行第三方API、不接受provider条款、不训练/重分发。当前HF卡同Apache，16/41 blob相同25不同只作版本事实，非准入阻碍。 |
| API-Bank | `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` 的README明确发布code/data、作者标注evaluation dialogues与API工具，并定位lv1-lv2目录；子目录Apache LICENSE没有写仅code。该固定子树仅README/LICENSE，没有另列样本许可。 | 不声称代码许可自动覆盖数据。root需判断整个subproject Work授权是否涵盖其发布的样本，或取得发布方明确scope。后续HF卡MIT是发布方数据授权旁证，但不是自动溯及旧样本的MIT许可，也无需以未来打包双射作为唯一授权路径。fixed README未披露evaluation样本来自另有条款的第三方语料；这不证明第三方权利绝不存在。 |
| BBH | `9ee07bd481feebf959a6b59d61ea57bdcf30964d` 根MIT，根README明确BBH数据目录和BIG-bench来源；**bbh/README.md第5行明确禁止进入训练语料**。该数据README SHA `2273d9fd5fc2fd175deb939b99ed52e2945b0225dd1f290a836308ca36efec18`。 | 可提交MIT evaluation-only限定提案，保留父源/逐任务原notice、引用和canary，永不训练、不重分发。本轮没读逐任务行，不主张原notice已全部检查，更不移除它们。BIG-bench当前固定父metadata要求保留canary；该父revision不冒称BBH当年的snapshot。 |

官方固定元数据：

- [BFCL data README](https://raw.githubusercontent.com/gorilla-llm/gorilla/916260dfc116bf06793a1af79b4ec8195b0453b6/berkeley-function-call-leaderboard/data/README.md)
- [API-Bank README](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/api-bank/README.md)；[subproject LICENSE](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/api-bank/LICENSE)
- [BBH data README](https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd481feebf959a6b59d61ea57bdcf30964d/bbh/README.md)；[BBH LICENSE](https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd481feebf959a6b59d61ea57bdcf30964d/LICENSE)

同名JSON含2个待root review admission objects、全部387固定输入SHA及官方metadata的URL/revision/SHA。脚本exclusive-create proposal，不应用准入；CPU公共输出及输入/base SHA拒绝测试3 passed，ruff通过。基础1530文件manifest未复制Git；复现只读原远端manifest并核基准SHA。
