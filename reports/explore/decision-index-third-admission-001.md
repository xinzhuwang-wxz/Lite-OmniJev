# Decision Index 第三批三项许可核查

仅 catalog 1 BFCL、3 API-Bank、58 BBH。基准为已批准16项 manifest SHA `e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f`。本轮新增准入建议 **0**；三项均列明确待核缺口。未修改远端 manifest、未读题/答案行、未运行上游/pickle/GPU、未接受条款、未下载数据。远端仅计算现有输入字节 SHA256/git blob SHA1，输出文件元数据。

| 项 | 新的官方事实 | 固定输入绑定与缺口 |
| --- | --- | --- |
| BFCL | 官方 HF **数据卡**明确 Apache-2.0，区别于原先仅 repo/model LICENSE。 | 数据卡固定 revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` 的树元数据与 recipe GitHub `916260dfc116bf06793a1af79b4ec8195b0453b6` 的41输入：16 blob相同、25不同。需旧发布版本适用的数据许可/映射；不把当前数据卡自动追溯不同文件。 |
| API-Bank | 固定 README明确公开 code/data，并链接发布者 HF；当前固定 HF card `12e8158b7628c168f07e8f31fbbe3445e99f44cf` 标 MIT。 | GitHub `188835d4f9948563a6b9c8ac50cd0f3ae4021ed6` 子目录 LICENSE为Apache-2.0，固定319输入同时含 API代码与逐对话样本；HF按level重新打包。需明确旧样本许可适用范围或经验证转换映射，不能任选 MIT/Apache填整项。 |
| BBH | 固定 README明确为 BIG-bench子集，BBH根MIT；官方 BIG-bench README要求保留canary，父repo LICENSE为Apache-2.0。 | BBH固定 `9ee07bd481feebf959a6b59d61ea57bdcf30964d` 的27文件已核 SHA；逐任务底层notice/许可链未核，不将BBH根MIT替代该链。仅evaluation、永不训练、保留canary/署名/原notice、不重分发的边界不变。 |

这里的待核不是认定禁止评测，而是当前 metadata 证据尚不足填写无条件 `evaluation_admitted=true`。不扩展到其余25项，也不读取BBH任务来补证。

官方固定元数据入口：

- [BFCL dataset card](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/raw/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/README.md)
- [API-Bank fixed README](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/api-bank/README.md)；[publisher card](https://huggingface.co/datasets/liminghao1630/API-Bank/raw/12e8158b7628c168f07e8f31fbbe3445e99f44cf/README.md)
- [BBH fixed README](https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/9ee07bd481feebf959a6b59d61ea57bdcf30964d/README.md)

完整 URL/revision/metadata SHA/输入 SHA 与16/25文件清单见同名 JSON。BIG-bench父metadata为本次解析main后固定commit，不冒称与2022 BBH同时快照。脚本输出exclusive-create proposal，不应用准入；CPU公共输出及变更SHA拒绝测试3 passed，ruff通过。基础完整manifest未复制入Git；复现时只读原远端manifest并核上述SHA。
