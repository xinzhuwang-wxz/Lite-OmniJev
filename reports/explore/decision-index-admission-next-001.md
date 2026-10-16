# Decision Index：固定发布作品的下一批评测提案

基准manifest仍为已批准16项002，SHA `e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f`。本批提案 **3 API-Bank、32 MuSR、37 Amazon ESCI**，只限本地离线evaluation/rebuild，保留署名/原license/NOTICE/变更说明，不训练、不重分发、不执行第三方API。`proposal_pending_root_review`，`applied=false`；root统一决定并应用，不覆盖前一批BFCL/BBH提案。

## API-Bank：建议按发布子项目Apache作品范围准入

固定 `AlibabaResearch/DAMO-ConvAI@188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/api-bank` 的README明确发布代码和数据，说明作者标注的评测对话与API工具，并把level1/2评测内容定位到lv1-lv2目录。实际manifest的319文件是264个lv1-lv2-samples文件和55个apis文件；文件数不是对话数量推算。最近的适用LICENSE位于同一api-bank子项目，为Apache-2.0；其Work/Contribution定义不限软件代码。固定树元数据仅有该README/LICENSE，无另列数据许可或例外。

**建议**：将作者在这个被Apache许可证覆盖的子项目中明确发布的评测样本与API目录视为其发布作品范围，提案Apache-2.0。这是结合发布范围的明确推论，非“代码Apache所以所有外来数据Apache”。本次README没有披露另有条款的外部聚合评测语料；不把未披露当成第三方权利不存在的保证，也不凭这种假想风险额外要求作者出具新确认。训练集、Lynx/Alpaca模型权重、服务执行/外部API条款均不在提案内。后续HF MIT是旁证，不冒称旧Git已变为MIT，也不要求未来HF文件双射才认可固定发布作品。

官方固定[README](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/api-bank/README.md) SHA `3c9347ad1261cffcb8b27092a32fe6ff97e2c2a7662b061152f2042b47869dd9`；[LICENSE](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/api-bank/LICENSE) SHA `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`。复用third-admission已有固定元数据，不重复外查内容。

## 新的两项清晰来源

| Catalog | 固定作品与数据范围 | 建议license和义务 |
| --- | --- | --- |
| 32 MuSR | 固定Git `b1f4d4168a9cfc6760e8b74d728e4516023dfaa5` README说明作者构造的GPT-4合成数据，定位三个domain JSON及生成方法。Git根MIT；**发布者TAUR-Lab/MuSR官方数据卡明确CC-BY-4.0**、列相同三domain、链接同Git source。 | 采用明确的数据CC-BY-4.0而不是code MIT。保留作者/论文/来源、license、改动标记。当前card是发布方数据授权旁证，不冒称后续CSV和旧JSON字节相同或原GitMIT消失。离线评测现有公开版本，不执行生成脚本/调用OpenAI。 |
| 37 Amazon ESCI | 固定Git `7916cdf6ab75a462e77f20ab40428a10923998d5` README以dataset为主体，明确发布examples/products parquet并描述人工标注来源，末尾直接声明project Apache-2.0。 | Apache-2.0，保留Amazon NOTICE及署名/变更说明。该许可是数据发布项目本身的grant，而非独立训练代码仓库借来的许可。仅固定manifest两文件，未读parquet行或运行baseline。 |

官方固定[MuSR Git README](https://raw.githubusercontent.com/Zayne-sprague/MuSR/b1f4d4168a9cfc6760e8b74d728e4516023dfaa5/README.md)；[MuSR dataset card](https://huggingface.co/datasets/TAUR-Lab/MuSR/raw/7c365b439a222150f317764d4f16ae6c96d7d94a/README.md) SHA `793154678fbec10387b116b4e4431cc05b1e44b49a79f030d7513a430c4d304e`；[ESCI README](https://raw.githubusercontent.com/amazon-science/esci-data/7916cdf6ab75a462e77f20ab40428a10923998d5/README.md) SHA `fe926f49adbb8dfd8dd7f2fbcd8cb30f6002267f17bc167f081386cc2e07aa64`；[ESCI NOTICE](https://raw.githubusercontent.com/amazon-science/esci-data/7916cdf6ab75a462e77f20ab40428a10923998d5/NOTICE) SHA `d4290ed64c2edd0fce1d84e3f9dfb2881240fe534def76b8cd29ed6af683e287`。

SimpleBench34只看了固定README/LICENSE元数据，README仅运行说明，未比这两项更好地说明数据作品来源，因此本批不提案它、不调查其余项。这不是认定SimpleBench无许可。

## 验证与边界

新helper严格核已批准baseSHA、324个manifest输入/实际SHA及精确每项文件数、固定publisher metadataSHA，exclusive-create输出3个Admission对象（evidence含完整绑定）。基础manifest未重复进Git；metadata无题/答案原文，未下载数据、未读private、未执行上游、未用GPU、未接受条款或申请账号/付费/远端manifest写入。远端只读既有文件字节算hash；cost=null。

CPU实跑提案成功；3项公共输出/错误输入SHA/错误publisher metadata测试通过，ruff check/format通过。数据准入不代表评分代码已运行、训练许可已批准或质量门通过。若root连同前批2项应用，16+2+3=21项；本报告不把待review等同已批准。
