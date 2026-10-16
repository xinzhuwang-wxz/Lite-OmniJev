# 标量温度 CPU 准备

当前只交付拟合/导出/严格加载核心，不代表进入S5。S4质量未通过；真实拟合、可信私有作业导出/准入、网关加载、ECE/coverage/封存/R1与正式发布全部pending。不得让开发agent读取真实calibration/holdout逐题数据。

## 极短规格

原HTML阶段5要求按**模态×原语**拟合温度。每个产物固定一个`cohort`（text/image/screenshot/audio/video/crossmodal）和`primitive`（choice/score/noul），各组单独拟合；不是把六模态混为一个标量。

基线目标为完整合法答案+U上的gold joint NLL（hard truth或明确U）。T固定范围[0.05,20]，在inverse temperature上做80步确定性golden-section搜索，并比较两端点与T=1；不恶化拟合集NLL。平坦目标选T=1。使用标准库，无新优化器依赖/LoRA更新。保存seed用于来源绑定，算法不使用随机性。范围与目标是此CPU基线算法约束，不改S5质量门。

应用调用既有`train.readout.read_probabilities`：同一T缩放完整legal logits，U保留joint概率，答案去U后条件归一化。joint NLL包含U，不能用conditional答案NLL替代。正温度保持答案argmax，但不保证ECE、coverage或封存表现；本入口不拟合弃权阈值、不做Brier训练、一致性/R1/保序或自适应网络。

## 输入及绑定

`python -m scripts.fit_calibration fit --input <explicit.json> --output <new-temperature.json>`

输入JSON结构（以下只是字段说明，不是真实校准数据）：

- `schema_version`: 1，`pool`: `calibration`。
- `identity`: `base_revision`和`code_revision`（完整40位Git revision）；`adapter_weights_sha256`、`official_template_sha256`、`answer_token_map_sha256`、`raw_logits_receipt_sha256`（完整64位SHA256）；`cohort`、`primitive`、非负整数`seed`。
- `rows`: 非空列表；每行必须明确`pool: calibration`、`logits_kind: raw_logits`、`complete_legal_logits: true`、`options`、`logits`、`target`。`logits`键精确等于options加U，值为有限数；target是其中一项。Choice为2–20选项，token为A开始的连续字母；Score为2–10，token为0开始的连续等级；Noul使用有序`["yes","no"]`。合法token/映射由受信任生产者与token-map摘要绑定。

不接受train/dev/holdout/unknown、缺U/缺选项/额外token、重复/单选项、NaN/Inf/boolean logit、采样top-k或logprobs声明、未知字段。温度缩放数值跨度溢出时拒绝。

CLI对输入**原始文件字节**计算`calibration_data_sha256`并写入artifact binding；空白改动也形成不同数据hash。artifact另自动记录实际`train/calibration.py`文件的`fitter_implementation_sha256`，不以调用者声明替代拟合器源码指纹；identity中的`code_revision`绑定输入生产者版本。artifact还包含方法、冻结范围、样本数、拟合前后NLL与`stage_acceptance: not_evaluated`。目标路径必须不存在，禁止覆盖既有产物。失败只打印异常类别，避免泄漏私有字段/答案。

数值无法证明其确实来自raw logits或calibration池。receipt SHA与字段严格绑定只是完整性/身份契约，**不等于来源准入**。正式S5必须由root后续扩展的固定可信私有作业核真实输入/池隔离、模型/template/token-map和receipt，再生产此输入；不能把调用者自声明当私有gate准入。本CLI不发现目录、不自动读取任何private文件。

## 严格加载与重放

`python -m scripts.fit_calibration verify --artifact <temperature.json> --expected-binding <trusted-binding.json>`

`expected-binding`必须由调用方受信任配置提供，包括上面identity的所有字段及`calibration_data_sha256`，不可从待验证artifact自己抄出后宣称验证了生产身份。任何字段失配、未知字段、温度越界或修改冻结范围均拒绝。外部bundle还须保存artifact文件SHA防止参数被篡改；本核心不提供签名/远端认证。

Python入口：`fit_temperature(CalibrationInput, data_sha256)`、`save_temperature(artifact,path)`、`load_temperature(path,expected_binding)`、`apply_temperature(artifact,expected_binding,logits,options)`。应用仍先比较绑定，随后复用readout。verify仅证明文件结构/参数约束与预期绑定吻合，不证明质量门。

## 已实跑验证

```sh
PYTHONPATH=. python -m pytest tests/test_calibration.py tests/test_core.py -q
ruff check train/calibration.py scripts/fit_calibration.py tests/test_calibration.py
ruff format --check train/calibration.py scripts/fit_calibration.py tests/test_calibration.py
mypy train/calibration.py scripts/fit_calibration.py
```

测试只用明确合成logits，包含实际CLI fit/export/verify、T=1身份、joint/conditional/U重放、NLL改善、所有绑定字段失配、非calibration池/坏logits拒绝。测试中的artifact不作为真实模型校准结果或阶段证据。
