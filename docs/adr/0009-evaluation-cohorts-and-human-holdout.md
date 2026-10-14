# ADR-0009：评测cohort与物理模态分离，人工κ只作用于封存集

状态：技术合同澄清已获独立architect CLEAR；实现待验证。依据用户2026-10-14授权按原方案自主技术安排，不改变基座/引擎/准确率/性能门槛，不自动采购或扩大人工判断预算。

## 原文与问题

原HTML图6列六类：文本/JSON、截图/GUI、自然图像/图表文档、视频、音频、跨模态组合；第5节来源表与阶段2分别要求图像/截图，阶段3明确跨模态组合。通用约定和阶段0每模态500dev、1k人工封存；κ≥.7明确修饰封存人工一致性。configs/project-data-assets.json也已经列这六个评测类别。

当前data/schema.py的四种物理输入text/image/audio/video是正确输入契约；eval/sealed_gate.py却复用该四值作为评测cohort，漏掉截图/跨模态，同时强制dev也由两人双标并算κ。现有6000题/12000判断是4×(500dev+1000holdout)×2，不能当作原文六类封存覆盖。

独立审查支持上述解释，指出原文545/559/578/586为dev/holdout与κ边界，656为图像/截图独立门槛，683及图6/来源表为crossmodal独立类别。图6从原HTML原始dataURI解码，未修改，artifacts/s0-cohort-contract/original-data-mix-figure6.png可复核。

## 决定

- 物理输入Metadata.modality仍四值；评测cohort另为text/image/screenshot/audio/video/crossmodal。既有gate wire字段modality可保留兼容命名，但文档明确其含义为评测cohort，不改变模型输入或推理。
- real gate要求六cohort各500dev与1000holdout。合计3000dev+6000holdout；不得跨cohort重复算同一item。来源/family跨train/dev/calibration/holdout隔离不变。
- 只holdout要求两位真实独立人工的原始标签、逐cohort非退化Cohenκ≥.7；不得以裁决标签、模型或程序真值冒充双人流。默认阈值.7，real manifest不得降低。
- dev另交合法硬金标/程序真实结果的元数据ledger：item/cohort/family/source哈希、来源/revision/许可和标签来源。验证500题数量、完整来源与hash绑定、重复/越界/跨池family；不要求dev人工双标或κ。不读取封存payload，不把dev gold包装成人类判断。
- 原人工预算仍为12000原始判断，现对应6000holdout×2；dev3000使用原始合法金标/程序真值。没有批准新增人员、付款或采购。若真实source/人员无法提供仍blocked，不伪造。
- 历史四cohort canary/报告保留，只能证明旧工程边界；不得自动升级为六cohort real pass。新合同显式版本化；原root开发环境依旧不能自证独立隔离，real输出不得因本改动自动passed。

## 实现与验证边界

新增dev ledger类型和CLI输入；holdout的双人/哈希/κ/拒绝载荷错误边界保持。canary兼容旧synthetic小例，但输出明确canary与合同版本。回归必须拒绝旧四cohort real、缺screenshot/crossmodal、低κ门槛、缺dev ledger/来源、重复item跨cohort、dev/holdout family重叠、holdout少标签/重复标注者；接受合法六cohort dev来源不带人工标签/κ。错误仍仅类别，不返回标签/题目/路径或可恢复payload。

本ADR只澄清验收合同。独立gate身份、真实人标、500dev数据构建、六模型基线与GPU/浏览器门槛仍未完成。s0:12在来源与可执行构建票冻结前不关闭，05/06仍外部条件。
