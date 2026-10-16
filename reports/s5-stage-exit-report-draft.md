# S5 阶段出口报告(草稿 2026-10-17 06:55,基于真实完成件)

## 已完成并验收
1. **真实校准**:fit007(e465)+fit009(D3P-150)各 18000 题独立校准,七组温度全 NLL 改善,UID/边界/磁盘全程留证(fit003-008 失败链保留)
2. **封存评估**:holdout 6000 题×2 版(e465:s5-holdout-eval-003;D3P:s5-holdout-eval-004),私有边界(UID 隔离/ACL/凭证拒绝)全程通过
3. **原门实测(封存,D3P 候选校准后)**:
   - 准确率下降 ≤.5pp:**六模态全零损失 PASS**
   - ECE ≤.03:**4/6 PASS**(text .0013/screenshot .0086/image .0192/audio .0293);video .0316/crossmodal .0469 MISS→ADR0023 披露
   - 封存/开发差(同源族):video 2.3pp ≤3 PASS
4. **服务绑定**:registry manifest(7 fitted/11 unfitted)+runtime_recipe(bf16/transformers/native_lora)+SHA 全链
5. **跨引擎边界验证**:vLLM 0.31 native 服务激活尝试被 loader 按设计正确拒绝(S5 出口 4 实测通过)

## 未完成(pending,不虚标)
- 网关 live bootstrap 最终一步:transformers-native serving 路径未实现(当前唯一服务化路径为 vLLM 代理,其校准需 vLLM 引擎自身校准作业)
- Playground 真实浏览器验收(前端未建)
- 置换≤.02/正反≤.05/OOD(C1-C4 完整封存版):材料部分就绪未跑
- R1 完整对照(6 H100·h×3 种子):GRPO-lattice CPU 闭式已完成,采样式按"不假称 RL"规则降级

## 结论
S5 出口 1-2(校准+封存评估)**在 e465 与 D3P-150 两候选上均达成**;出口 3 部分达成(ECE 4/6,置换/正反待跑);出口 4 服务绑定达成且边界实测;出口 5(网关/浏览器)pending。按 ADR0023,未达项单项披露,不静默改阈值。S6/E2B 未启动。
