# 单一程序gold时间顺序pilot：CPU数据就绪

真实CPU准备28.262秒，exit0；基于最新已含第三coverage实际更新的exposure012，全库存隔离结果isolated。**只创建一个数据集，未选择父模型、未启动GPU或训练；training_authorized=false、parent_model=null。**

产物远端目录：`/root/autodl-tmp/s1-text-lora-001/artifacts/s4-temporal-order-contrast-pilot-001`。同名JSON保存完整哈希、来源和聚合运行结果；原题/annotation/派生样本/ledger不进Git。

- 固定namespace `42:s4-order-pilot-001:`，从已准入8299 TRAIN中的原temporal词规则family按hash排序，最多64；实际检查31，1拒绝，取得前30安全family。没有按模型错题选、没有不安全补齐。
- 每family一个安全collision事件对：三属性描述全场景唯一、每object-pair全视频恰好一次、参与者在各自事件帧inside_camera_view、事件帧间距≥16。实际所选gap16–104；同pair重复或任意三属性歧义的整个场景拒绝。
- 每个事件顺序独立求`less(frameA,frameB)`及反序，验证一yes一no；两条自然语言输入只用color/material/shape，不含object_id或frame答案。原postfix仅做来源绑定，未冒称复执行；不是现有no题反转。
- 60video来自30family，按seedhash反转两题位置；实际首位置20yes/10no。每步3family对=6video（3yes/3no），加原课程完全相同的2cross+4old；总120唯一draw、10steps/accum12/seed42/16帧/1120image budget，pureCE且原学习率/5步midway保留，不启用pair/Brier/KL。
- 30个原媒体SHA实际逐一验证，原TRAIN sample/source question/program/annotation场景绑定验证；派生ledger记录同family/source/media/train、annotation-memberSHA、事件身份、规范程序/求值schema及canonical sampleSHA。

事件标注可见性与gap只提供16均匀帧的可见机会。未解码验证采样帧是否捕获碰撞/有无遮挡，未宣称模型收益或视觉证据充分。CC0及ADR0005既有来源范围保持，无新增来源/许可接受/下载。

## 绑定与可用命令

本轮审计前后未观察到更新作业；只读评测不阻断CPU准备。exposure012 report SHA `db21553a169282aa4eb440e8fe79b5021e9ac6d37357e90bb27dc8be53f85fdf`，base inventory SHA `039de536c088249f1bf7abc5b811d46c7adc95b0fe7844ce37cea14c53113a24`。准备前后复核两者不变；若后续再发生更新，应在真正训练前刷新inventory绑定。

数据SHA `f16fbf93029461cf520522f4037ff829284a7a4a7c531d1ecd4b9e0b7bbe8ef1`；配置SHA `abfab49cecf920db76e60d6b0314ddb3d721ddcd6e79183cebac33e0baf9beda`；ledgerSHA `f8c8245bfee85b81c8658b54d57cd63008f99e3428f7c5dd5f47b4e2e5d314dd`；draw顺序canonicalSHA `afc086b6cad393bf76af0fe8d9059fcbd77aa9d174754a8123d67f94903ec91e`。全库存auditSHA `d8d317971f53bae42f639e49ff2e5d474bffb0f88693ece7080b33bdddeda2db`。

实际执行命令（输出exclusive-create，原产物已存在，禁止原地重跑覆盖）：

```sh
cd /root/autodl-tmp/s4-motion-code-142e744
CUDA_VISIBLE_DEVICES= PYTHONPATH=/root/autodl-tmp/s4-motion-code-142e744 \
/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python \
/root/autodl-tmp/s4-temporal-order-prepare-owned-001/prepare-temporal-order-contrast.py \
--root /root/autodl-tmp/s1-text-lora-001 \
--official /root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files \
--output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-temporal-order-contrast-pilot-001 \
--exposure-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-012 \
--exposure-report-sha256 db21553a169282aa4eb440e8fe79b5021e9ac6d37357e90bb27dc8be53f85fdf
```

复用相邻`inspect-temporal-order-contrast.py`和`prepare-claim-pair-pilot.py`的来源扫描、语义检查、canonical ledger/cohort、逆shuffle、写入与既有inventory工具，不改共享trainer或默认流程。派生失败保存receipt；若安全family不足30只报告不足，不生成凑数配置。现有产物不覆盖。

验证：8项CPU公共语义/拒绝/确定性测试通过、ruff check/format通过；实际smoke_samples/ledger校验/全库存隔离通过。未进行token/processor模型preflight、训练或GPU质量验收；这仍须父模型选定后由root调度。cost=null。
