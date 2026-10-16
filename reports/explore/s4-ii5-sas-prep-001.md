# s4-ii5-sas-prep-001 — II-5 SAS 子决策辅助监督包:CPU 构建与 240s 三臂规格

状态:**CPU 构建完成并全量校验;GPU 三臂待 main 排队(D3P 之后)。**
定位(用户指示:做前人没有的、项目落地的颠覆创新):训练期模态子题辅助分解 + 推理仍单前向单 token——
检索零直接先例(前沿 002 报告 II-5);本切片用现有 TRAIN AND 池落地,不依赖 II-2 合成基建。

## 数据包(已实跑构建+校验)

- 源:TRAIN AND 组合池 `s4-crossmodal-pairs-001`(600 AND 行=200 组×3 变体,sha
  933db691…;groups c334c42a…、components c1404f8b…,已钉)。
- 构造(`.scratch/s4/build_ii5_sas_package.py`):每 AND 题 → 保留主行 + image 子题
  (choice,原题选项,gold=组级**真值** component_gold.image,非变体声明值)+ audio 子题
  (noul,gold=component_gold.audio);子题媒体只挂对应单模态;data_version +`+sas-sub-v1`;
  全部留在 train 池同 crossmodal family(隔离不变)。
- **校验实跑**:600/600 指令模板解析成功;600/600 AND gold 与组件真值一致性
  (claim 双真 ⟺ AND=yes)复核通过;1800 行全部 pydantic 通过、id 唯一;ledger {item_id,pool=train}。
- 产物(已上传远端 s4-ensemble-oracle-001/):
  - `ii5-sas-package.jsonl` 1800 行(600主+600图+600音,sha d569a1a5…)+ ledger;
  - `ii5-mains-only.jsonl` 600 行(臂1 对照)+ ledger;
  - `ii5-sas-package-w025.jsonl` 782 行(600主+182子,辅助占比 0.233≈权重0.3,确定性 sha%20<3)+ ledger。

## 240s 三臂规格(同 launch-b1-wsft-pilot 命令模式,init=e465 step0100,seed 42,ckpt 5)

| 臂 | 数据 | 行数 | steps |
|---|---|---|---|
| 1 CE-only 对照 | ii5-mains-only.jsonl | 600 | 40 |
| 2 SAS 全量 | ii5-sas-package.jsonl | 1800 | 60 |
| 3 SAS+权重(~0.3) | ii5-sas-package-w025.jsonl | 782 | 50 |

config 克隆 D3P 学生 config,仅改 sample_count/steps/media_roots(加 image/audio 两个既有
media root:s4-crossmodal-pairs-001/media 在池目录内自含 `media/` 子目录,已核)+ expected_data_version=
`crossmodal-controlled-train-pairs-v1(+sas-sub-v1)`;modality=**mixed**(含 image/audio 行)。

## 预注册门(失败即停留证)

- **过门**:crossmodal 固定 DEV50 ≥ 既有基线 **+2 题** 且 video150 不回退;
- 附判(报告 II-5):ECE 改善≥30% 也可过;放弃线=辅助稀释主任务(主任务掉 >1 题 且无增益)→ 臂2/3 停。
- 风险预案:子题措辞偏置(模板已随机化于池构造期,本包不改写措辞);梯度竞争由臂3 的低权重版直接检验。
