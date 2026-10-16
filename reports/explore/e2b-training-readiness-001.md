# S6 E2B：真实 CPU 就绪，gold 起步可运行，跨模型蒸馏仍待生成

2026-10-16。实际远端CPU预检005完成23.005秒、exit0；CUDA未初始化、未加载模型tensor到GPU、未下载。对应JSON保存原远端receipt SHA、完整资产/数据/config绑定和启动命令。当前结论只覆盖E2B训练准备，不能宣称E2B质量、端侧音频、量化或S6验收通过。

## 已完成的真实准备

- E2B驻留 `google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7`；9文件共10,278,849,571 bytes，全部根据现有上游LFS SHA256或Git blob SHA1真实重核。权重10,246,621,918 bytes、SHA `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`。有效参数规模不等于驻留大小。
- 原project inventory为bytes/lfs_sha256字段，训练器使用size/upstream_sha256；从已绑定原inventory无损转换生成E2B runtime manifest。`.gitattributes`不进运行时白名单，但单独核验，没有删除原资产。manifest SHA `dc92da2ba05816627461f0789ff285f33f7add43ced42e0cbc6769990acf5edd`。
- 官方tokenizer/template实测33个答案（A–U、0–9、yes/no）均单token、唯一、边界稳定；显式thinking关闭成立，template SHA `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`。
- 用真实E2B config在meta device实例化Gemma4ForConditionalGeneration，35 decoder层；205个decoder projection的真实驻留safetensors header形状与模型逐个吻合。PEFT rank16/alpha32实际附着后410个可训LoRA tensor、24,158,208参数；仅decoder，塔及主干冻结。未加载任何E4B adapter，未做随机小模型替代。
- 对现有TRAIN课程六个来源各一例实跑官方processor，涵盖text/image/screenshot/audio/video/image+audio；实际长度169–1923 token，视频16帧，image1120预算；所有输入/张量摘要绑定。不是全1200题预处理、更不是GPU前向验证。
- 原核心只允许E4B，已独立修复 `324cf59`：精确允许E4B/E2B两个repo，仍要求40位revision、逐文件hash、完整权重与Gemma4架构。73项CPU训练/资产行为测试、ruff/mypy通过；准备器另4项配置测试通过。未泛化任意基座。

首次001因把project inventory直接当runtime manifest而拒绝，002因既有E2B runtime manifest不存在而拒绝；失败报告保留。003/004成功后追加来源receipt和真实权重projection形状核验，最终005为权威本次结果。

## 最小实际课程：先 gold，不把它称蒸馏

复用 `artifacts/s4-program-stratified-pool-001/train.jsonl`，不新抽样、不读取cal/holdout。源receipt SHA `ac2f17fc3f5417e93e04d15292763384858f5e3ba1c4de533ca6f5e57d0aaf99`；data SHA `795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f`；ledger SHA `b5172781520546ef974a57825fa8ce98033b20e6d8a66daf8c1f550efc186322`。source config、组件ledger都与原receipt核对，1200/1200声明TRAIN且数据版本契约通过。

实际1200draw：video600、audio100、photo100、screenshot100、text100、crossmodal200；1010不同family（cross复用原组件，不能称1200独立样本）。**1200目标全部hard，没有本轮新增U-positive监督或教师概率。** 这是现有多模态gold的低开销起步与管线验证，不足以代表原方案完整E2B训练覆盖；U质量和各模态须实测。

生成的草案位于远端 `artifacts/s6-e2b-training-readiness-005/gold-bootstrap-config.json`，SHA `0e2243cd9cda1395597a1540c1c76aa88ea33467793b71f30897e86907b4baf9`。固定seed42/rank16/alpha32/LR2e-5/BF16/accum12/100steps/1200draw/16frames/image1120，CE，无Brier/KL/pair；每5步checkpoint、lazy CPU准备。E2B从官方IT基座新建LoRA，不能从e465或f2db续训。采用已验证课程与保守学习率以优先省卡，不预断它足够训练一个新学生。

## 最小启动与连续恢复

完整逐字命令在JSON的commands字段。只在root完成当前GPU排程、源归档/课程身份核对后执行；本代理未启动。使用已部署代码目录 `/root/autodl-tmp/s6-e2b-readiness-code-001`，其核心来自324cf59。该目录无Git元数据，训练报告code_revision会是unavailable；root launcher必须把Git提交与JSON中源码归档SHA写入独立启动receipt，不能把unavailable冒充版本绑定。最终可复现源码归档 `/root/autodl-tmp/s6-e2b-readiness-final-001.tar`。

```sh
cd /root/autodl-tmp/s6-e2b-readiness-code-001
# 新shell必须使用真实GPU可见性；CPU检查脚本不会替你切换或启动训练。
timeout --signal=INT --kill-after=20s 220s \
 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python -m scripts.train_s1 run \
 --config /root/autodl-tmp/s1-text-lora-001/artifacts/s6-e2b-training-readiness-005/gold-bootstrap-config.json \
 --model-dir /root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E2B-it/3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
 --asset-manifest /root/autodl-tmp/s1-text-lora-001/artifacts/s6-e2b-training-readiness-005/e2b-runtime-manifest.json \
 --data /root/autodl-tmp/s1-text-lora-001/artifacts/s4-program-stratified-pool-001/train.jsonl \
 --allocation-ledger /root/autodl-tmp/s1-text-lora-001/artifacts/s4-program-stratified-pool-001/allocation-ledger.jsonl \
 --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s6-e2b-gold-bootstrap-001 \
 --invocation-steps 10 \
 --report /root/autodl-tmp/s1-text-lora-001/reports/s6-e2b-gold-step0010.json
```

后续保持所有参数与global config原样，增加`--resume-from .../s6-e2b-gold-bootstrap-001/checkpoints/step-0010`，report换step0020；依次到最新checkpoint，最多global100步，不加入`--init-adapter`。每段均先核实上一段正常paused/completed与checkpoint绑定，禁止timeout后盲目重启。先取得E2B自己的固定dev基线，再first10/20/40/60/80/100趋势监控；不能继承E4B成绩，也不把10步小样本未涨当最终能力结论。

第一段240秒总硬上限（220INT+20kill），真实GPU前向、梯度、loss、峰值与每步时延都pending。E4B同类10步约175–191秒仅用于粗预算：E2B100步暂估约30–40分钟，加评估另计；不得按2B/4B比例声称实测速率。first10实测后更新预算，若处理/加载耗尽预算就保留失败和checkpoint，先诊断，不加购/续租。每次续段前有聚合监控、身份/非有限/OOM失败即停；完整S6质量在真实各cohort验证，不用训练loss证明。

## 教师软分布蒸馏：明确的下一准备缺口

原HTML阶段6要求阶段4池20万样本、E4B输出分布蒸馏E2B；原质量要求各模态≥E4B92%、量化掉分≤1pp/ECE≤.035、Q4≤3GB。当前1200gold bootstrap不等于20万蒸馏完成，ADR0023实测局限推进也不允许假称该原门通过。校准/封存只能独立gate汇总。

教师只能是实际e465 adapter SHA `e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd`配其E4B pinned基座前向，**当前未生成任何教师logits**。最省卡顺序：先在相同TRAIN课程前120draw采集一次完整合法K+U logits/概率，绑定教师base+adapter、原样本/媒体、官方template、token map与数值精度；CPU核验概率归一、gold冲突率及joint U后再补齐剩余1080draw并缓存，不反复在线teacher-student双模型。既有开发集/校准输出不得挪用作训练标签。

学生损失候选保持gold优先：0.8 CE(gold,q_student)+0.2 CE(q_teacher_joint,q_student)，两项同K+U空间；硬gold分量确保组合靶gold至少.8，任何其他项至多.2。真实insufficient gold应同样保留，不拿低置信或teacher错题造U。它是普通蒸馏候选，尚未实现/检验收益；教师来源和学生输入hash各自绑定，E4B与E2B token/processor不能未经验证假定一致。

**当前接口有具体阻断：**`SoftTarget`仅容纳非U且强制归一、U隐式0，不能用它承载教师联合分布；`load_reference`又限定300draw并把reference绑定为学生init_adapter/asset_hashes，不能把E4B参考冒充E2B初始化。需要独立的cross-model teacher sidecar/目标身份实现及测试，保留原限制，而不是删校验或丢p(U)。本次按授权只修E2B资产准入，未改蒸馏核心；root可先真实gold训练，蒸馏入口由后续明确所有者处理。

不用新模型下载、不租卡、不推HF，不改变主干/分类头/推理引擎。当前可启动的是明确标识的gold bootstrap；教师软标签、跨模型蒸馏和最终E2B质量仍pending。
