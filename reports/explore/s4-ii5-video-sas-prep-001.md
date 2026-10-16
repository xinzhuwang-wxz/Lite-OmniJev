# s4-ii5-video-sas-prep-001 — 视频版 SAS:CLEVRER 程序切面监督包(零先例组合)

状态:**CPU 构建完成、三臂数据上传远端、自测通过;GPU 待 main(优先级:D3P > crossmodal-SAS > 本包,或按窗口并行)。**

## 范式定位(用户指令:全维度创新打 video 19.8pt 差距)

前人把 CLEVRER 程序当**推理时执行器**(NS-DR/CRCG);本切片把同一程序真值当
**训练期切面监督源**:每 TRAIN 视图生成程序可判定切面子题(count/color/enter-order/
start-end/collision/motion-direction/speed),与原主问题混训,推理仍单前向单 token。
检索零先例(前沿 002 II-5 的视频泛化);不造不可判定 U(ADR0024 合规:只用程序可判定切面)。

## 数据包(实跑生成,全 train 池,同 family 隔离)

- 生成器 `.scratch/s4/build_ii5_video_sas.py`(复用 B1 的 scene_truth 解析:对象/出入帧/碰撞,
  扩展 count/color/方向/速度切面;速度方向取轨迹平均速度符号,平局跳过)。
- 选择:视频按 `sha256("ii5v:"+scene)` 序取前 400;每视频从其切面 bank 按
  `sha256("ii5v:pick:"+facet_id)` 确定性取 1 题 → 400 切面。
- 切面分布(400):motion_direction 102、color 64、start_end_stay 56、start_end_entry 47、
  enter_order 31、collision_absent 25、collision_exist 23、count(choice)22、motion_speed 24、
  start_end_exit 6;gold 偏 yes(279/99 + 22 字母)——已知偏置,附判里由对照臂隔离。
- 三臂数据(已上传远端 s4-ensemble-oracle-001/,ledger {item_id,pool=train} 全过,pydantic 全过,
  **统一 data_version=clevrer-train-video-noul-s3-pools-v1+ii5v-v1**,trainer 单版本校验可过):
  | 臂 | 文件 | 行数 | sha256 |
  |---|---|---|---|
  | 1 mains-only 对照 | ii5v-mains.jsonl | 400 | bb48a280a995374c35a885f3738512bea5ecd7b93e047e8a4826c3321bb7c3fb |
  | 2 混切面(占比0.5) | ii5v-mixed.jsonl | 800 | a8e65288d519492fa6b8e904547ba9ade77d61f9759a2c36aa0d855e49898c58 |
  | 3 切面加权(占比0.226≈0.3权重) | ii5v-arm3.jsonl | 517 | e302fad1fbf554346a03689e0bb8aa5b71e292b1950713e79f9f44369da60bc0 |

## 240s 三臂规格(init=e465 step0100,seed 42,ckpt 5,lr 2e-6)

命令=launch-b1-wsft-pilot 模式,--data/--allocation-ledger 换本包文件+ledger;
config:mains 400/40 步、mixed 800/60 步、arm3 517/50 步,modality=video,
media_roots=[s3-media-candidates video],expected_data_version=
`clevrer-train-video-noul-s3-pools-v1+ii5v-v1`(三臂同版本,已统一)。

## 预注册门(失败即停留证)

- **过门**:video150 ≥121/150,**或** 关系(enter_order+collision)+运动(direction/speed)+
  start-end 子项合计 ≥**+5 题**,且 DEV50 不回退(>1 题掉即停);
- 对照判定:臂2/3 vs 臂1 同数据族同种子,只有切面行差异;臂2 过而臂3 不过 → 高占比辅助有效但权重敏感;
- 放弃线:臂2/3 相对臂1 video150 差 ≤0(辅助稀释主任务)→ 视频切面路线停,证据保留。
