# ScreenSpot原版无卡准备补漏

2026-10-14审计。原始设计 `docs/Lite-OmniJev设计方案-最终版.html:506` 将ScreenSpot与JevBench、Decision Index、MMAU、MVBench并列为评测集。既有全项目原料表漏记这一项；本次独立补充 `configs/screenspot-assets.json`，不改设计门槛。

## 来源与版本

[原作者SeeClick](https://github.com/njucckevin/SeeClick/blob/0ef37ac4d7aaf37ba7b990e3e3c3ca77e1fb8f93/readme.md)明确给出ScreenSpot截图和标注的[NJU共享目录](https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/)及[Google Drive备份](https://drive.google.com/drive/folders/1FuFT05yXOV_QxhwYft85YTLOgaIYm_fS?usp=sharing)。使用该原版，不替换为ScreenSpot-Pro、ScreenSpot-v2或HF重打包版。

[NJU公开目录API](https://box.nju.edu.cn/api/v2.1/share-links/5b8892c1901c4dbeb715/dirents/?path=/)显示4文件，修改日期均为2024-01-22；两官方共享目录匿名HTTP200。作者仓库revision为 `0ef37ac4d7aaf37ba7b990e3e3c3ca77e1fb8f93`；共享文件本身无Git不可变revision，因此下载快照以逐文件SHA256固定，不能把仓库commit冒充媒体版本。

| 原版文件 | 原目录bytes |
|---|---:|
| `screenspot_imgs.tar` | 597,412,864 |
| `screenspot_desktop.json` | 58,673 |
| `screenspot_mobile.json` | 90,636 |
| `screenspot_web.json` | 79,224 |
| 合计 | 597,641,397 |

官方目录没有公布可直接比较的SHA256。本轮逐文件实际下载、核对目录尺寸，再记录本地SHA256；该证据不能称为“官方SHA256匹配”。额外保留固定版本作者README和LICENSE后总597,662,567B，低于2GB预算。远端隔离目录为 `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/screenspot-original/nju-20240122`；结果见 `reports/screenspot-source-audit.json`。

## 许可与访问边界

作者仓库[LICENSE](https://github.com/njucckevin/SeeClick/blob/0ef37ac4d7aaf37ba7b990e3e3c3ca77e1fb8f93/LICENSE)为Apache-2.0，但README License段明确具体dataset/checkpoint保留原许可。原目录仅4数据文件，没有单独数据LICENSE；作者明确公开邀请下载原评测数据，本轮据此做原料审计/研究评测隔离缓存，没有新账号、点击接受条款或申请表。

不能将仓库Apache或[HF转换版card](https://huggingface.co/datasets/bevaya/ScreenSpot)的Apache标签扩展为所有第三方截图素材的商业再分发许可；本次不准入训练或商业媒体发布。HF `rootsautomation/ScreenSpot` API现重定向至bevaya版，且其card说明bbox坐标格式已转换，因此没有用其parquet代替原作者文件。若未来发布截图/进行训练，仍需完成具体权利范围审核。

本次不解析标注JSON、不看答案、不运行模型预测，未读取封存内容。原版public评测与封存集仍需样本族去重隔离；原任务为GUI grounding，转为项目Choice候选的协议与匹配ID必须另行冻结，不能用下载完成代替模型评测达标。

## Decision Index是否已覆盖

固定[Decision Index原版registry](https://github.com/apolinario/decision-index/blob/f8552a13b643d39f61185f5951bb5fed16bedf51/decision_index/data/benchmarks.json)共42,564B、SHA256 `e7c9ef019a0f2ecc3269cbc3897a07a524550629d74cade64b9567c6c130db3c`，不含ScreenSpot引用。远端 `decision-index-public0.3-retry-002.json` 亦不含其引用，整个对应原料目录的文件名扫描无ScreenSpot命中。因此没有既有下载清单覆盖证据，需单独准备；这不是截图内容级穷举去重结论。

## 验证

6个文件curl退出码均0、4原数据尺寸与官方目录一致、SHA256已入报告、总量小于2GB。远端CPU重任务锁被已有GUI/CRC任务占用，取消本任务尚未开始的等待进程，将同SHA256的597,412,864B临时tar副本复制至macOS，逐图完整解码；仅使用tar成员头与图像字节，不提取、不解析标注。远端原料+本地临时副本共1,195,075,431B，仍低于2GB；未做Linux端媒体解码。结果详见报告 `cpu_media_probe`。tar共613成员，610图像全部header校验与完整像素解码通过、0不安全成员，退出码0；不能从截图数推导题目ID完整性。使用macOS Python3.10.18/Pillow11.3.0独立媒体工具，不作为项目Python3.11应用/模型验收。未做GPU推理或质量评分。未新增需要本人登录的当前下载阻塞，商业发布权利与后续评测协议保持待审计。
