# 端侧契约核对与三档包模板

结论：**当前能准备/校验包布局与运行时依赖，不能交付已验收端侧模型。Android手机不是文档指定的唯一终局平台。** 原文§7.2将Android/iOS的首选运行时定为llama.cpp，LiteRT-LM是验证后的加速选项；没有要求首发同时支持两OS、Android APK/Gradle、Google Play账号、iOS签名或必须通过模拟器。已有Android CLI/NDK缓存是选定准备路径，不会自动增加这些门槛。

## 原文与终局约束

| 来源 | 实际约束/边界 |
|---|---|
| HTML§1、§6阶段6、§7.2 | E2B蒸馏后合并决策LoRA；GGUF Q4_K_M，质量不够可实测Q5_K_M/Q8_0；校准参数随模型一起发。文本/图像/视频帧需实测，端侧音频路径及限制须写进兼容矩阵。 |
| HTML§6、configs/gates.json | 量化掉分≤1pp、ECE≤.035，E2B各模态≥E4B92%；质量不能靠CPU构建或清单通过替代。Q4内存≤3GB、旗舰手机单题<500ms属于原工程估值，ADR0010允许在代表性实测后有据修订，当前没有新实测就不能编新数值或自动放行。 |
| HTML§6/§7.1/§7.2 | Ollama/llama.cpp/vLLM各一条命令；单卡包括vLLM+gateway镜像、Ollama Modelfile/GGUF读出、Mac llama.cpp路径。LiteRT能否取logit、E2B视觉及音频兼容调查是必做；不等于第四个运行时必须成为已支持加速后端。 |
| HTML§9.2/发布清单、delivery spec | 干净机器一条命令、服务器/GGUF/端侧同200题差异、模型与校准/网关同版本及整体回滚。读取完整legal logits，U独立、答案去U归一化；截断/采样top-k不准冒充原始概率。 |
| delivery:01、docs/02-project.md | 必须有实际支持设备的内存/时延/离线/模态证据；还需S0–S6、浏览器等完整出口。平台支持范围/端侧音频限制不能靠模板标pending或beta静默免验。 |

HTML允许音频不支持时明确写明；较新的项目契约要求范围改变记录兼容矩阵并作范围决策。因此本片不自行删端侧音频，不将旧文fallback自动当范围授权。

## 现有机器可执行的真实路径

| 路径 | 当前能做什么 | 仍缺什么、不能替代什么 |
|---|---|---|
| 租机Linux x86_64 / llama.cpp v0.6.0 | 已有四CPU二进制及SHA/version证据；E2B成品GGUF+媒体projector到位后，可在CPU真实离线推理/完整logit读出、同200题对照、记录该Linux设备RSS/时延。真正入口在包根`bin/llama-cli`。 | 决策GGUF/媒体产物未生成，完整legal-logit与模态adapter未验收。Linux测量不证明手机500ms或移动端3GB。 |
| 本机Mac arm64 / llama.cpp v0.6.0 | 实际arm64，clang与cmake路径可用；可以从已有锁定源码做原生CPU/Metal构建与干净目录包重放，再对真实产物做桌面离线/200题及资源测量。 | 这轮没有原生构建或模型执行；不能把工具存在称为端侧通过。移动端ABI/能耗/内存不同。不能把本机未知版本Ollama直接替换锁定v0.35.0。 |
| 租机Ollama v0.35.0 | 已有官方Linux包/版本/hash缓存；可以准备Modelfile与版本绑定目录，产物到位后实跑one-command及概率/媒体兼容。 | 缓存二进制不等于完整选项raw logits可用。未启动daemon/推理，不可用生成top-k绕过概率门。 |
| 租机LiteRT-LM v0.9.0 host | 官方Linux CLI及private Noble loader已做加载/help/no-model边界证据；可复用它做明确.litertlm产物的CPU可行性验证。 | 成品转换、完整logits与视觉/音频实测pending；不是Android性能证明，也不应在未验证前称为可发布后端。 |
| 已缓存Android arm64 CLI/NDK r28b | 现成LiteRT CLI、NDK版本/ABI及hello编译证据可复用；也可为锁定llama.cpp准备Android原生CLI包，无需先造APK。 | 真正Android运行需可用目标设备/adb授权/窗口；购买/租赁仍需用户决定。已有协议已批准，不重问NDK授权。iOS若列入支持矩阵，则须单独真实iOS目标证据。 |

Mac本身是实际硬件，能证明Mac支持项；租机也是实际Linux硬件，能证明Linux支持项。它们不能证明“旗舰手机”指标/Android或iOS支持。最终端侧目标应由现有产品兼容矩阵明确冻结；本片既不加“唯一Android真机”门，也不擅自把移动端承诺缩成桌面版。模拟器/ABI识别/CPU编译不能冒充真实目标设备内存/时延。

## 本片实际CPU闭环

没有现成发布模型打包器；`scripts.gold_media_bundle`和`s0_development_bundle`负责数据/媒体，不能拿来装私有数据或冒充模型包。本片复用已有`release-readiness/check_inventory.py`及底层`assets.verify_file`，新增三档目录/manifest/binding模板，没有新增生产代码/依赖。

模板入口：`.scratch/delivery/bundle-template/README.md`。三档都有真实README和bindings模板SHA；其它必需文件保持pending，不填伪hash/大小。bindings中的release、decision weights、template/token-map、calibration、runtime/config、设备、200题与回滚字段未冻结则为null。

实际命令/退出码：

- `check_inventory.py --manifest <api>/manifest.json --root <api>`：exit1，2 verified模板、8 pending、0 invalid。
- `check_inventory.py --manifest <single-card>/manifest.json --root <single-card>`：exit1，2 verified模板、10 pending、0 invalid。
- `check_inventory.py --manifest <edge>/manifest.json --root <edge>`：exit1，2 verified模板、10 pending、0 invalid。
- 标准库copytree→按manifest建目录→tar.gz打包/读回：exit0，37成员、三个template_only manifest、无symlink/模型文件；临时副本和归档已清理。SHA/大小见`bundle-template-cpu-001.json`。
- 本机`uname -m`及`command -v clang/cmake/ollama`均exit0，只记录工具存在，不称运行时验收。

`inventory_incomplete`是正确结果；即使将来全文件完整，checker仍仅证明完整性，不自动签发S4/S5/S6/FINAL。复用当前已验证的校验器，没有改其行为，不为目录文案写镜像测试。原release-readiness001是历史快照：本会话已补S5 scalar core，但真实fit/gateway、合格模型/量化/蒸馏/真机/三档运行和终局仍pending。

保守租期到明日09:15，03:15开始发布缓冲；模板不会把剩余模型/端侧真实验收变成已完成，也不依赖公开发布来关票。无GPU、训练、模型推理、服务启动、联网下载、付费或公开发布。
