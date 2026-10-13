# DocVQA官方访问与原数据条款实查

2026-10-14。用户已亲自注册、邮件激活并登录RRC，不能再标“账号未开通”。本次通过用户Chrome的官方[Download页](https://rrc.cvc.uab.es/?ch=17&com=downloads)核到了原数据条款和文件标签。

## 实际可读路径

Chrome扩展绑定报 `nodeRepl.fetch request failed`；原生Chrome能读窗口，但普通页面正文呈紫色空白。使用Chrome既有菜单 `View → Developer → View Source` 后，官方页源码在原生可访问性树完整可读。这是浏览器本身显示的已授权网页，不读取cookie、keychain、profile文件，不使用CDP，也没有绕过证书/浏览器安全警告。

## 下载会触发的协议

官方页明确：下载下面的图片数据与标注即表示同意其条款。这与账号注册/登录不同。页面条件概括如下：

- 不分发下载URL。
- CVC与IIIT Hyderabad不作侵权风险/适用性等保证。
- 使用者承担使用责任，并就相关主张为机构及其人员提供抗辩与赔偿。
- Single Page DocVQA图片来自UCSF Industry Documents Library；多数原文档版权仍属来源公司/信托，托管者没有这些文档的权利，只为非商用研究及教育托管。
- 随图标注没有额外限制；这不扩大原图片权利。

用户此前确认“DocVQA弄好了”能证明账号操作已完成，不能凭该句推断已经读过并接受这份新增dataset协议。cua_repl的Computer Use Confirmation Policy要求在接受法律协议时取得action-time确认，因此本轮没有点击下载；主会话需向本人具体展示上述条款后取得确认。即使同意研究下载，也不能把它记为商业图片训练或发布许可。

## 已看到的原文件与范围

Task 1为 **Single Page Document Visual Question Answering**，原V1.0标记2020-04-20发布，2023-08-31更新训练/验证问题类型。官方列出：

| 文件标签 | 本项目当前必要性 | 实际状态 |
|---|---|---|
| Annotations（questions, answers, question types） | 必需，只准备train/val | 看到入口，未触发下载；大小未知 |
| Images | 必需，只准备train/val | 看到入口，未触发下载；大小未知 |
| OCR（Microsoft OCR） | 不默认必需，待现有管线证明需要 | 看到入口，未下载 |
| IMDBs（MP-DocVQA framework预处理包） | 非当前默认依赖 | 看到入口，未下载 |

此页新格式把资源合包列出，不能假定旧版train/val单独包仍为当前入口。取得条款确认后，先核官方包清单/大小与split组成；单包超过5GB先向主会话报告预算。没有读取隐藏test答案，没有下载MP-DocVQA、DocCVQA或Infographic扩展任务。

具体下载URL不写本文/Git，账号信息不入报告。仅引用官方入口，必要的私密下载位置由浏览器会话处理。没有获得字节/哈希，不宣称DocVQA外部准备完成。机器可读证据见 `reports/docvqa-access.json`。AudioCaps已按用户明确省略及ADR0004保持不处理。

## 一次集中确认所需的Android协议版本证据

同时只读打开用户已有[NDK官方下载tab](https://developer.android.com/ndk/downloads)的Linux下载披露按钮。当前是r30，弹窗SDK License Agreement日期为 **2026-04-28**；首次观察时checkbox已显示勾选。agent没有点击checkbox或最终download，不把既有UI状态当成人工签约证据，也不下载r30。

项目固定LiteRT-LM v0.9.0所需工具链是NDK **r28b / `28.1.13356709`**。[官方SDK repository索引](https://dl.google.com/android/repository/repository2-3.xml)中该包引用 `uses-license=android-sdk-license`，但这个license元素的正文日期是 **2019-01-16**，正文16962字符，UTF-8 SHA256为 `1f8729233617b193fd619213792ae16a41b95d2bbbf525dfe66998252ba68b16`。

两者属于同名Android SDK License Agreement，不能说日期/文本完全相同。集中确认应明确覆盖实际r28b索引引用的2019版本；若选择当前网页办理路径，也应向本人列明2026修订版差异，不因r30页面已经勾选就宣称r28b接受手续完成。不要求用户为本地构建开付费/Play发布账号。协议涉及SDK用途限兼容Android应用、使用者代表权限/合法资格、OSS组件另按OSS许可及责任/赔偿等义务，完整条件以对应官方文本为准。

可集中询问：是否接受DocVQA-SP上列下载条款，仅供非商用研究隔离使用，并接受NDKr28b实际适用的SDK License Agreement（官方索引2019-01-16版；当前下载网页显示2026-04-28修订版，需一并明确是否接受）？两份协议同意均不代表DocVQA图片商业训练授权；本轮等待本人明确回复，未点击接受或下载。

正常DocVQA条款页面截图已在工具结果inline展示，未展示原文件download URL。截图API没有本地输出路径；OS区域截图/Chrome内置capture尝试未生成可找到的文件，因此不编造截图路径。截图保存问题不影响已核实的条款文字证据。

对annotations另核：页面虽然明确其无额外限制，但入口仍位于“下载即同意”总条款下；没有核实独立免协议公开入口，不能据此绕过当前action-time确认或猜测未列出的下载URL。

## 明确批准后的执行更新

用户通过主会话当前结构化询问明确回复“同意上述协议和来源调整”，覆盖SP非商用研究/教育、不分享下载URL、免责/赔偿，以及NDKr28b 2019 SDK协议及适用的2026修订。此前“等待action-time确认”是历史审查状态，现已解除，不再重复询问同一协议。Android执行归运行时代理，本代理不下载r30。

原SP官方Annotations为1,713,866B，Images为8,580,231,801B，合计8,581,945,667B，主会话批准总预算30GB；Mac空闲仅11GiB，因此通过已在本人Chrome官方页实际观察的合法URL直接远端缓存。URL只放0600私密临时记录，通过curl stdin配置输入，不入argv/Git/报告/日志，不读取cookie/profile或密码。两个官方原文件HEAD匿名200，并以ETag/If-Match固定本次服务器快照。

原站缺失GEANT TLS RSA1 intermediate导致远端初次verify20。按leaf证书公开AIA取官方CA intermediate后，分别在Mac和远端使用既有系统根执行issuer验证通过，leaf链+hostname验证通过；补齐已验证链后维持TLS验证，未使用insecure模式、未绕浏览器警告或加入未验证根。

Annotations已下载、exit0，SHA256 `546be9fd4ed6aff9d5a76a5120a71738a010833fa71dc919d5b8ead8e65ad85d`。zip元信息3成员，只有train/val被解析用于计数：train39463、val5349；test成员只记录文件元信息，未读取/解析其payload。Images在独立有界恢复任务中下载；partial不计通过，最终实际大小、SHA256及退出码以 `reports/docvqa-source-download-001.json` 为准。OCR/IMDB和扩展DocVQA任务未下载，图片不准入商业训练/发布。

原先截图API未给路径的限制后来通过Chrome内置Capture截图落盘得到解决。正常条款裁剪与SDK modal图现存忽略目录 `artifacts/docvqa-agreement-evidence/`，无私密下载URL或账号信息；不入Git。下载已从原单流切换为4×512MiB有界Range恢复，复用1,063,796,736B原partial；最终每段206/Content-Range/ETag、按序合并、原partial前缀一致性及全文件SHA256全部通过前仍不标Images完成。
