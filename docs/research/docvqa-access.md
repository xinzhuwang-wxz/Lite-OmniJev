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
