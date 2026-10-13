# 开卡前最后的来源与协议决策（已确认）

依据：docs/research/stage3-remaining-media.md、reports/stage3-remaining-media.json、docs/research/docvqa-access.md、docs/runbooks/required-authorizations.md。用户2026-10-14明确同意，已批准决策见 docs/adr/0005-permitted-first-release-media.md；下载与验收继续，不代表已完成。

## 具体执行方案

1. 首发商用训练使用许可明确的来源。阶段3的1.2万视频问答由已下载的CC0 CLEVRER train提供（152,572题/10,000原视频族）；原LLaVA 0.5万题改由允许的教师在CC0/BY媒体上生成，属于后续派生产物，保留5000题数量及family隔离，不将同题重复计数。NExT-QA/VidOR、VGGSound、STAR/Charades、LLaVA受限原媒体不作为首发商用训练前置；未来扩来源须逐媒体许可。
2. DocVQA原图官方仅限非商用研究/教育，接受具体SP下载条款后仅在隔离研究/评测池缓存。商用文档训练采用自有程序化文档截图及其他许可明确文档；阶段2原图表/文档1万题不减少，不重复计入原合成页面1万题。未通过真实文档开发集门槛仍不得宣称能力达标。
3. 不改音频/视频/文档质量门槛，不把CLEVRER合成场景代表全部现实视频。来源多样性下降是明确风险，阶段3仍用真实公开音视频评测；不达标则再补许可明确的现实媒体，而非降低门槛。

## 需要确认的具体协议

- DocVQA SP：下载即接受，不分享下载链接；原图版权归原权利人，官方只为非商用研究/教育托管；CVC/IIIT免责，使用者承担责任及条款中的赔偿义务。annotations与原图权利分开，不能靠annotation允许使用扩大原图许可。
- Android SDK/NDK：为本项目下载/使用NDK r28b（28.1.13356709）。r28b官方repository所引用android-sdk-license结尾日期2019-01-16；当前官网下载弹窗为2026-04-28修订，两个版本不能说完全相同。接受覆盖该工具链使用的具体SDK协议后才下载安装；不租卡、不付费、不下载无用r30。

协议正式文本以官方页面及工具审计记录为准。浏览器工具要求在真正接受法律协议的动作前具体确认，先前“自己申请”不替代这次具体条款确认。账号密码已由用户本人设置，agent不索要或回显。
