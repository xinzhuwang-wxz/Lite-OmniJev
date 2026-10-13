# STATUS

更新时间：2026-10-13 18:52（Asia/Shanghai）
当前里程碑：M0已完成；进入P0，等待用户提供持久无卡Linux实例。
原生goal：blocked（D001缺少无卡实例/SSH），终局目标不变；尚未交付真实模型/API/端侧产品，不能complete。

## 当前票与所有者
- bootstrap:01–08：全部done；各票验收与证据已记录，无活跃执行者。
- p0:01：ready-for-human / blocked；用户提供无卡实例，阻塞p0:02。
- p0:02：ready-for-agent / open；依赖p0:01，完成无卡环境/真实权重和数据准备。
- p0:03：ready-for-human / blocked；只在p0:02完成后通知用户开卡。
- p0:04：ready-for-agent / open；依赖p0:03，执行真实100样本10步到API集成。
- delivery:01：blocked；P0及S0–S6出口未完成，后续继续追加阶段集成票依赖。

## 阻塞项
- 尚无租用无卡Linux实例/SSH入口；本机macOS arm64，无CUDA。
- 本地只下载E4B7个metadata文件（32,226,275字节），完整权重未下载；真实训练0步。
- 当前smoke入口无优化器/RNG恢复，P0票已要求长训练前补齐；adapter warm start不能冒充resume。
- S0前须处理教师格式/来源准入、精确benchmark子集、人工封存集与私有榜单；证据见research报告。

## 待我决策
- D001（现在）：用户租无卡Linux x86_64实例，提供SSH连接别名/主机端口与持久工作目录；先不要开GPU。
  建议8vCPU/32GB RAM；E4B起步持久盘至少200GB，包含两教师/视频数据建议500GB以上，按实际manifest复核。
  原因：租用/付费与账号是用户保留操作；代码、锁文件、资产工具和数据/训练入口已准备。
- D002（后续）：S4放量/预算、任何门槛或产品范围改变、生产发布再提交决策；不提前默认授权。

## 下一步
1. 用户给出实例后核对架构、网络、持久磁盘；传送本地代码快照，无需push。
2. 认领p0:02；按docs/runbooks/compute-handoff.md及assets.md下载/校验真实资产，锁Linux训练/服务环境，完善恢复和预检。
3. no-gpu-readiness全通过后通知用户开卡；随后P0真实闭环，S0逐批出规格/拆票/triage。
4. 无新实例/用户信息时当前frontier为空；不要重复下载、重跑已通过测试或制造票空转；保留这个阻塞交接。

## 最终本地证据
- reports/bootstrap-validation.txt/json：111 tests passed，无跳过；Ruff/format、strict mypy20文件、wheel/sdist、CPU smoke、Chromium、queue、git diff全部exit0。
- reports/browser-smoke.json/png：真实Chromium操作Swagger health200、未配置推理503、零page error；不是最终Playground验收。
- reports/e4b-token-probe.json：33答案单token/无冲突/thinking关闭；reports/real-template-validation.json：400题实际生成边界通过。
- reports/train-preflight.json：100样本max77tokens已编译；完整权重/CUDA缺失，expectedblocked/exit1/0训练步。
- reports/m0-review.md：6项实质问题先复现后修复，独立定向复核通过。

## 资源与Git
- origin=https://github.com/xinzhuwang-wxz/Lite-OmniJev；仅本地提交，未push、未创建远端issue/PR、未轮询gh。
- 本轮子代理全部结束；所有临时worktree已移除；验收服务器与浏览器均已退出。
- 保留artifacts模型metadata、manifest和100样本缓存供续用；不删除用户原有资源。
- 接续入口：plan/START.md；交接：plan/handoff-2026-10-13-bootstrap.md。

## Goal阻塞复核
- 时间：2026-10-13T18:55:40.066810+08:00。
- 连续阻塞轮次：3（含用户发起轮结束时的D001）；上一轮无开发进展，本轮复核同一外部阻塞，不声称完成。
- 本轮证据：queue valid、frontier=[]、claimed=[]；仅主worktree；无新的实例/SSH信息，无须等待的已知活跃作业。
- native update_goal 已返回blocked。停止自动空转；收到实际无卡实例信息后恢复P0，先不启用GPU。
- 本地M0验收保持有效；不重复下载/验证，不降低终局门槛，不擅自租用或付费。

## 租机配置决策补充
- 用户正在选择实例，尚未提供平台/SSH。已核对镜像、CUDA驱动要求与三档配置，详见docs/runbooks/compute-handoff.md的2026-10-13补充。
- 推荐H10080GB整卡路线：正常开卡16–24vCPU/128GB RAM，系统盘100GB、数据盘500GB可扩至1TB；只做CPU准备且支持独立规格时8vCPU/32GB。
- AutoDL特殊无卡模式按官方文档仅0.5核/2GB，不能当作普通8核32GB CPU实例；GPU会释放，不保证恢复时有库存。
- 未租用、未开卡、未更换模型或阈值；原goal仍因实例接入缺失保持blocked。

## AutoDL选型进行中
- 用户已指定AutoDL并授权agent选择配置，用户负责付款（按量/包日均可）；未授权自动开始计费。
- 创建页实际跳转登录，当前无法核实账户可见库存/报价。已打开独立headed Chromium供用户登录，不读取或保存密码/验证码。
- 旧浏览器工具session=90810已关闭；当前活跃session=72292，脚本artifacts/rental/browser_session.py；收到登录回复后先snapshot核实，再继续选择，不能重新开浏览器丢失登录状态。
- 暂定按量计费，单张H10080GB优先、正常开卡CPU≥16vCPU/RAM128GB、数据盘500GB可扩容；实际主机和镜像待登录后核验，未声称已选中或有库存。
- 尚未点击租用/创建/付款，未产生实例；最终选择停在提交前交给用户。

## AutoDL配置已选（尚未提交）
- 登录已完成；实际选择北京A区700机/bce24a9cd7，H20-NVLink96GB×1，16核/150GB RAM，系统盘30GB，数据盘50+450=500GB，按量10.50元/小时+数据盘3.15元/日。
- 镜像：PyTorch2.12.1 / Python3.12(Ubuntu22.04) / CUDA13.0；项目另建Python3.11环境。实际记录见reports/autodl-selection.json。
- 库存扫描时A800专区无空卡，多台RTX PRO6000与H800空闲机数据盘容量不足，未选择这些不满足存储条件的主机。H20实际训练吞吐仍待验证。
- 当前账户需充值；agent未提交创建/付款。页面按钮“创建并开机”会开始正常GPU计费；若用户要无卡，需创建后按平台控制台关机→无卡模式开机。开关卡由用户决定。
- 活跃浏览器session72292保留选中配置供用户确认付款；选中不代表预留，提交前需看实时库存。
