# 程序gold事件顺序pilot：择父与单次预算

原rawcoverage第三段在更广video150退化108vs116，已经停止追加。首段f2db与第二段250ef首次完整视频同协议分别354/500与356/500；两题差异下首段NLL/Brier/ECE/coverage及原cross概率保留更好，选择f2db为新pilot唯一初始化，保留两个原checkpoint。full1ECE .061116/Brier .381444/coverage .164，相比full2 .100680/.399016/.128。均非S4正式通过。

新dataf16fbf93…为30TRAIN视频、每family一个安全collision-event pair的两条先后关系题，60次独立程序求值；同20cross/40old回放、同CE/10step/120draw/accum12/seed42/16帧。不同于旧no反转或单纯新增family，监督来自官方标注事件时间；parent/账本/媒体源隔离绑定，源部件不含封存。原CPU parentnull/training_authorized=false报告永久保留，root单独launch显式授权本次f2db初始化。

预算preflight→10step240秒（第5步checkpoint）→固定原temporalDEV50单遍180秒/25题progress，session56377；没有后续自动长串。比较parent33/50及概率loss/纠正退化；只有可解释额外收益才扩原video150再做保留检查。无收益停止这条具体关系数据/预算配置，不泛化为全部程序监督无效。TRAIN碰撞帧的可见性和间隔验证不等于实际uniform16捕获事件，因此质量必须实测。

SwanLab只真实step指标，未知费用null；不是RL、不是新增分类头、不是SOTA证明。有限动作expected-reward原型另在CPU数学检查，不与本次数据变量混合。S5温度代码和三档发布清单在CPU并行准备，门槛不提前放行；用户24h剩余租期末至少留6h发布缓冲。
