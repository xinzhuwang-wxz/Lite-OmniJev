# 第二段 coverage full500：偏置缓解，时序判别仍是主要残差

250ef 在同一公开 DEV500 为356/500，对历史0deef的337/500净增19：47纠正、28退化。NLL .610662→.585696，Brier .422930→.399016。比较对象不是首段f2db；不能将这19题归因于第二段单独训练。cost=null，不新增GPU。

两份终态report与records SHA已核；同500 source SHA、顺序、ID、family、gold、model revision、BF16评测路径、16帧、effective280 image budget和库版本。两份report的evaluator SHA分别匹配本地历史/当前源码，`forward_readout` AST完全相同（其间新增进度输出）。**双方历史records都没有prepared input/tensor hash，不能声称tensor exact**；原始source已同SHA且媒体引用相同，不重解码补证。源码版本不同及未留存实际processor tensor是比较边界。

| 启发式组（非官方task） | N | 正确数旧→新 | 纠正/退化 | NLL变化 | Brier变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| temporal | 329 | 207→224 | 31/14 | -.01600 | -.02284 |
| motion | 42 | 31→41 | 10/0 | -.26608 | -.22010 |
| existence | 129 | 99→91 | 6/14 | +.03066 | +.03723 |

极性确实改变：gold yes265/no235，yes预测184→249；yes正确143→185，但no正确194→171。temporal false negative91→62同时false positive31→43。existence false positive9→20，解释该组主要退化；运动则修复全部10个旧false negative、仅保留1个false positive。因此不能仅看总体+19或yes预测更接近gold比例就认定全面改善。

但纯极性解释也不充分。temporal p_yes AUC仅.72526→.72872；新模型用同DEV gold乐观拟合单调阈值也只有226/329，对224多2题，仍错103。当前temporal错105，占全部144错的72.9%，其中91题两个模型都错。整体新模型oracle356等于当前argmax，旧模型oracle350已被超过6题；motion AUC .95791→.97577，existence AUC .82888→.77886，体现跨题型排序取舍，非一个全局阈值能修复。

Oracle只是同DEV标签的乐观诊断上界：不导出/采用阈值，不把三个分组oracle相加，不作为calibration或泛化证据。时序组按before/after/when/while词规则定义；这提示时序关系判别/覆盖仍是重点，不能单凭分组证明视频时间顺序表征是因果瓶颈。

下一实验的判别重点：支持root已准备的matched polarity-control，固定同data/init/预算，用受控相反claim观察是否只移动yes倾向，还是提升temporal排序并减少双向错误。保留existence-no作为退化哨点；只改变整体yes数量而temporal AUC/概率loss不改善，不应视为机制修复。运动已41/42，继续扩大该组小目标的信息收益有限。本轮不建议DEV阈值调整，也不预排第三段。

## 固定进度节点的历史配对

原始source顺序，提前固定25/100/250/500，全部列出，不挑有利前缀。概率loss为同一批题的conditional gold NLL及去U Brier；各节点样本累计。

| 前缀 | 正确数旧→新 | NLL旧→新 | Brier旧→新 |
| ---: | ---: | ---: | ---: |
| 25 | 18→18 | .53744→.64463 | .35812→.44590 |
| 100 | 68→67 | .63675→.64921 | .44240→.45283 |
| 250 | 165→178 | .62548→.58798 | .43704→.40540 |
| 500 | 337→356 | .61066→.58570 | .42293→.39902 |

前100略差而全500更好，说明早期同题差异适合发现停顿/灾难退化，不能事后把任一短前缀当作可靠晋级/停止阈值。未来在真实进度节点读取累计同题正确数及概率loss可支持预算判断；本报告不拟合自动停止规则。

机器证据见同名JSON：绑定source、adapter、两report与sidecar SHA，完整分组confusion/oracle/固定前缀。raw500仅暂存本地/tmp用于CPU分析，未进Git。复用已测paired/ranking math；新增3项公共比较边界测试通过（源码格式不影响AST、forward改动拒绝、source错位拒绝），ruff通过。无private读取、模型修改或重测full。
