# coverage video500排序与DEV阈值拟合诊断 001

2026-10-16。只读同一coverage公开DEV500、原SHA绑定records，不看calibration/holdout、不GPU、不改预测/阈值或训练数据。完整绑定、源码SHA、耗时与分位数见同名JSON。

**时序残余更符合排序/判别不足，而非仅yes保守偏置。** 时序p_yes ROC AUC=.72526；在同一DEV标签上乐观拟合单阈值，最多218/329=66.26%，较当前207/329=62.92%多11题，仍错111。整体oracle上界350/500=70%，比当前337/500=67.4%多13题（2.6pp），仍错150；单调移动阈值不能消除大部分剩余错误。这是此固定模型/数据的观察，不区分排序不足究竟来自表示、训练覆盖、目标或标签噪声。

| 启发式（非官方task） | N | p_yes AUC | 当前argmax | 同DEV oracle上界 | 乐观gap | 上界后仍错 |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| all | 500 | .77459 | 337 / 67.40% | 350 / 70.00% | 13 / 2.60pp | 150 |
| existence | 129 | .82888 | 99 / 76.74% | 100 / 77.52% | 1 / .78pp | 29 |
| motion | 42 | .95791 | 31 / 73.81% | 40 / 95.24% | 9 / 21.43pp | 2 |
| temporal | 329 | .72526 | 207 / 62.92% | 218 / 66.26% | 11 / 3.34pp | 111 |

运动组高AUC且较大乐观gap，提示该小组的yes保守偏置可能贡献较多错误；但仅42例，且单独使用该组DEV gold拟合，本轮不能采用其阈值或声称泛化。三个分组各自拟合的上界不是全局单阈值结果，不能相加当成已部署策略。存在组oracle也几乎不改善。

FN/FP的p_yes分位数（线性插值q*(n-1)）：

| 范围 | 类型/数量 | p10 | p25 | p50 | p75 | p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | FN122 | .09773 | .18243 | .29421 | .37754 | .43782 |
| all | FP41 | .50000 | .56218 | .67918 | .79819 | .85195 |
| temporal | FN91 | .08510 | .18243 | .32082 | .37754 | .43782 |
| temporal | FP31 | .53121 | .56218 | .67918 | .77730 | .83548 |
| motion | FN10 | .19987 | .23426 | .32143 | .40733 | .43782 |
| motion | FP1 | .70579 | .70579 | .70579 | .70579 | .70579 |

FN/FP按当前yes/no argmax定义，分位数分处.5两侧本身不是额外因果证据；时序同时有偏低FN与偏高FP，结合AUC和阈值上界，说明仅放松yes阈值收益有限。未按单题选训练样本。

计算定义：AUC为正负样本两两concordance，分数tie给.5 credit；一类为空返回null。oracle仅遍历yes iff p_yes>=threshold的全部tie-preserving cut，含all-no/all-yes端点，不能拆同分样本；不导出或采用拟合阈值。当前argmax按canonical yes/no（平票yes），与保存prediction逐项一致。合法yes/no/U全集、target、family、data/sidecar/实际adapter SHA均复用原诊断helper严格验证。

输入边界仍为same data + recorded config；records没有prepared tensor/token input_sha或数值token IDs，不能声称逐tensor匹配。继续沿用screen词集合temporal优先、motion次、其余existence启发式，非官方分类。effective280字段是image_max_soft_tokens，未解释成视频每帧预算。没有重新推理或新阈值策略。

验证：实际只读CPU成功；10项公共数学测试（全tie AUC=.5、完全排序=1、反序=0、混合tie=.875、oracle两端点/不可拆tie、单类/无效输入）通过；ruff check/format与py_compile通过。SSH只通过stdin传入新脚本和原bound helper源码，wrapper对实际bytes计算两源码SHA，不写远端文件。未继续许可调查或扩大范围。
