# HF 模型卡社区惯例(三样例实证:Qwen2-VL-7B / llava-hf-onevision / UNA-TheBeagle)

## YAML front-matter(Qwen/LLaVA 惯例)
pipeline_tag: image-text-to-text; library_name: peft(adapter 发布必写); license;
base_model: google/gemma-4-E4B-it(adapter 发布必写,UNA 反例:只写散文被视作缺失);
tags(多模态/decision-making/conversational); datasets; metrics 可选; arxiv 可选。

## 章节顺序(Qwen 黄金模板)
1. 标题+简介要点(Key enhancements bullets)
2. 架构说明(可配 1 张架构图——社区图表惯例:**架构图用图,数据用表,不放指标柱状图**)
3. **评测表**:竞品模型做**列**,benchmark 做**行**,每行最优**加粗**,缺数据用"—";
   引用他人数字须同协议或标注协议差异
4. Requirements(安装注意一条)
5. Quickstart(可跑的加载代码;adapter 发布必须含 PeftModel.from_pretrained 模式)
6. Usage tips(输入格式/参数控制)
7. Limitations(编号列表,如实)
8. Citation(BibTeX)

## Adapter-only 发布特有(综合三样例+HF peft 惯例)
- base_model front-matter 必写;PeftModel 加载代码必须;adapter 文件体积说明
- 可选:merged 权重版另行发布(社区两种都常见)
- HF Hub 会自动渲染 "Use this model" 块/model tree/safetensors 面板——无需自己做

## 诚实惯例(UNA/Qwen 共同点)
- 评测注明运行方式("Ran with vLLM,数字可能与榜有差")
- Limitations 明说做不到什么(编号清单)
- 与榜单/他人数字协议不同时明确不可直接比

## 对我们的映射
- 竞品列 = 冻结基座(同尺寸)+ 教师 27B(标注"参考线,尺寸不同");内部迭代名(D3P-150/600)不出现在公开卡
- 我们加分项(社区少见,差异化):校准 ECE 列 + 封存协议说明 + 全哈希复现链
- 可选 1 张架构图:冻结基座+LoRA+符号校验级联的示意(产品形态 Jev 式输出)
