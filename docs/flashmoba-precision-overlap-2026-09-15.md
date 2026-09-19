# 路由精度方向的重合核查

- **P01 — Switch Transformer §2.4, Table2**：[原文v3](https://arxiv.org/html/2101.03961v3)已经用路由局部FP32改善BF16稀疏专家训练稳定性。它研究MoE而非块注意力，但足以否定“只把路由升成FP32”是新的通用机制。这里未复现其训练数字。
- **P02 — uncertainty-gated-block-sparse-attention，README机制和budget-match说明**：[作者公开仓库](https://github.com/ThomasRossi/uncertainty-gated-block-sparse-attention)按选块截断处的分数间隔给不确定query tile扩大预算，且已有预算匹配对照。不能把“低margin多算块”当全新提法。本次仅核查作者公开说明，未独立确认效果或审稿状态。
- **P03 — SpotAttention**：[原文摘要](https://arxiv.org/abs/2606.22874)研究冻结主干上的选择器训练、预算选择和低精度选择器存储。低精度存储不等同于本次块均值累加误差；此处只记邻近工作，不据此声称本实验已被完整覆盖。
- **P04 — 搜索结果源冲突排除**：第三方搜索结果将arXiv2510.19875关联到“Certified Logit-Faithful Sparse Tracing”，但[arXiv官方记录](https://arxiv.org/abs/2510.19875)实际标题为“Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention”。不采用该第三方匹配作查重依据，也没有核实那份证书论文的真实性。

决策：FP32池化首先视为数值控制，不视为原创贡献。利用已知模型做有限必要性检查；若真实文本影响不足，则暂停该方向的训练扩张。没有检索到完全相同的实验，不意味着已证明无人研究过，也不构成新颖性保证。

来源版本、下载UTC及SHA见 literature/flashmoba-precision-screen-2026-09-15/sources.json。阅读范围为上述章节、摘要及仓库说明，并非全领域全文精读。
