# 粗筛摘要与位置编码：追加查重

本页只记录新增核查，不改变已冻结的算子门槛或择优计时。未启动相关新方法训练。

[Prism: Spectral-Aware Block-Sparse Attention](https://arxiv.org/html/2602.08426v2)（2026-05-25 v2）§3.2–3.4 已明确分析 RoPE 与均值池化的作用：块平均会衰减部分高频分量，并以频带分解及能量温度校准改进块选择。这使“均值摘要消掉位置信息 + 频率补偿”的完整问题—解释—修复链条直接重合。文中 §4 的质量与速度是作者报告，本项目未复现，不能搬成自己的结果。

[Hierarchical Global Attention](https://arxiv.org/html/2606.30709v1)（2026-06-29 v1）已有带 RoPE 信息的层次摘要，以及 chunk → group → exact-token 路由。其主要场景还涉及分层存储和推理；不能因此推论任意原生训练改进都已被做过，但“层次粗筛 + 最后算真实 token”本身不再是可申报的新机制。

与已核查的 LongLoRA、PBS-Attn、Boundary Repair 合看，本轮排除三种宽泛命题：移动分块、给固定边界补连接、以 RoPE 频带纠正均值摘要。排除的是这些提法的原创性，不代表所有动态稀疏训练问题已被解决。

完整页面与 SHA/下载 UTC 分别落在 `literature/flashmoba-baseline-2026-09-15`、`literature/flashmoba-rope-screen-2026-09-15`。这是针对相关章节的原文核查，未对全部引文逐篇精读，也不保证穷尽所有文献。
