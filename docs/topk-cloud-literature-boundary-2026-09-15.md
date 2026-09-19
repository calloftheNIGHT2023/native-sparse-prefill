# 本轮工程与前人工作的边界

本轮实现用来得到可信的计算成本基线，不作为已确认的原创贡献。当前确认原创贡献仍为 0。

1. Gupta 等，Memory-efficient Transformers via Top-k Attention，SustaiNLP 2021，§2.2 及局限性：已经讨论 query chunking、保留紧凑 top-k 信息、重计算反向以及二次候选评分。我们实现这些步骤、改善 PyTorch 调用开销，不能据此宣称一种全新注意力方法。核查的是上述章节，不是逐页完整精读。原文：https://aclanthology.org/2021.sustainlp-1.5/
2. SAS，arXiv:2609.13141v1，§4 的 gate/kernel 描述、§5.1–5.3、§6 的效率分析：已有可训练排序、融合 gated attention、联合 continued pretraining 的作者实验，并指出长上下文的 selector scoring / top-k 开销。我们没有复现其质量或速度；本轮 exact top-8 的数学规则和其 block gate 不同，但“发现选点贵”和“把选点相关步骤融合”不足以单独证明选题原创。这里只核查了相关段落，不声称完成全文复现。原文：https://arxiv.org/html/2609.13141v1
3. CUDA Graphs 减少 CPU 发射开销是标准工程工具。我们对完整与稀疏两侧均使用 graph，不能把只有稀疏一侧 graph 的结果作为跨方法加速。实现依据 PyTorch 2.10 的 side-stream warmup、静态地址与动态输入重放规则；云端实际运行版本另记为 2.4.1+cu124。文档：https://docs.pytorch.org/docs/2.10/notes/cuda.html
4. Triton 的 tiled attention 与 atomic accumulation 均为已有编程机制。本轮 selected-edge kernel、max/argmax 排序、流式保留少量最大值，是为了检查系统开销，不因自行写了内核就自动成为新算法。文档：https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html ，https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html

前两篇原文及 PyTorch 文档原件保存在 `literature/topk-efficiency-cloud-2026-09-15/`，URL、字节数和 SHA256 见其中 `sources.json`。此记录是具体工程边界检查，不是“没有任何相似论文”的保证。
