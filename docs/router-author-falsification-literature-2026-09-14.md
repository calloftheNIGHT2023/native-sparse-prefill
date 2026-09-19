# 当前候选的相关工作边界

检索与原文核查：2026-09-14 UTC。证据原件及 SHA 见 literature/router-author-falsification-2026-09-14/sources.json。本记录不是完整领域综述，也不等于复现这些论文。

1. **SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space**，arXiv:2511.20102v1，§3.2、§3.3、Algorithm 1。作者讨论硬选择排除位置的梯度缺失，采用完整/稀疏主分支随机切换及逐层双向输出对齐。与“稀疏学习有缺口，借完整注意力补梯度”这一宽泛角度明显重合。其块选择、规模和任务不同，不能称为我们这个 MQAR 设置的逐项重复；但不能把宽泛机制或双流修复当新贡献。原文：https://arxiv.org/html/2511.20102v1

2. **The emergence of sparse attention: impact of data distribution and benefits of repetition**，arXiv:2505.17863v2，§2.2–2.4、§3.1–3.2、Appendix D。已有平台期、突然学会、数据/优化器改变学习时间的理论和联想回忆实验。该文的“稀疏”主要指完整注意力学到的集中模式，而非我们从第一步施加的 hard top-8。因而它不直接否定两种训练的差距，却使“等很久后突然形成回忆能力”本身不足以构成新贡献。原文：https://arxiv.org/html/2505.17863v2

3. **Transformers Learn Faster with Semantic Focus**，arXiv:2506.14095v2。最初 HTML/带版本PDF访问失败；随后从不带版本的原始PDF地址取得61页全文并提取文本（原件与SHA保留），核查§4（PDF p.7）及§5.2/Theorem5（p.17–18），不是只依据摘要。实验为 ListOps 和七个 NNCH 分类任务，含 MLP，并非本次无MLP的因果MQAR。该定理假设选中/未选中分数有正间隔、每个key被查询的次数受限等；因此不能据其标题推导所有top-k训练都更快，本次有限设置失败也不推翻它的条件定理。通用收敛/泛化角度已有大量实质工作。原文：https://arxiv.org/pdf/2506.14095 ，官方收录：https://proceedings.neurips.cc/paper_files/paper/2025/hash/3035bafea7fdf0ddee585a906dde6a82-Abstract-Conference.html

我们自己的判定：先完成未改方法的 19→64 轮续训，以排除过早截断；若普通训练即可成功，必须撤回“需要新修复机制”的当前依据。若失败，也只是保留一个待解释的有限预算现象，不能用失败次数代替新理论或因果证据。

4. **ZETA: Leveraging Z-order Curves for Efficient Top-k Attention**，arXiv:2501.14577v1，§3.4、§4.1。原文已有 top-k 排除位置梯度中断的讨论，以及追加历史均值向量的处理；也报告 MQAR。因此，“用便宜摘要覆盖被删位置”也不是空白。ZETA 同时改变相似度、维度和检索算法，不能把它当作本机 exact-QK 控制的同一算子。原文：https://arxiv.org/html/2501.14577v1
