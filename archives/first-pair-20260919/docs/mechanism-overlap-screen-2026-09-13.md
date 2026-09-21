# 集合外监督：定向查重与允许的验证范围

2026-09-13。本次只筛“训练时为索引器补充未选集合的教师比较，主注意力前向不扩大集合，推理删除辅助分支”。不是对全部稀疏注意力文献的穷尽审查。

## 当前判断

本轮检索与下列已核对的原文范围内，未确认与这一具体组合完全相同的方法；这不等于证明不存在重复。均匀抽查、采样归一化、预热和辅助梯度隔离都有成熟背景，不能将这些组件直接包装成原创贡献。可以进行低成本的本地机制诊断，尚不具备宣称新方法或投入大规模预训练的依据。

| 原文与阅读范围 | 对照结论 |
|---|---|
| [QSA 技术报告](https://arxiv.org/html/2608.30320v1)，§2.1.2，式12–20，原快照 `qsa.txt` 281–346行 | 压缩索引器、完整因果块、尾块、头间聚合与 max-pooling、仅已选集合 KL 是本参考实现的依据。全局层替换发生在 CPT。 |
| [DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)，§2.1.1，式3–4 | 先全量索引器预热，再仅在选中 token 集合监督；索引器输入 detach。它已覆盖两阶段训练与梯度隔离，不能据此声称我们的贡献。 |
| [MiniMax Sparse Attention](https://arxiv.org/html/2606.13392v1)，§3.2、附录B.2–B.4 | 已研究早期监督变化、预热与梯度来源；“训练初期索引器不稳定”并非新发现。当前只在这些已核对段落未发现本候选的集合外抽查分支，不能据此断言整篇及后续版本完全没有。 |
| [Adaptive Sampled Softmax with Kernel Based Sampling](https://proceedings.mlr.press/v80/blanc18a.html)，官方摘要 | 采样 softmax、偏差与采样分布选择是既有研究；抽样比较候选不是全新思想。此轮没有把它作为稀疏注意力的直接重复。 |
| [Sampled Estimators For Softmax Must Be Biased](https://papers.neurips.cc/paper_files/paper/2025/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html)，官方摘要 | 子集归一化的偏差有直接理论背景。尚未核对其定理的完整假设，不能照搬摘要来证明本目标的不可能性或无偏性。 |

QSA/MSA 原快照在 `literature/qsa-prefill-screen-2026-09-13/`。新增三个页面在 `literature/mechanism-screen-2026-09-13/`，含 URL、抓取起止时间和 SHA256。后两个页面是官方摘要页，不能称为已全文精读。

## 检索记录

通过网络检索以下关键词组合，并优先查 arXiv、OpenReview、官方论文页面：

- `"sparse attention" "indexer" "exploration" training`
- `"sparse attention" "unselected" supervision`
- `"QSA" attention "sampling" training indexer`
- `"sparse attention" "indexer" "random" "supervision"`
- `"sparse attention" "negative sampling" training`
- `"sparse attention" "selected" "sampling" indexer supervision`
- `"DeepSeek-V3.2" report arxiv`
- `"sampled softmax" "bias" Blanc Rendle`

搜索中出现的视频稀疏注意力、decode KV 缓存和二手自动摘要没有作为该具体机制的排重证据。NSA、MoBA、HiLS 的此前筛查结论仍见原可行性报告，本轮没有声称重新全文检查这些论文。

## 本地验证对应的主张

1. 验证掩码、梯度路径和归一化是否按所写算法实现。
2. 在共享参数索引器上检查：即使选中集合以外没有该 query 的直接 KL 梯度，共享参数是否仍能学习并恢复排序。
3. 检查集合外抽查是否提供超出原监督或短预热的明确诊断信号，并记录它多比较了多少候选。

当前参考实现不带完整语言模型主干，不是原生预训练复现。简单的“未选 score 梯度为零”检查只是计算图事实；不能上升为未选块永久无法恢复的结论，更不是一条新的论文定理。
