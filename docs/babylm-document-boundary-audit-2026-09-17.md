# BabyLM 2026 Strict-small 文档边界与划分审计

2026-09-17。范围：已下载的六份固定版本训练文件、官方历史预处理代码、2026 数据卡和 dev 元数据。没有下载 dev/test 文本，没有训练 tokenizer 或模型，没有启动云卡。本文是来源/数据结构审计，不是模型实验结果。

## 结论与对当前设计的影响

1. **不能把每一行当成完整文档，也不能由单行很短推出整个语料没有长上下文。** Gutenberg、CHILDES、SimpleWiki 仍保留明确的上游记录起始标记；标记之间存在很长的候选片段。
2. **目前也不能把两个标记之间的所有文字认证为同一完整、连续的原文。** 历史官方流水线按固定行数抽样、直接拼接保留块，不另写采样边界；2026 又做了句子过滤，但当前版本没有附精确行映射。
3. 可以开始逐行保留、带稳定记录 ID 的 tokenizer/词 token 账目和短上下文工程验证；科学主训练的文档拼接策略仍需冻结。不得把无来源证明的 8K/32K 拼接当成长依赖证据。
4. 最有价值的下一步不是换数据或立即训练，而是从 Gutenberg 的 PG ID 和 CHILDES 会话路径中恢复一小批**确实连续的已保留训练片段**。不能在恢复过程中补进原始 10M 文件以外的训练文字。

## 固定来源

- 训练数据：`BabyLM-community/BabyLM-2026-Strict-Small`，revision **c92ab16b4f08858304b0815706065b3354d8fc0a**。六文件本地 SHA 与 `data/babylm-2026-strict-small-raw-v0/manifest.json` 对应；本次复算的 SHA 见 `literature/babylm-boundary-audit-2026-09-17/raw-boundary-markers.json`。[固定数据卡](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small/blob/c92ab16b4f08858304b0815706065b3354d8fc0a/README.md)
- 官方历史预处理：`babylm/babylm_data_preprocessing`，commit **848e98f8d031154c14eeeae568748143c6b03af8**，commit 日期 **2025-02-24T15:00:54Z**。这是历史源码证据，**不是已证明与 2026 训练文件完全对应的可重放流水线**。[固定源码树](https://github.com/babylm/babylm_data_preprocessing/tree/848e98f8d031154c14eeeae568748143c6b03af8)
- 官方 dev 仓库元数据：`BabyLM-community/BabyLM-dev`，revision **169f42e32d0aaf65ec6b91d55bafad27a3afc729**，目录有六个 `.dev` 文件，无 README；未读取这些文本。尝试 README 返回 404，此失败是仓库没有该文件，不是下载了测试数据。
- 所有获取到的源码和数据卡、URL/哈希收于 `literature/babylm-boundary-audit-2026-09-17/manifest.json`。边界统计可用同目录 `reproduce_boundary_statistics.py` 重跑。

## 当前六文件实际保留了什么

所有训练文件都没有空白行。以下标记计数基于整行匹配 `= = = ... = = =`，没有用模型推断文档边界。

|源|总行数|起始标记数|已知标记含义 / 当前限制|
|---|---:|---:|---|
|BNC spoken|65,220|0|历史脚本一行一个 XML `s` 单元，文档间原有双换行；2026 文件没有保留空行，当前不可恢复所有会话边界|
|CHILDES|531,305|1,565|标记含 `.cha` 会话文件路径；会话内的清理后话语分多行|
|Gutenberg|59,934|59|标记含 PG 书号；书的段落经过换行合并，书不等于单行|
|OpenSubtitles|386,046|0|当前不能定位所有影片/对话边界；官方 README 明确历史预处理由他人提供，未公开完整流水线|
|SimpleWiki|58,709|14,800|标记含文章标题，正文可能分多行|
|Switchboard|2,892|0|历史转换逐行处理对话标签；当前没有经验证的完整对话分隔字段|

上述行数相加 **1,104,106**。标记的上游语义直接来自固定代码：`preprocess_childes.py` 的 `incorporate_metadata`，`preprocess_gutenberg_child.py` 的书号前缀，`preprocess_simple_wiki.py` 的 `<doc>`/标题处理，`preprocess_bnc.py` 的 XML `s` 提取。样例和行号已保存，报告不复制正文。

按“一个标记之后到下一个标记之前”计算的只是**候选片段长度**：

|源|候选片段数|词数中位数|最长词数|至少 2,048 词的片段数|
|---|---:|---:|---:|---:|
|CHILDES|1,565|1,334|11,134|516|
|Gutenberg|59|38,865|140,223|52|
|SimpleWiki|14,800|54|7,789|10|

这里不计标记行自身；CHILDES 第一个标记前有 840 行，Gutenberg 有 328 行，归属未知。末尾片段可能截断。**该表不是完整文档长度分布，也不能用来直接估计真实长程依赖数量。** 它证明有值得进一步追踪的长片段，避免把“单行最长 1,033 词”误作最大文档长度。

## 上游如何抽样和划分，能推出什么

固定 `sample_chunks_and_split.py` 按输入顺序读取行：`--split_at` 为整数时，累计指定的**行数**形成一个块，随后把整块随机分配至 train/dev/test 或丢弃。脚本虽计算 `chunksize`，实际结束条件使用 `counter == int(args.split_at)`，不能把该参数解释成词数。输出仅 `writelines(chunk)`，没有新增块 ID 或断点标记。[固定采样源码](https://github.com/babylm/babylm_data_preprocessing/blob/848e98f8d031154c14eeeae568748143c6b03af8/sample_chunks_and_split.py)

该版本主 shell 脚本对六源分别使用 10,000 / 5,000 / 2,000 行块；先生成 100M/train、dev、test，再从 100M 的 train 抽取 10M。**这验证历史代码的设计，不验证当前 2026 文件与该脚本的逐字对应关系。** 仓库另有旧 `sample_chunks_and_split_small.sh`，仍列出更多旧源和旧文件名，不能把它当 2026 命令直接执行。[固定主脚本](https://github.com/babylm/babylm_data_preprocessing/blob/848e98f8d031154c14eeeae568748143c6b03af8/sample_chunks_and_split.sh)

相邻保留块保持原顺序，但中间可能省略许多块。一本书或一个会话可以跨多个块；若块被分到不同 split，则源文档也可能跨 split。因此：

- 已验证：历史 split 的单位为行块，不保证源文档隔离。
- 未验证：2026 的精确抽样种子、最后精确词数裁切、过滤顺序、是否另做了文档级修复。
- 未验证：2026 10M 是否逐字属于 2026 100M；不能因历史流程如此就当作已核验。
- 未验证：当前 train/dev/test 的具体重合情况；此次未读取 dev/test，不能写“已消除泄漏”。

2026 数据卡说明删除被识别的问题句子，但未附这些删除的逐行日志或恢复映射。过滤后相邻句子不一定原先紧邻。有限官方源搜索没有定位到与当前六文件 SHA 对应的完整 2026 生成脚本；这应记录为未解决的来源缺口，不能宣布该脚本不存在。

## 可执行的短上下文准备建议

**现在可做、不会引入隐含文档假设的步骤：**

1. 保持原 TXT 不变。为每个原始非空行记录 `source_file`、一基 `line_number`、原始字节/文本哈希、空白分隔词数；给起始标记另加 `record_kind=metadata_header`。tokenizer 统计可以包含原始记录，但正式训练是否去掉 metadata 必须先固定，去掉的词数单独记账。
2. 技术验证暂用最多 256 或 512 tokens 的**单行内**因果片段；长行按预定规则切分，保留全部文本和重复暴露账目。不要为撑到固定长度复制行。若多个片段装进同一个物理 batch，attention 采用分段 mask；GDN 的递归状态、短卷积状态也必须在每个片段重置。仅给 attention 加文档 mask 不足以隔离混合模型。
3. 此设置仅称“保守的记录级工程检查/敏感性对照”。它切断了真实跨行对话或段落信息，因此不能悄悄变成唯一主实验，更不能拿它证明 QSA 不需要长程路由。主线程的 token 长度和实际稀疏密度审计应决定该短设置有没有足够非平凡稀疏查询。
4. 对 Gutenberg/CHILDES/Wiki，另生成 marker-aware sidecar，仅把显式标记作为**强制重置点**，其后片段暂标 `continuity=unverified`。不允许越过这些重置点拼接，更不允许按固定行号猜测已丢失的采样块边界。
5. 正式科学 packing 前，对长片段做原始来源匹配。以 PG ID / `.cha` 路径追踪，只把已有训练行映射回原文；把删除区间、跨书/会话跳转记录为断点。长上下文试验只使用连续性通过审计的片段，并报告其占训练词/token 的比例。原文只用于核验，不能把缺失内容补回训练集，也不能让原文进入 tokenizer。

开发/测试处理仍沿用官方 split，后续在固定版本和冻结评测策略后另做 exact/near-duplicate 审计；不要将当前 train 按行随机切成新的 dev 来冒充独立文档泛化。官方来源中潜在共享文档的问题须透明披露，不能以“官方”二字替代独立性检查。

主线程随后完成的 train-only tokenizer 账目已核对：`data/babylm-2026-tokenizer-16k-v0/manifest.json`，共 **16,325,414** 原文 tokens，全部 **1,104,106** 行可逐行解码还原。在逐行隔离这个反事实设置下，64 个四-token 微块仍保留约 **94.05%** 的因果边（按独立QSA审计修正当前完整块的计数后；初版94.09%保留在纠错归档）。这支持“逐行隔离不足以充当有效稀疏主实验”的判断，**不支持**“原始语料没有长文”的判断；两项审计回答的是不同问题。

## 下一批可直接实施的来源对齐（本次不下载大规模原文）

先用确定性规则选择最前面的 3 个 PG 标记（31993、52018、69225）与最前面的 3 个 CHILDES 会话标记，不能根据模型成绩或匹配效果挑来源。目标只确认方法是否能恢复可信片段，不代表整个语料已认证。

- **Gutenberg：**`PG31993` 可直接指向 [Project Gutenberg 书目 31993](https://www.gutenberg.org/ebooks/31993)，该官方页提供 plain-text 链接、书号和版本更新日期。后续对每本书只下载单本 UTF-8 原文到隔离的 audit-only 目录，存 URL、抓取时间、哈希。历史 BabyLM 脚本使用的 [pgcorpus/gutenberg](https://github.com/pgcorpus/gutenberg) 可帮助核对文件命名和历史清理逻辑，但实时原文不自动等于当年快照。
- **CHILDES：**本地标记如 `childes/CHILDES_NA/Brown/Eve/020100b.cha` 对应 [TalkBank 官方 Brown corpus](https://talkbank.org/childes/access/Eng-NA/Brown.html) 下的 Eve 会话。官方页提供可浏览与可下载 transcripts；遵守该源的访问与引用要求。复用已固定 BabyLM `get_record` / `process_text` 的清理逻辑进行离线匹配，不能把路径相同当作字节一致的证明。
- **匹配判定：**先对已保留训练行做唯一、精确的规范化匹配，规范化函数与版本固定；再做单调的原文位置映射。任意映射不唯一、位置倒退、跨文件或无法解释的缺口，都断开连续段。重复短句不能单独作定位锚点，需连续多行的联合锚定；输出每段原始训练行区间、源文本区间、置信类别及失败原因。仅凭文本语义相近的模糊匹配不晋升为已认证。
- **结果计数：**报告可认证词/token 覆盖率、最长连续段和大于目标 context 的可用段数；保留匹配失败。假如可信长段比例很小，仍可进行 BabyLM 语言学习研究，但暂不能据此承诺长上下文训练节省。
- **严格防扩充：**所有学习数据仍由固定训练文件中的原行选出；原文对齐缓存绝不进入 tokenizer/训练输入。被过滤/未抽中的文字不回填，原文中附加词的训练暴露必须为零。

## 本次没有得出的结论

没有认定 BabyLM 不适合本课题；没有认定它必然提供有效的长程检索训练；没有认定标记间片段连续；没有确认三种 split 文档隔离；没有测得模型能力、速度或稀疏收益。下一步可以明确缩小为“恢复训练语料内的可信上下文 + 验证真实路由密度”，再决定科学训练长度。
