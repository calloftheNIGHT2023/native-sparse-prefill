# BabyLM D/E 配对训练流与词暴露规则 v0

2026-09-17。遵从 `babylm-baseline-first-priority-2026-09-17.md`：先建立从零开始的 D/E 语言学习对照，不以完整原文重建阻挡首个基线。这里固定的是输入流与账目，不是训练超参或科学结果。

实现：`scripts/prepare_babylm_windows_v0.py`。结果：`data/babylm-2026-windows-v0/manifest.json`。只引用已有 `.ids.u32`，不复制大 token 文件，不训练新 tokenizer，不读取 dev/test，不运行模型/GPU。

## 固定输入政策

- 使用已验证的 BabyLM 2026 Strict-small revision `c92ab16b4f08858304b0815706065b3354d8fc0a` 及固定 16,384 词表 ByteLevel BPE；全部 tokenizer artifact/source 哈希在生成前复验。
- 以源文件与显式整行 `= = = ... = = =` 标记为强制边界。标记本身保留在**后一个片段**开头，词/token 都纳入原始 10M 词账目。
- 每片段保持原 token 顺序，切成不重叠、最多 **2048 tokens** 的窗口；最后不足 2048 的尾窗保留，不补齐、不丢弃，不跨文件或显式标记。
- 每个窗口是一个独立模型状态片段；position、attention、GDN 递归状态和短卷积状态均从该窗起点重置。多个窗口可同批，但不得互看或共享状态。
- 默认不添加 BOS/EOS。输入包含窗口全部 token，训练目标为窗内 next-token；最后一个输入 token 不预测下一窗。因此长度 L 的窗口有 L 个输入 token、max(L−1, 0) 个有效 next-token loss 位置。
- 上游抽样/过滤引入的隐藏断点仍未知；这些窗口是固定的训练流切片，不被描述为连续完整自然文档，不由 2048 的工程长度推出 2048 的真实语义依赖。

## 一轮遍历的精确账目

|量|值|
|---|---:|
|原始语料空白分隔词|10,000,000|
|训练流窗口|22,598|
|所有输入 token|16,325,414|
|有效 next-token loss 位置|16,302,816|
|保守窗口词暴露|10,001,709|
|因词跨窗口而额外计入的暴露|1,709|
|丢弃 / 重复 / 插入的 token|0 / 0 / 0|

**词暴露如何得到：**完整行直接使用已验证的原始词数。仅对窗口切开的行，用原固定 tokenizer 重新编码以取得字符 offsets；每一枚重编码 token ID 必须逐个等于持久化 token，且完整解码还原原行。原始 `\S+` 词与窗口所含 token 的字符区间相交，就为该窗口记一次；一个词若被切在两个窗口，两个窗口各计一次。处理包含共享 Unicode 字符 offset 的 byte tokens，不用固定 tokens/word 比例或 token 字面空格猜词数。

共对 **5,713 行**补取 offsets。额外 1,709 次暴露另用“每个窗口切点跨越了几个原词”的计数独立核验，必须恰好等于窗口累计暴露减 10M。

每次模型真正读入一个窗口时，训练器累计其 `word_exposures`，不能只在 epoch 末更新。优化器失败后若重放窗口，重放的输入另计暴露和成本；不同 scientific run 分别计预算，研究总账另行汇总。

**不能机械运行十整轮：**十整轮将记 100,017,090 词，比 100M 上限多 17,090。按当前规范顺序，一个可用的完整窗口前缀为 9 整轮加第 10 轮前 22,583 窗，共 **99,999,853 词**，剩 147 词不足容纳下一窗。它只是预算示例，尚未执行训练；若协议使用固定随机顺序，训练器必须在该顺序下重新逐窗执行上限检查，D/E 使用相同终点，不截掉半窗来凑满上限。

## 当前 QSA 预算的组合密度

以微块大小 4、最多选择 64 个完整可见块计算。窗口内 query 索引 q 从 0 起：

`retained(q) = 4 × min(floor((q+1)/4), 64) + ((q+1) mod 4)`。

**当前块恰好完整时也进入可选集合，可以被 top-k 丢弃；只有不足 4 tokens 的尾部始终可见。** 不能强制额外保留当前完整块。该约定已用显式因果集合枚举核验，包括第 260 个可见 token 处从 65 个完整块选 64 个的边界。

- 全部窗口的密集因果边：14,285,117,262。
- QSA64×4 保留边：3,651,528,538，保留率 **25.561768%**。
- 实际减少支持的 query：12,612,273，占输入 query 的 **77.255456%**。

这是该输入政策下的精确支持数量，与 top-k 具体选中哪些块无关。它**不是已测速度、索引器学习结果或质量证据**；辅助目标开销另计。相较逐行隔离的高保留率，当前训练流至少具备非平凡的稀疏选择机会。

## 读取接口和 sidecar

`windows.u64.npy` 是 22,598 行的 uint64 数组，列名在 manifest 的 `columns`：

`source_index, segment_index_in_source, source_token_start, source_token_end, first_record_index, last_record_index, first_record_token_start, last_record_token_end, word_exposures, input_tokens, next_token_loss_positions`。

token 区间左闭右开，record 索引从 0 起，first/last record 均包含。源 token 路径与哈希在 `source_summaries`；显式标记的原 record/token 坐标在 `known-boundaries.json`。每窗只属于一个片段，全部 token 在本轮恰好覆盖一次。

```python
from scripts.prepare_babylm_windows_v0 import iter_windows

for item in iter_windows():
    ids = item["input_ids"]       # readonly NumPy uint32 view
    words = item["word_exposures"]
    loss_tokens = item["loss_tokens"]
    # 转 tensor 时由训练器选择安全 dtype / copy；不在此接口启动训练。
    # 模型状态在每窗重置；loss 只配对 ids[:-1] 位置的 logits 与 ids[1:] 标签。
```

`iter_windows(window_indices=...)` 也接受预先冻结的窗口索引序列；不会自行打乱。默认复验 sidecar 和访问到的源 token 哈希；调用方还应核对协议内固定的 manifest 哈希。NumPy 只读视图不允许原地修改，padding/masking 应在 batch 端另建数组。

## 已通过的检查及边界

- Unicode、空白与词中切分的全部小切片，对照逐 token/逐词字符相交的独立枚举。
- 不完整尾窗、显式边界、右端正好位于 record 边界的索引测试。
- 完整支持/64 块边界上的显式因果集合计数。
- 全文件源哈希、token/record/词数一致，逐源无缺口无重叠覆盖，首中末窗口只读读取器检查。
- 科学模型训练次数、优化器更新、GPU 小时数均为 **0**。

该数据入口可以服务第一个 D/E 基线；模型数值正确性、训练运行协议、开发集与官方评测、费用上限仍由相应门槛决定。前一份文档边界审计的未知项继续保留，不因有可运行的 packing 就声明原文连续性已解决。
