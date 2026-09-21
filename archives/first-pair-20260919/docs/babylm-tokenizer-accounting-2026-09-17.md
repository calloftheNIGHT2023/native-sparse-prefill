# BabyLM train-only tokenizer 与原始记录账目

2026-09-17。仅本地 CPU 数据准备，科学训练更新为 0，无新增 GPU 运行。语料限于固定的 Strict-small 六个 train 文件，没有读取 dev/test 或预训练模型/tokenizer。

## 产物与核验

`data/babylm-2026-tokenizer-16k-v0/tokenizer.json` 为 ByteLevel BPE，词表 16,384，min_frequency=2，不做文本归一化、不自动增加前缀空格；特殊 ID 为 pad=0、bos=1、eos=2。本阶段不插入 BOS/EOS、不做截断或 packing。训练数据中无这三个特殊符号的字面字符串。

每个物理行原样保留，包括换行符；只作为可逆存储单元，不声明它是一篇文档。保存 uint32 token 流、uint64 记录偏移和 uint32 记录词数。模型训练不得把存储中相邻记录直接当作已认证的连续文档。

|源|空白分隔词|原文 token|物理行|
|---|---:|---:|---:|
|BNC spoken|762,073|1,039,290|65,220|
|CHILDES|2,841,101|5,741,137|531,305|
|Gutenberg|2,557,721|3,482,910|59,934|
|OpenSubtitles|2,282,877|3,687,419|386,046|
|Simple Wiki|1,531,437|2,333,310|58,709|
|Switchboard|24,791|41,348|2,892|
|合计|10,000,000|16,325,414|1,104,106|

构建时每条记录均检查 encode/decode 无损；独立验证脚本再从磁盘 token/offset 文件解码所有记录，重建六个源文件的完整 SHA256，均与固定原始数据相同。逐记录词数、token ID 范围、偏移、总词数与全部产物哈希通过。验证证据：`results/babylm-tokenizer-ledger-validation-v0.json`。

16,325,414 是一次原始语料完整编码的 token 数，**不是已经训练的 token 数**。十次读取对应原文部分 163,254,140 token；最终输入还取决于实际分段、BOS/EOS、重叠/截断和 loss mask。不得据此直接记录训练暴露。tokenizer 学习不是语言模型预训练。

## 短记录的稀疏率诊断

如果假设每个物理行单独隔离，以 QSA 4-token 微块、最多64完整块、0–3个尾 token计数，则保留 421,579,347 / 448,258,339 = **94.0483%** 的因果 query-key 对。此设置只丢弃约5.95%连接，并非所期望的明显稀疏工作负载。

这是条件性的布局算术，不是吞吐、质量结果，也不意味着整个语料缺少长文。部分源保留上游文档标记；标记间片段与隐藏采样拼接需另行认证。数据审计的连续性问题不通过缩小k或不加说明地跨行拼接来掩盖。

计数初版错误地在当前微块刚完成时仍将其视作强制尾部。独立 QSA 源码审计发现后已修正：完整块包括当前刚完成的块，均参加 top-k；只有不完整尾部强制保留。初版脚本、manifest、验证结果保存在 `provenance/babylm-ledger-mask-correction-20260917/`。tokenizer、原始记录与任何科学结果没有改变。修正后的公式通过768组独立显式集合枚举。

## 后续门槛

1. 根据源文档标记和可追溯抽样路径认证片段；固定数据划分与 packing 策略。
2. 为实际输入建立 words/read exposures、text tokens、special/pad tokens 和 loss tokens 的不同计数。
3. 用真实输入布局重新统计有效稀疏比例，并重置 GDN 卷积/递归状态与 QSA mask。
4. tokenizer 与记录账目已就绪；训练样本、完整模型、GPU内核与付费门槛尚未通过。

复现：`scripts/prepare_babylm_tokenizer_v0.py` 验证已有产物后复用；`scripts/validate_babylm_record_ledger_v0.py` 从磁盘重建源字节。所有实验应固定保存的 tokenizer SHA256，不应依赖不同版本库的重新训练隐式一致。
