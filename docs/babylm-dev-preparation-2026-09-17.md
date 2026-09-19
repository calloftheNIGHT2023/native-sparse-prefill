# BabyLM 官方 dev 固定与训练集重合审计

2026-09-17。完成时间 **20:59:48 UTC**。只准备开发集和词/token账目，没有运行模型前向、科学训练或GPU，没有读取最终test，也没有改变冻结tokenizer、训练数据或候选协议。

## 已固定内容

来源为 [BabyLM-community/BabyLM-dev 固定版本](https://huggingface.co/datasets/BabyLM-community/BabyLM-dev/tree/169f42e32d0aaf65ec6b91d55bafad27a3afc729)，revision **169f42e32d0aaf65ec6b91d55bafad27a3afc729**。先读取已有 `literature/babylm-boundary-audit-2026-09-17/dev-metadata.json`，再获取这一revision的官方文件树。六份`.dev`实际合计 **56,752,783字节**，低于事先规定的100,000,000字节下载上限。

每份文件均校验官方字节数及LFS SHA256或Git blob SHA1，并另外保存本地SHA256。未将仓库metadata中的usedStorage当成本次六文件下载总量。

- 原文：`data/babylm-dev-raw-v0/`，`manifest.json`记录URL、官方hash、下载/验证时间。
- 官方tree与来源：`literature/babylm-dev-preparation-2026-09-17/dev-tree.json`及`source-manifest.json`。
- token账目：`data/babylm-dev-token-ledger-v0/`。
- 重合统计：`data/babylm-dev-token-ledger-v0/train-dev-overlap.json`。
- 脚本：`scripts/prepare_babylm_dev_v0.py`。
- 首轮原始日志：`logs/babylm-dev-preparation-v0-first.log`。

使用现有训练集学习、已经冻结的16,384词表byte-level BPE，仅调用`Tokenizer.from_file`和encode，绝不重新训练。tokenizer SHA256前后均为 **230b9d6993dcaf32f40cec2c44ac79d7d713213616ba2d8e45ad4a96e5a9dbe6**。

## 数据与token账目

|源|原始行记录|空白分隔词|原文tokens|零词记录|
|---|---:|---:|---:|---:|
|BNC spoken|130,000|1,252,593|1,755,630|117|
|CHILDES|520,153|2,716,591|5,951,802|870|
|Gutenberg|65,000|2,819,070|3,898,319|58|
|OpenSubtitles|375,000|2,077,019|3,430,198|405|
|SimpleWiki|60,000|1,405,366|2,150,489|10,637|
|Switchboard|18,000|148,340|251,096|0|
|合计|**1,168,153**|**10,418,979**|**17,437,534**|**12,087**|

所有1,168,153条记录逐条完成encode/decode原文回环，包括原有换行和空白；不插入BOS/EOS，不截断，不去重。dev含空白记录，与当前train没有空白行不同；这些原文记录仍完整保留。

每源输出与train相同的低层账目格式：`<source>.ids.u32`是小端uint32 token数组，`offsets.npy`为uint64记录边界，`words.npy`为uint32原始记录词数。逐项检查token文件长度、offset差分和词数总和；manifest保存19个产物hash（18个ledger文件加重合审计）。再次运行返回`already_verified`，核验已有文件而不重新下载或编码。

这些token数组只是无损序列化；相邻行不是自动认证的连续文档。**尚未创建dev NLL的window index，尚未生成任何NLL数值。** 下一步需将训练已固定的source/header reset、最长2048输入窗口、末token不计next-token loss等规则原样用于dev，逐token归约NLL；不能简单把全部dev token串成一条序列，也不能对各文件NLL无权平均。

## Exact非空行重合

比较定义：只移除行尾CR/LF，保留其余空白、大小写、Unicode原样；排除纯空白行。比较当前固定Strict-small全部train与本次全部dev；另保存同源和跨全部train来源两套计数。没有做近重复、语义匹配或文档重建。

- dev共有 **1,156,066** 次非空行出现，**841,108** 种不同非空行。
- train/dev共有 **29,913** 种完全相同行；对应dev中 **189,455** 次出现。
- dev中至少20词的非空行共 **89,095** 次；其中 **16** 种完整行与train完全相同，在dev出现 **22** 次，合计 **661** 词。

大量完全相同的短问候、回答或常见短句不能直接等同训练泄漏，因此同时给出长行阈值。反过来，长行重合少也不证明文档级独立；本审计不检查近重复或未标出的同文档片段。

|dev源|与全部train完全相同的dev行出现次数|其中至少20词的出现次数|
|---|---:|---:|
|BNC spoken|28,447|4|
|CHILDES|88,127|2|
|Gutenberg|2,224|0|
|OpenSubtitles|62,017|5|
|SimpleWiki|3,698|11|
|Switchboard|4,942|0|

**没有删除任何重合行，没有根据模型结果选择子集，也没有声称已经去除泄漏。** 初次D/E的原始官方dev分数须保留此限制。若以后另做去重敏感性分析，应预先冻结规则，单独报告，不能事后替换主结果。

## 显式文档标记重合

沿用已有边界审计的`= = = ... = = =`整行标记模式，标记比较只修剪行首尾空白，不做语义归一。

|源|train显式标记数|dev显式标记数|完全相同标记标签|
|---|---:|---:|---:|
|CHILDES|1,565|997|0|
|Gutenberg|59|58|0|
|SimpleWiki|14,800|10,639|0|

其他三源没有该形式标记。观察到的标记标签无重合，只说明这些已出现的标签字符串不同；有未带header的前缀、历史抽样连接和缺少来源映射，因此**不能据此认证文档级独立，不能称已完成文档去重**。

## 复现与后续接口

在项目根目录执行：`.venv/Scripts/python.exe scripts/prepare_babylm_dev_v0.py`。

准备脚本首轮成功，无下载/校验/编码失败；复跑也通过。科学更新0、模型前向0、GPU小时0。提供的是固定全量dev原文与ledger，不是最终BabyLM benchmark，不是论文质量证据。正式dev window adapter必须消费上述独立目录，不触碰既有训练流和tokenizer；任何固定限额dev筛查子集须在看模型成绩前单独定义并记录。
