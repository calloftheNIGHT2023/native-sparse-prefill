# BabyLM BLiMP/EWoK 题目转换与汇总接口

实现：`src/babylm_hybrid/official_task_records.py`。本模块只处理调用方已经提供的内存对象，不读取文件、不下载题目、不调用模型。当前新增工作是合成 fixture 上的工程接入，不是官方任务结果。

## 固定规则来源

只使用已经保存的官方 `babylm-org/babylm-eval` commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`：

- [read_files.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/read_files.py)：`decode_blimp`、`decode_ewok`。
- [dataset.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/dataset.py)：`__getitem__` 的 metadata 字段选择。
- [compute_results.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py)：`rank_and_evaluate` 的预测、平分随机选择和分类计数。
- [run.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/run.py)：`process_results` 的子类准确率及 UID 等权均值。

上游原样文件及 SHA-256 已在 `literature/babylm-causal-adapter-2026-09-17/manifest.json` 固定；没有获取新的真实 benchmark 内容。

## 题目转换

```python
normalize_record(
    raw_record, task="blimp", source_path="fixture.jsonl", line_number=1,
    source_sha256="<已核验原文件的64位SHA256>", raw_line=original_line,
    full_sentence_scores=False,
)
```

只支持两个 task 名：`blimp`（含 supplement 的格式）和 `ewok`。返回的 `sentences`、`prefixes`、`completions`、`label`、`UID` 及任务分类字段忠实跟随固定 decoder。额外提供独立的 `metadata`、`source` 和 `record_id`，它们不能混入官方子任务分类。

BLiMP：`sentence_good` 为候选0，`sentence_bad` 为候选1，label固定0，整句为completion。标准格式要求 `field`、`UID`、`linguistics_term`；`syntax_semantics` 按上游变为 `syntax/semantics`。没有 `field` 的 supplement 格式以文件stem作UID，field和linguistics_term都为supplement。

EWoK：固定题型比较同一 `Target1` 在 `Context1` 与 `Context2` 下的概率，label固定0；拼接时保留上游的一格空白，completion也包括前置空格。`Target2` 没有被这个 decoder 使用，不能擅自扩展为四候选题。`Domain` 变为UID，其他分类是 `context_type`、`context_contrast`、`target_contrast`。上游此函数的 `full_sentence_scores` 参数未改变输出；本模块如实记录请求值，保留相同实际语义，不将其误称为已实施的整句评分切换。

必需文本/类别字段类型严格校验，但允许原始JSON中的其他字段，它们只进入原始对象哈希，不进入分类。`source` 保存逻辑文件路径、一开始为1的原始物理行号、完整文件SHA、原始UTF-8行SHA（包括实际换行符）以及规范化JSON对象SHA。若调用方未提供原行，原行SHA明确为null，不能冒充原字节证据。record_id同时绑定task与这些来源字段；聚合拒绝重复ID。

文件存在性、文件哈希是否正确、manifest固定顺序、逐行计数和真实读取上限由外部runner负责。这个纯函数验证的是传入raw_line和raw_record一致，不能仅凭声明的source_sha证明文件已经被核验。

## 汇总和随机平分

```python
aggregate_scores(normalized_records, combined_score_result, tie_seed=0)
```

`combined_score_result` 采用已有 `score_records` 的结构；必须明确 `normalization="completion_token_sum"`，不能把均值打分混成这两个任务的sum打分。所有候选必须有正的 `scored_target_counts`，并明确 `scorable_candidates=True`、`ranking_allowed_per_record=True`。NaN/Inf、缺失题目、维度不等、任务/版本不符、零target、重复record_id均报错，不生成部分成功成绩。可附带 `record_ids` 核对来源顺序。

多batch调用scorer后只汇总一次，并附 `batch_sizes=[每批题数]`。平分按官方的 **batch → temperature → record** 次序，只在确实平分时用CPU `torch.randint` 在并列最高候选间均匀选择。本模块使用独立 `torch.Generator` 和固定tie_seed，不改动全局RNG；每题保存候选分数、并列候选、实际选择及来源关联。上游 `read_files` 使用未排序 `Path.iterdir()`，且原随机选择依赖全局RNG，因此此处固定manifest顺序与隔离seed是可复现的运行约定：相同顺序/seed下算法对齐，不能保证复现没有记录文件顺序/RNG的旧官方数字。

返回 `results` 中每个元素对应一个预先固定温度，不选择最佳温度：

- `counts_by_metadata` / `accuracy_by_metadata_pct`：各官方分类字段的总数、正确数和百分比准确率。
- `official_uid_mean_accuracy_pct`：先算每个UID的准确率，再等权平均，与两个支持任务的上游 `process_results` 相同。
- `diagnostic_item_accuracy_pct`：逐题总体准确率，仅为另列的诊断量，不能改称官方总分。不等大的UID下它可以与官方UID均值不同。
- `official_predictions`：`UID -> {predictions: [{id, pred}, ...]}`；ID沿用UID及其出现顺序。EWoK导出所选完整句子，BLiMP导出所选completion，符合上游。
- `items`：稳定record_id、源行、官方预测ID、选择与平分明细的对应关系。
- `tie_uncertainty`：只针对平分随机性的最坏、最好和期望准确率，分别计算逐题总体和UID均值。这些**不是统计置信区间**，不能把一次随机平分胜负解释为稳定提升。

标准BLiMP与supplement在官方shell中分别run，后续真实评测也应分开manifest/执行/汇总；不能合并两者的UID均值称作官方BLiMP分。合成样例可以混合两种schema来测解析器，但必须继续标明synthetic-only，不能冒充官方suite。

## 未覆盖和验证边界

本轮不扩展COMPS、Entity Tracking或GlobalPIQA的原始record与汇总，也不支持AoA、阅读或finetuning任务。已有scorer对其他规范化候选的数学打分支持，不代表本模块已实现这些任务的decoder与官方汇总。

独立测试代理负责合成fixture、固定官方函数AST oracle、失败输入及不等UID/tie案例；本实现任务没有模型前向、反向、更新或GPU调用。真实官方数据和真实检查点评测仍未执行。最终测试和源码hash以对应测试结果报告为准，不能由“代码已经接上”推断模型学会语言或论文结论成立。

独立测试结果保存在 `results/babylm-task-records-cpu-v0.json`，包含7项检查的最新状态、各次attempt及当时源码hash。首轮原始行hash断言暴露了Windows测试fixture自动LF→CRLF转换，修复测试文件写法后对相关项定向复验通过；不是悄悄删掉失败记录。另在严格contract审阅中补充拒绝缺失/错误normalization，针对第06项做纯函数回归。上述过程不执行模型。
