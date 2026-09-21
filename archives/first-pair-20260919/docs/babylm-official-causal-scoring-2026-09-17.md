# BabyLM 官方 causal likelihood 最小接口：工程审计

更新时间：2026-09-17 UTC。该接口只完成原生 HybridLM 与官方文本候选打分规则的衔接，**没有执行官方评测，没有读取最终测试题或答案，没有任务准确率，也没有新训练结果**。

## 固定来源

上游是 `babylm-org/babylm-eval`，固定 commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`。代码和文档保存在 `literature/babylm-causal-adapter-2026-09-17/`，`manifest.json` 列出每个来源的 URL、字节数与 SHA-256。下载范围仅包含代码、README 与文件树元数据；没有运行下载脚本、官方数据加载器或远程模型加载器。

直接核对的官方实现：

- [dataset.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/dataset.py)：text causal tokenization、completion offsets、右 padding 与输入/目标移位。
- [compute_results.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py)：logits 温度、目标 token 概率、求和与任务特定归一化。
- [run.py](https://github.com/babylm-org/babylm-eval/blob/6f825c291e2c4c78ad33b1935fd64d45f52642dc/strict/evaluation_pipeline/sentence_zero_shot/run.py) 与固定 `eval_zero_shot*.sh`：默认温度 1.0，官方脚本没有覆盖此默认值。

## 实现与可直接使用的接口

新增 `src/babylm_hybrid/official_causal_scoring.py`，不修改核心模型、训练器或 tokenizer。

```python
from src.babylm_hybrid.official_causal_scoring import (
    FixedLocalTokenizer, HybridCausalAdapter, score_records,
)

processor = FixedLocalTokenizer("data/babylm-2026-tokenizer-16k-v0/tokenizer.json")
# model 为已经核验来源/配置/权重哈希、加载完成的 HybridLM。
records = [{
    "sentences": ["A bird can fly.", "A bird can swim."],
    "completions": [" fly.", " swim."],
}]
scores = score_records(model, processor, records, "blimp", device="cpu")
# scores["scores"] 的维度为 [temperature][record][candidate]。

# 官方 compute_causal_results 接口也可调用以下 wrapper：
official_model = HybridCausalAdapter(model)
# official_model(input_ids=..., attention_mask=...) -> {"logits": ...}
```

示例为人为编写的接口样例，未作为真实 BLiMP 题目或实验数据使用。scorer 不加载检查点；model-only snapshot 仍须由调用方核验 SHA、按 `protocol.model_config` 与 `mode` 构建模型后加载 `model_state`。本轮没有运行这段示例。

## 与官方保持一致的数学和边界规则

1. 对完整 sentence 只分词一次，用字符 offsets 判断 token 是否与 completion 后缀重叠，不能分别分词 prefix/completion 后拼接。跨越后缀起点的 token 计入后缀。
2. 官方 causal 处理本身不添加 BOS/EOS。冻结的 train-only 16K ByteLevel BPE 无 postprocessor，本接口也不插入。整句打分自然不能计算第一个 token 的概率。特殊 token 仍存在于词表，但不会自动进入输入。
3. 每个候选位置分别批处理：先右 padding，再输入 `tokens[:, :-1]`、目标 `tokens[:, 1:]`，attention mask 去最后一列，phrase mask 去第一列。短句在混合长度 batch 中可能多前向计算自己的末 token，该位置的目标 phrase mask 为零；账目保留这项实际开销。
4. 对温度 \(T\)，概率为 `log_softmax(logits / T)`，按目标 token gather；只累加移位后的 phrase mask。默认且建议预先固定的温度是 **1.0**。接口可接收预先固定的温度列表，但不做 best-temperature 选择，不读取标签或调参。
5. `global_piqa_parallel` / `global_piqa_nonparallel` 使用 completion token 均值；`blimp`、`ewok`、`comps`、`entity_tracking` 使用 completion token 对数概率之和。BLiMP supplement 在官方管线中同属 `blimp`。
6. wrapper 将 attention mask 的有效位置映射到 segment 0，右侧 padding 映射到 -1。GDN 每行有效段独立重置，padding 不进入其卷积或递推状态。scorer 使用 `torch.no_grad()`，结束或异常时恢复模型及全部子模块原来的 train/eval 状态。
7. 使用模型输出 logits，**不使用模型内部 lm_loss 或 aux_loss**。没有因为后缀打分而改变 QSA 的 attention 支持，也没有新增 teacher/方法。

## 覆盖范围与明确未完成项

| 项目 | 当前状态 |
|---|---|
| 上述六个 task 名称的、已规范为 sentences/completions 的文本候选打分 | fixture 与固定官方代码逐项对齐 |
| BLiMP/supplement、EWoK、COMPS、Entity Tracking 的官方 raw reader / 元数据字段格式 | 只读代码了解范围；未连接或运行真实数据 reader |
| GlobalPIQA | 仅已规范文本候选的均值打分规则；未下载数据、没有比赛成绩 |
| 官方准确率、子集聚合、随机 tie-breaking、预测导出 | 未实现/未运行；由后续官方管线衔接负责 |
| 阅读 surprisal、AoA、(Super)GLUE finetuning、multimodal、MLM/MNTP/encoder-decoder | 不支持，不能用该接口代替 |
| 官方 fast suite / full suite | 都未执行；本项目内部 dev NLL panel 仍不是官方 fast suite |
| 新模型质量、dense/sparse gap、速度/成本优势 | 本轮没有证据，仅工程准备 |

候选数量必须在同一 batch 中一致，completion 必须为 sentence 的真实字符串后缀，默认 forward 输入上限 2048 token；超限直接报错，不能悄悄截断。该上限不是长文本能力证明。

## 失败边界和账目

- 固定上游对空文本会访问不存在的 `offset[0]`，对 flat one-token 文本会误当嵌套 batch 展开。接口明确拒绝不足两个 token 的文本；没有通过添加 BOS 来偷偷改变评测。此前沿上的官方整套兼容性仍需处理，不能宣称所有输入都覆盖。
- 空 completion 的求和/归一化结果按上游规则为 0，但 `scorable_candidates=False`、`ranking_allowed_per_record=False`。不能把缺少可评分目标的 0 当成好成绩或参与正常候选排名。
- 对 logits、温度后的 log probabilities 和最后 score 均检查有限性；无效值 fail-closed，不沿用上游将 NaN 改为负无穷后继续排名的行为。
- 运行失败抛出 `CausalScoringError`，保存原始 cause、task、candidate index，以及已提交/已完成前向、实际非 pad/padded input positions、已完成 scoring 的 token 数。预先规范化/校验阶段失败发生在模型调用之前。
- 返回的 `source_tokens` 是当前 batch 全部候选的原始 token 数，`submitted_*` 是实际提交的工作；`model_forward_attempts` 包括失败调用，`model_forward_calls` 只包括成功返回的调用。逐温度重复 log-softmax 不会冒充额外模型前向。

## 验证和本轮实际计数

测试文件 `tests/test_babylm_official_scoring_v0.py` 仅通过 AST 提取固定源码中的处理、collate 与 causal compute 函数；不会导入官方 data loader，也不会触发远程权重/数据下载。合成字符 tokenizer、合并 token 后缀重叠样例和冻结 BPE 的 Unicode 样例检验 offsets、padding 和首 token；fake logits 检验六个 task 的求和/均值以及 1.0/0.7 两个预设温度。比较到官方算子为逐值一致。

第一次运行：

```powershell
.venv/Scripts/python.exe -m tests.test_babylm_official_scoring_v0
```

2026-09-17T22:15:54.671680Z：当时 9 个测试全部通过，26 次 fake forward、**3 次 tiny CPU sparse HybridLM forward**。同一个随机 tiny 模型的一次两行 batch 与两次单行前向检验 padding 不改变分数，固定容差 `atol=5e-5, rtol=1e-5`；没有训练、反向或参数更新。

独立只读审阅指出失败账目需要附带异常返回，以及极端正温度仍可能数值溢出；修复后仅运行 fixture 测试：

```powershell
.venv/Scripts/python.exe -m tests.test_babylm_official_scoring_v0 --skip-tiny
```

2026-09-17T22:17:32.276176Z：9 个 fixture 测试全部通过，27 次 fake forward，0 次真实模型 forward。新增温度溢出和异常账目断言。未重复 tiny 模型前向。

**本轮累计：53 次 fake forward、3 次真实 tiny CPU forward、0 次 backward、0 次 optimizer update、0 次 GPU forward、0 次 97M forward、0 个官方真实 benchmark item 读取。** 三次预期故障调用（两次 fake 显式故障、一次温度溢出）包含在 fake 计数，测试均按预期捕获，没有被删除或冒充成功模型工作。

工程结果与失败轨迹保留在 `results/babylm-official-causal-scoring-v0/history.jsonl`；最后 fixture 复验为 `latest.json`，完整本轮汇总为 `task-summary.json`。`latest.json` 的 tiny 计数为 0 仅表示最后一次 fixture-only 运行，不能据此抹掉此前 3 次真实前向。

最终 adapter SHA-256：`04e8434901e774b0c46256156c81be1acc6e899b1c97b1d0639609671b1a22c9`。

最终测试 SHA-256：`917dafcc62c77e0539947ad6daec8e9429a1f58501a271280f3fb09fbb0d481b`。

冻结 tokenizer SHA-256：`230b9d6993dcaf32f40cec2c44ac79d7d713213616ba2d8e45ad4a96e5a9dbe6`。

上游来源 manifest SHA-256：`e8a5b16831d8733f3c36b33dbdbf432c0ce7b834db79a80c2c11a60f4989c671`。

下一步可以把经过哈希核验的 D/E model-only snapshot 交给这个 wrapper，再按事先固定的官方任务子集与检查点协议运行评测；这一步仍未执行。完整 suite 的 reader、标签/聚合及任务数据治理仍是明确待办，不应把当前接口改名为“已通过官方评测”。
