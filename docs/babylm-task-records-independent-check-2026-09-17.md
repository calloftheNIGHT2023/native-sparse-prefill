# BLiMP / EWoK 任务记录与聚合：独立语义核验

本次仅使用人为构造的原始 JSON 对象和候选分数。未读取或下载真实 benchmark 样本，未构造 tokenizer 或模型，未运行完整模型前向、反向、训练或 GPU。任务文件的读取/完整性由另一个 runner 负责；这里不把纯函数检查称作完整文件读取器验收。

独立 oracle 来自固定版本 `babylm-org/babylm-eval`：`6f825c291e2c4c78ad33b1935fd64d45f52642dc`。测试先核验已保存来源的 SHA256，再以 AST 仅抽取 `decode_blimp`、`decode_ewok`、`update_subset_to_stats`、`rank_and_evaluate`、`process_results`。没有执行上游模块的下载、数据集导入或 CLI。原始来源位于 `literature/babylm-causal-adapter-2026-09-17/strict/evaluation_pipeline/sentence_zero_shot/`。

## 核验的语义

- **BLiMP 与 supplement**：标准句/非标准句的候选顺序及 label=0 与官方 decoder 一致；`syntax_semantics` 标准化为 `syntax/semantics`。supplement 使用同一 `blimp` 任务名，UID来自文件stem，而不是另外虚构一个上游任务。
- **EWoK**：固定官方配对定义是 Target1 分别接在 Context1、Context2 后面，completion保留前导空格。这里没有擅自加入 Target2 并改成四候选评分。官方decoder的 `full_sentence_scores` 参数没有改变其返回记录；记录器对此作明确说明。EWoK导出的预测是选择的完整句子，不能误写成两个候选相同的target字符串。
- **身份与来源**：保留原始物理行号、文件SHA、原始行SHA、规范化原始对象SHA和稳定record_id；额外原始字段参与来源摘要，不偷偷新增统计子组。重复record_id、提供的score record IDs与顺序不一致、原始行与对象内容不一致时拒绝。
- **宏/微平均**：一个UID有3条全对、另一个UID有1条全错的独立例子，官方UID宏平均是50%，条目微平均是75%。两种数值分别命名，没有混为同一个指标。
- **并列候选**：两batch、两预先固定温度的抽样结果，与隔离环境中执行的官方抽样顺序 `batch → temperature → record` 一致。只在实际并列时抽样；本地generator不改变全局Python、NumPy或Torch RNG。固定seed重放一致，不根据结果选择温度。
- **并列的范围说明**：手算并列导致的最低/最高/期望正确率与实现一致；它描述既定候选分数下的并列不确定性，不是训练种子置信区间。
- **无效结果拒绝**：NaN、Inf、零目标、缺少候选、缺少记录、错温度维度、非法batch描述和非真实bool的rankable标记均不进入准确率。
- **评分定义**：BLiMP和EWoK必须明确声明 `completion_token_sum`。缺少该声明或声称 `completion_token_mean` 的结果被拒绝，不能将不同目标的分数静默归入同一官方指标。

## 有意保留的边界

官方ranking将NaN换为负无穷后继续选择；本实现拒绝这种输入，避免全无效候选变成随机预测。有效有限分数上的一致性不意味着必须复制这种异常处理行为。

当前只核验BLiMP/supplement和EWoK的记录及聚合。它不等于完整BabyLM suite，也不是正式模型分数。其他任务、全任务文件覆盖、实际模型似然接口和最终运行的端到端证据需分别验证。

## 失败保留与结果位置

第一次核验中，BLiMP两项子检查的原始行SHA断言失败。原因是**测试fixture**在Windows上用文本模式写文件，将输入的LF转换成CRLF，而传给normalizer的字符串仍是LF。随后fixture改成明确UTF-8 bytes，并增加Unicode U+2028正文及CRLF的原始行哈希检查；只定向重跑受影响检查。没有修改模型或编造评分来消除失败。

随后的独立源码审查发现聚合接口没有检查 `normalization`，runner组装结果也未传递该字段。实现者补充显式sum约束后，仅对第6项异常输入测试定向复测，确认mean/缺字段均被拒绝。该问题是审查发现的真实边界缺口，与首次测试fixture换行问题分开记录。

最终7项检查的最新结果均通过。累计执行17次固定官方纯函数、完成25条合成记录的规范化；3次attempt包含首次失败和两次针对性复测，不重跑无关算法，也不把早期检查重新归属到后改源码。

所有attempt、每项源码SHA和合成fixture保存在 `results/babylm-task-records-cpu-v0.json` 引用的日志中；初次失败没有删除。完整模型forward、backward、optimizer、真实benchmark读取与GPU计数均为0。以该JSON的最新汇总为准，不将早先通过项静默归属到后改源码。
