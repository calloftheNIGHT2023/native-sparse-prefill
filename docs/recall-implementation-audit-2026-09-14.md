# 关联回忆公开实现的定向核对

只核对生成器、默认配置和注意力/位置模块，未执行外部代码，也未复现论文全套实验。9个原始文件固定提交保存于`literature/recall-implementation-audit-2026-09-14/`，清单含下载时间与SHA256。

## Zoology

[固定版本生成器](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py)的例子和130—140行：在重复查询键的位置预测对应值，其余标签忽略；一段输入可以包含多个查询。键值记录在前，查询分布在后面，答案不是放在同一个查询位置的输入里。

该版本键、值均在每段内无放回采样；非查询位置可用随机词填充。我们使用独立填充词表、四token记录标记、可重复数值和单个最终目标的反事实家族，不能把自己的数据称为原版MQAR。

[默认配置](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/config.py)包含2层、128宽度和1e-3学习率。注意力模块使用因果softmax；模型可添加可学习的绝对位置嵌入。我们的GPTNeoX使用部分RoPE、4头和3e-4学习率，仍有模型与优化差异。当前对照只改变监督位置，没有同时改这些项，不能声称完整配方复现。

## MAD

[固定版本instances.py](https://github.com/athms/mad-lab/blob/0f49a452b84ca0d13f8eb9c1ffa649032376fb1b/mad/data/instances.py)的generate_in_context_recall_instance在训练分支返回完整序列的下一token目标，评测分支才使用带忽略位置的检索目标；noisy版本调用同一生成器。它与“训练时只对最终答案计算损失”并不相同。multi_query还会改变探测标签和末尾格式，因此不能只凭这个开关称为严格监督密度消融。

## 本项目据此做什么

多查询监督是已有任务设计，不作为新机制。本轮冻结同一输入、同一模型、同一更新预算，比较最终一题与四题的监督；多查询输入的前三题按非目标记录位置排序，最后一题是目标，属于本项目特定的训练控制。评测仍使用没有额外查询的单题，另做只替换查询键的开发集干预，避免把训练内位置规律误认成可泛化查找。

先检查这一因素能否帮助模型形成可靠的编号—数值对应，再决定是否值得研究稀疏路由。当前没有额外GPU作业。
