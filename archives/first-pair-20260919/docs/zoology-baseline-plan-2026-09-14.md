# Zoology公开最小例子的CPU复现计划

固定上游提交1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb，直接使用basic_examples/basic.py、LanguageModel、prepare_data和Trainer。平台适配包括TokenEmbeddings默认设备cuda改cpu，以及prepare_data种子显式使用np.int64；后者修复Windows默认C-long为32位导致验证种子越界。原文件、18文件来源SHA和补丁保存在third_party/zoology-1ad20d1。使用新版本机依赖，完整锁定表另存；不是声称与作者GPU环境逐位一致。v0数据准备失败，更新数0，原记录保留。

## 原始配置保留

- 词表256、长度64、4组键值，多查询。10000训练序列、1000原验证序列，数据seed123。
- 128宽、2层、1头，绝对位置嵌入，MLP倍率4、共享输入输出词嵌入；embedding dropout0.1、attention dropout0.1。
- batch32，AdamW学习率1e-3、weight decay0.1，不加梯度裁剪；CosineAnnealingLR每epoch更新，T_max100，eta_min0。
- 模型seed123，最多100epochs；使用上游规则：epoch末原验证准确率严格大于99%才提前停止。这里原代码叫test的集合实际参与停止决策，报告中统一称验证集。
- 原数据加载器不shuffle，batch最后不足32保留；不擅自改成新题流、warmup或逐步调度。

## 记录与核验

使用原Trainer.fit/train_epoch/test，仅在子类和优化器hook中记录计时、批量大小、梯度/损失、数据、随机状态、配置、源码和epoch checkpoint；不改变损失或更新。W&B不连接账户，所有日志本地保存。

预先核验源文件唯一设备差异、标签在查询键位置且来自前方记录、共享权重和无未来梯度。运行结束另生成1000条新序列，只用最终权重测一次，不依据这个集合调整停止或配置；保存实际种子和数据SHA。

本机CPU安全时间上限3600秒，异常或时间上限保存checkpoint和原因；不能把中止记为复现完成。若通过，表述为公开最小例子在CPU上跑通，不能当作新方法、原论文全套复现或Qwen结果。只有基线通过后，才逐项加入远距离约束。本轮不启动GPU。
