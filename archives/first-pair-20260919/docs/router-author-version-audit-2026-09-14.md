# 运行期间的代码版本核查

当前固定commit的paper_configs目录是历史配置，不应仅凭目录名声称完整复现2024发布栈。第一条件训练中，另获取2024-05-01之前最后修改model.py的提交b603008ff1a46880e1824f9218f9572d0c1b6bce（2024-02-20），比较本任务使用的Transformer默认分支。

在同一当前配置、相同seed123及相同已安装attention/config模块下，旧model.py与当前model.py全部状态名及初始化权重SHA相同，两个样本的完整logits最大差0。此项只隔离model.py，不是完整2024依赖栈复现。新增优化更新0，也不能据此排除所有实现、数据或优化问题。

原始源码、来源URL、SHA、API提交记录及数值结果保存在literature/router-author-version-audit-2026-09-14/。来源：[2024模型文件](https://github.com/HazyResearch/zoology/blob/b603008ff1a46880e1824f9218f9572d0c1b6bce/zoology/model.py)。当前测试无配置或训练中途修改。

补充0更新核查：旧attention.py与当前文件文本diff为空；旧数据文件当时名为associative_recall.py，首次按新文件名获取返回404，查仓库树后已找到并归档正确文件，不影响实验。单次通过、长度256/KV16/词表8192/零填充条件下，3个种子各32条数据的输入和标签与当前生成器逐元素一致。证据data-comparison.json。比较使用同一个当前NumPy及DataSegment，仍不等于完整旧依赖栈复现。

旧版[实验README](https://github.com/HazyResearch/zoology/blob/b603008ff1a46880e1824f9218f9572d0c1b6bce/zoology/experiments/iclr24_zoology_figure2/README.md)说明其结果取四个学习率中的最好表现。因此单学习率或本地子集失败不能被称为原论文基线失败。本批较高学习率0.01最终通过，较低预选值的负结果同时保留。
