# 高记录数对照配置核查

在新题流80轮实验运行期间，只读核对固定上游commit 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb。来源源码和SHA存于literature/router-pressure-upstream-2026-09-14/；本记录不改变已冻结的两个实验。

作者仓库的[iclr24_zoology_figure2配置](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py)包含(长度256,16对KV)，而本次从basic教学配置直接扩到(128,16)。以下是源码配置差别，不代表我们已经成功复现作者的16对结果。

| 设置 | 本次压力任务 | 作者该目录的16对条件 |
|---|---|---|
| 长度/词表 | 128/256 | 256/8192 |
| 填充位置 | 随机tokens | random_non_queries=False，保留0 |
| 训练集 | 固定10000条或每轮新10000条 | 固定100000条 |
| batch | 32 | 256 |
| 逐token state mixer | 4倍宽MLP | Identity |
| 宽度/学习率 | 128/0.001单点 | 宽度64/128/256/512，学习率logspace(-4,-2,4) |
| 轮数 | 80 | 最多64 |

后续应先建立高记录数的阳性对照，并逐项加回当前压力条件；不能把现有教学设置单点失败视为作者基线普遍失败。也不能只去掉噪声就宣称解决了原有噪声任务。具体哪项造成困难仍未分离，本次资料核查不是因果验证或新贡献。

此外，另一个较新original_mqar_configs.py文件使用混合长度/记录数、不同样本量、batch256及32轮，并非上述paper_configs条件；不混用二者作为同一个精确复现配方。本轮没有据此启动新的架构/学习率扫描。
