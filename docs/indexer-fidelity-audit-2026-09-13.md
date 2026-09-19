# 索引器实现核对：先修诊断基线，再评价方法

本次核对仅针对公开公式、推理实现与当前缩小版索引器。没有找到并复现Qwen官方完整训练程序；不能把下面的改动称为修好了官方代码。

## 可以确定的差异

| 项目 | 当前旧参考 | 公开来源 | 本次处理 |
|---|---|---|---|
| RMS增益 | 固定为1 | NeMo QSA在Q/K各有一个可学习RMSNorm；vLLM GemmaRMSNorm使用1+weight | 加入从0开始的增益参数，保留旧实现 |
| 分数缩放 | sum ReLU(dot) | 报告式15及所查vLLM内核未除sqrt(d)；NeMo选块参考除sqrt(d) | 作为独立训练温度诊断，不能宣称这是官方训练的遗漏项 |
| 位置编码 | 相邻维配对、theta=10000 | 公开模型配置theta=10000000；vLLM文本路径使用NeoX式半区配对 | 记录差异，本次因果诊断不同时更改 |
| 维度 | 2头、32维，16维RoPE | 公开配置4头、128维；报告64维RoPE | 当前仍为缩小代理，不能声称架构完整一致 |
| 初始化 | PyTorch Linear默认初始化 | NeMo的scratch helper使用std=0.02截断正态 | helper属于推理框架，不能认定就是官方训练初始化；本次保持旧初始化 |
| 训练 | 自建子集KL、CPU参考 | NeMo入口明确冻结索引器，未给出辅助训练loss | 不能拿推理实现当完整训练配方 |

来源及精确代码位置：

- [QSA报告§2.1.2，式12–20](https://arxiv.org/html/2608.30320v1)，原快照 `literature/qsa-prefill-screen-2026-09-13/qsa.txt` 281–346行。
- [NeMo固定提交qsa.py](https://github.com/NVIDIA-NeMo/Automodel/blob/f7ccd6f7902634af34c2f31b3294ac250dc97670/nemo_automodel/components/models/qwen3_8_flash_next/qsa.py)，136、211、241、473–479、774–778行。
- [vLLM固定提交预处理](https://github.com/vllm-project/vllm/blob/319cc5ef19946d34c2e66cbbec5bda29d0bfa328/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py)，68–77行：可学习增益、归一化和RoPE。
- [vLLM固定提交打分内核](https://github.com/vllm-project/vllm/blob/319cc5ef19946d34c2e66cbbec5bda29d0bfa328/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py)，逐块打分与ReLU求和路径。
- [Qwen公开模型固定配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)。仅下载配置，未下载巨型模型权重。

上述6个实现/配置文件均保存于 `literature/indexer-fidelity-2026-09-13/`，带下载时间、固定提交与SHA256。外部源码仅阅读，没有运行。

## 一个会影响实验解释的数学区别

令索引分数为I，正数c乘上所有分数。在精确算术与相同平局规则下，TopK(cI)=TopK(I)。因此推理框架有无统一正比例缩放，可以产生相同路由。

但训练目标L=KL(p || softmax(cI))的导数为：

`dL/dI_j = c * (softmax(cI)_j - p_j)`。

所以相同路由不意味着相同训练。高分值的错误排序会改变softmax尖锐程度和梯度；ReLU所有头均处于负半轴时，正比例缩放也不能凭空恢复其局部导数。缩放可能改变进入失效状态的训练轨迹，不能保证恢复所有失效配置。

这是直接代数和常规归一化/温度控制，不是新的论文定理。现有技术报告、归一化实现与蒸馏文献均构成直接背景，不能把加RMS增益或除sqrt(d)包装成原创贡献。

## 事先固定的诊断

只选此前出现问题的两处：step1000第2层、step10000第5层。四个变体为原实现、仅可学习增益、仅除sqrt(d)、两者同时；分别跑原监督、全监督、抽查，3个配对种子，每次400步。共72次小索引器运行，学习率统一1e-4，其余维度、初始化、数据顺序和位置编码不变。

此轮只读取24训练段和8开发段，不加载测试段。观察全零query比例、梯度、完整教师KL及注意力输出误差。它检验失败原因，不是新数据上的方法收益验证。

另外固定一个新的文本确认配置：step1000检查点、第2/5层、3种子，所有方法均使用可学习RMS增益+1/sqrt(d)分数，比较原监督/抽查/全监督/原监督多训练。24训练、8开发、32测试均排除此前48段的文本哈希与token前缀哈希。若诊断未恢复学习则不启动；即使恢复也只使用本地CPU，不自动触发租卡。
