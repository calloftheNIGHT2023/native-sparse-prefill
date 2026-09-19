# D/E 从头预训练参考实现：本地进度

2026-09-17。遵守最新 baseline-first 顺序：先密集 D 与 step-zero 稀疏 E，观察 gap 后才研究修复。本文记录实现门槛，尚无语言学习 gap 结果。

## 可运行实现

独立模块 `src/babylm_hybrid/`：config、model、attention。主干为 GDN/GDN/GDN/global 的重复结构，普通残差与 dense SwiGLU；不加载预训练权重，主干所有参数可训练。GDN 用已核验的本机 Transformers 数学参考，输出门改为 sigmoid；每个连续 segment 独立调用，卷积及递归状态同时重置。

`build_model(cfg, mode, backbone_seed, indexer_seed)` 分开主干与索引器 RNG，返回时不改变调用者 CPU RNG。D 不构建或计算索引器；E 在共同主干完成初始化后创建索引器，第一步就使用硬路由及已声明的 selected-support KL。teacher 与 indexer 输入 hidden detach，KL 对所有有效 loss query 归约，并在 global 层间取均值。没有添加密集预热、探索采样、路由课程或其他修复。

注意力参考包含 GQA、Q/K norm、NeoX 部分 RoPE、sigmoid 输出门、四-token 微块及因果尾部。**当前实现物化完整主注意力分数张量再施加稀疏 mask，属于正确性/质量参考。** 能用来判断一个有预算上限的质量问题，不是高性能稀疏训练内核，不能据此宣称节省总计算或据其慢速否定优化实现。

## 通过了什么

`tests/test_babylm_attention_v0.py` 七项检查首轮全部通过；`scripts/check_babylm_hybrid_v0.py` 执行完整混合模型九项检查，结果 `results/babylm-hybrid-cpu-v0.json`：

- 配对主干初始 tensor 完全相同，调用者 RNG 不变。
- 稀疏选择全部支持时，D/E logits 及所有共享主干梯度最大绝对差均为 0。
- 真正稀疏 fixture 下 LM-only 不给索引器梯度，KL-only 不给主干梯度。
- embedding、全部 GDN、global attention、FFN、norm、indexer 参数组均有有限梯度，在对应工程更新中改变。
- 未来输入、跨文档、相同标签再次出现、padding 的隔离通过；padding query 不计 loss。
- 模型、optimizer、CPU RNG、数据游标保存后，第二步输出、参数和 optimizer state 逐位精确重放。

这些是合成小张量工程测试，不是语言模型学会语言的证据。测试脚本计25次模型前向、10次反向、5次工程optimizer更新；主线程另有1次前向+1次反向 smoke、0更新。attention单模块测试另列，不与模型前向数混计。科学更新、真实训练token和GPU作业均0。

约97M候选已在CPU实际构建并计参：主干 **95,391,000**，索引器 **1,475,328**，E共 **96,866,328** 参数，全部可训练。未执行候选大模型前向。tiny模型D为41,626参数，E为42,410参数，只用于工程验证。

## 两组共用的真实训练输入

`scripts/prepare_babylm_windows_v0.py` 从已固定的官方 train tokens 生成只读索引：按源文件/显式记录标记重置，每窗最多2048 tokens，不跨已知边界。隐藏抽样拼接仍未知，不能把2048输入长度等同于2048自然语义连续性。此限制不阻止官方训练流上的首个D/E语言学习比较，但限制长上下文主张。

每轮22,598窗、16,325,414输入token、16,302,816个可监督next-token位置；尾窗不丢，token不重不漏。切开一个空白词时，在每个接触到该词的窗口计一次保守暴露，故每轮 **10,001,709词**。不能机械训练十整轮，采样器必须按实际窗口累计词数执行100M上限。

按这些真实窗口的长度，64×4预算保留 **25.5618%** 的因果边；77.2555%查询存在支持缩减。这确认输入下稀疏选择不是几乎全选，不是速度或质量测量。逐行隔离94%保留率的旧诊断仍保留，但不再作为主训练布局。

## 接下来只做首个D/E必要事项

1. 固定独立的官方dev编码/验证入口，tokenizer保持不变，最终测试继续封存。
2. 实现配对训练入口：相同窗口顺序、精确暴露计数、共同主干优化配方、独立indexer参数组；保存模型/optimizer/RNG/游标/配置/日志。先做工程演练，再冻结科学配置。
3. 在有连接与明确费用上限后，短GPU验证数值/显存/吞吐，估算两组完整轨迹；当前不套用旧LoRA吞吐，不启动未计费上限的任务。
4. 完成首个D/E学习曲线；单种子差距只作探索信号。观察稳定gap后再研究原因和修复，L/W暂缓。

完整Qwen4架构、GPU数值一致、混合精度、同质量成本、真实语言学习、原创贡献与论文可用证据都尚未由这些CPU测试确认。
