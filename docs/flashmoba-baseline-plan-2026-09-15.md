# 官方优化基线：冻结验证计划

日期：2026-09-15 UTC。本计划在官方算子安装完成、执行新 gate / benchmark 之前记录。

## 本轮问题

前一阶段的候选块 min/max 原型可以近似旧 token-top8 小模型，但没有测到相对 Flash 的速度优势。本轮接入官方 FlashMoBA，确认一个强块稀疏基线在现有 RTX 6000 Ada 上能否正确运行，以及真实的全查询 prefill 开销。官方块选择是已知算法；安装、调用适配与本轮测量不计原创研究贡献。

不再把旧原型、官方 MoBA 和 dense 当成同输出算法。MoBA topk 包含当前块，选中块内全部 token；旧原型选若干候选块，最后只算 6 个远端与 2 个局部 token 的注意力。质量匹配要在训练相应算法后另证。

## 环境与源码

- 官方 https://github.com/mit-han-lab/flash-moba ，commit `39d9ac043b271d046a2181a9991e99a26b67bca1`。
- CUTLASS `a2439551c765c5393aebe557ee75d3a0412d2211`。
- 隔离 Python 3.11 / torch 2.8.0+cu128 / 私有 CUDA 12.9.1 工具链；不替换原环境、驱动或系统 CUDA。
- 官方编译参数：sm80 cubin，MAX_JOBS=4，NVCC_THREADS=2；Ada 8.9 的实际执行仍须测试。
- CUDA toolkit 12.9 与驱动报告 CUDA 12.8 的小版本兼容不等于无条件可用，须以具体算子运行结果为准。
- 官方当前聚合 kernel 要求 key block 是 64 的整数倍，不能直接拿原型 block32 作官方配置。

## 先过正确性门槛

12 条 case：FP16 / BF16，head_dim 64 / 128，MHA / MQA / GQA，非整块长度和变长打包，自回归全查询，不涉及 decode。分开验证：

1. 从官方 CSC 索引还原路由，与独立均值路由比较；记录边界近似并要求评分 regret < 2e-4。
2. 在相同路由上，与普通 FP32 dense masked softmax 比较前向、Q/K/V 梯度。
3. 修改每条序列后半段 K/V，前半段输出须逐元素一致，以排除未来泄露。

门槛失败则保存失败日志、定位原因，不将失败当作成功计时。硬 topk 不对选中集合本身求导。

## Prefill 比较

固定同一个随机 BF16 QKV，B=1、D=128，H=1/8，N=8192/16384/32768。三个 MoBA 配置为 B64 K2、B128 K2、B128 K4；K 包含当前块。dense 强制 PyTorch Flash SDPA，记录 profiler 运算符以确认未退回 math。11 次交错顺序采样、每次 3 个完整调用，保存全部样本、GPU event 和同步墙钟、预热与峰值额外显存。

所有调用计入摘要、路由、索引排序与聚合。首先比较未经修改官方 API 的 eager 总耗时。官方均值池化的 metadata `.item()` 会造成主机同步；可另测只缓存固定长度元数据的包装器，但必须先证明前向/梯度一致，且 CUDA Graph 录制后改变 Q/K/V，输出仍与重新计算的官方 API 一致。禁止缓存摘要、路由或选中支撑再把时间说成端到端。

随机输入速度不能说明长文本质量，也不是从头训练吞吐或训练收敛。此轮 scientific optimizer updates 预定为 0；编译、测试时间单独记录。

## 分块边界方向的查重结论

[LongLoRA](https://arxiv.org/abs/2309.12307) 已有 shifted short attention，不能把移动窗口或跨块传播作为新机制。[PBS-Attn](https://arxiv.org/abs/2510.21270) §3 已利用 token 重排改变块内聚集和稀疏布局；不能把重新分组本身当成新贡献。

[Boundary Repair](https://arxiv.org/abs/2606.02680) §3–4 研究固定局部块图的边界不可达性并加入边界连接。它没有直接证明动态 MoBA 有相同的结构性屏障，但使“边界有问题 + 补连接”的宽泛提法严重重合。本轮不启动该提法的训练。

这是有限范围的全文章节核验，不是证明没有其他相似论文。是否存在动态内容路由中特定、可推广且未被覆盖的问题，仍未建立。
