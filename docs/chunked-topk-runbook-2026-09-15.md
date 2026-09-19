# 分块 top-k 工程基线的运行说明

## 已完成与适用范围

此实现保持 2 个局部位置 + 6 个内容位置的因果规则。候选精确打分仍为二次复杂度，选中边的汇总与反向传播不保存 N×N 矩阵。它是 PyTorch 工程基线，不是新的算法或融合内核。

本地已有两个可用解释器，无需现在租卡或改变已有环境：

- CPU：`D:/ChatGPT/projects/native-sparse-prefill/.venv/Scripts/python.exe`，Python 3.12 / torch 2.10.0+cpu。
- 本机 GPU：`C:/Users/USER/AppData/Local/Programs/Python/Python314/python.exe`，torch 2.10.0+cu128，支持 RTX 5070。此环境缺少完整 Zoology 任务的一些依赖，当前只运行纯 PyTorch 算子验证和计时。

本地 Windows PyTorch 报告未编译 FlashAttention。`dense_sdpa_auto` 实际使用 CUDA memory-efficient attention；强制 Flash 的失败单独记录，不冒充 FlashAttention 实测。

## 可复用命令

以下在项目根目录执行，`python` 必须指向期望的环境。每次输出目录必须是新的，不覆盖历史结果。

```sh
python -m unittest discover -s tests -p test_chunked_topk_attention.py -v
python scripts/preflight_chunked_topk_cuda.py --output results/NEW-cuda-preflight
python scripts/benchmark_chunked_topk.py --device cuda --dtype float32 --lengths 256 1024 4096 --query-chunk 64 --max-matrix-gib 0.5 --max-seconds 120 --output results/NEW-fp32-benchmark
python scripts/benchmark_chunked_topk.py --device cuda --dtype bfloat16 --lengths 1024 4096 --query-chunk 256 --max-matrix-gib 0.5 --max-seconds 120 --output results/NEW-bf16-benchmark
```

GPU preflight 必须先通过。测试里的 FP16/BF16 条件检查固定选点下的输出和梯度，不是半精度模型质量验证。实际 CUDA 固定检查点质量重放需要配置完整模型依赖，并继续记录 FP32 选点的数值敏感性。

完整模型重放在 CPU 环境已经完成：

```sh
python scripts/verify_chunked_topk_checkpoints.py --output results/NEW-replay --dtype float32 --logit-policy record
python scripts/verify_chunked_topk_checkpoints.py --output results/NEW-replay-fp64 --dtype float64
```

FP32 的 `record` 仅让诊断继续收集所有差异；它不会把失败的 logit 容差标成通过。实际结果状态应为 `predictions_match_but_logits_differ`。

## 计时口径

- 主稀疏计时包括分块精确选点，不能用只测已缓存索引的汇总时间代替。
- 每条件 2 次预热、7 次原始样本，保存中位数、范围、环境和源码快照；CUDA 计时前后同步。
- `prefill_core` 只包含一层注意力核心，不含 Q/K/V 投影、其他网络层或完整 prompt 处理。
- `attention_forward_backward` 含单层前向与反向，不含优化器更新；不是整模型训练步。
- 显存同时给出该进程的基线、峰值与增量。CPU 保存张量统计不是 GPU 峰值显存。
- 本机 GPU 不独占，开始时观测利用率 21%–35%。本轮数据用于工程筛查，不作为论文级性能结论；不直接外推到 A40。
- 时限在操作之间检查，不能中断正在运行的单个 CUDA 调用；矩阵上限是保守估计，不能代替实际峰值统计。

## 当前决策

不新增租卡、不启动长训练。此版本在已测本地配置里没有形成相对优化完整注意力的实用效率优势。扩大卡数不解决当前瓶颈；下一步先在本机研究减少候选打分和调用开销，并在扩大前重新判断具体贡献重合。

若以后使用 Linux 云机，先恢复既有归档，再覆盖本轮新增文件，运行正确性 gate；强制 FlashAttention 必须报告实际成功或失败。旧 SSH 超时不能证明 Pod 已停止或不再计费，旧账单仍需通过控制台核对。
