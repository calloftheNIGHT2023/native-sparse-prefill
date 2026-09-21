# 完整匹配训练与QK恢复：质量—成本复验

同一4090，两种子从初始检查点各做128步密集/K32 QKVO LoRA训练。统一非重入梯度检查点、数据顺序和优化器；这是原有方法的同卡实际成本复验，已用过的评测数据不是新独立测试。

|种子|K|评测权重|长题正确率|Wiki困惑度|PG19子集困惑度|
|---|---:|---|---:|---:|---:|
|2026091660|0|original|79.69%|10.3070|16.5064|
|2026091660|0|restored|82.03%|10.3915|16.3966|
|2026091660|32|original|75.00%|10.7257|16.9536|
|2026091660|32|restored|91.41%|10.7146|16.6714|
|2026091661|32|original|78.91%|10.7235|16.9783|
|2026091661|32|restored|90.62%|10.6954|16.6635|
|2026091661|0|original|92.19%|10.3100|16.5238|
|2026091661|0|restored|85.94%|10.3851|16.4227|

成本：[{"seed": 2026091660, "dense_process_seconds": 578.5965718550142, "sparse_restored_artifact_seconds": 533.8549788771197, "artifact_saving_percent": 7.732778788241057, "step_saving_percent": 8.225865319906678, "loop_saving_percent": 8.130513226685277, "symmetric_restored_saving_percent": 7.746103940230331, "dense_restore_seconds": 0.08357248408719897, "sparse_restore_seconds": 0.05395231815055013}, {"seed": 2026091661, "dense_process_seconds": 578.6953879259527, "sparse_restored_artifact_seconds": 534.5807863301598, "artifact_saving_percent": 7.623112697320755, "step_saving_percent": 8.1812082168371, "loop_saving_percent": 7.963028000732053, "symmetric_restored_saving_percent": 7.630783708674704, "dense_restore_seconds": 0.04805907281115651, "sparse_restore_seconds": 0.020726229064166546}]

预定复验筛查：{"short": true, "accuracy": true, "wiki": true, "pg_full": true, "pg_tail": true, "cost": true, "all": true}

主成本为完整进程到训练完成的时间，加上稀疏模型恢复及适配器写盘时间；双方均不包含最后科研质量评测。训练过程的校准、检查点、预检与初始化计入。为比较恢复前后而保存原权重的诊断性CPU副本不计入部署恢复成本。另报训练步、循环、对称恢复成本。不是完整基座预训练、官方PG19榜单或成本收敛曲线；QK-Restore已有前作。所有原始和恢复后检查点、步日志、逐题/逐书结果、失败、UTC与审计均保存。
