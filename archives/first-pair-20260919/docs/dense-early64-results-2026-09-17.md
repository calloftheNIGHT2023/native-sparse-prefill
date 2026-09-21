# 密集64步早停对照：相同质量是否更便宜

直接评测已保存同4090密集64步检查点，分别原权重和已知QK恢复；对照已有稀疏128步恢复模型，无新增训练。全部题已暴露，属于开发成本曲线控制。

|种子|密集64恢复QK|长题正确率|LAMBADA正确率|Wiki困惑度|PG19困惑度|
|---|---|---:|---:|---:|---:|
|2026091660|False|84.38%|49.02%|10.3390|16.4440|
|2026091660|True|87.50%|47.66%|10.4162|16.3584|
|2026091661|False|92.19%|48.24%|10.3426|16.4673|
|2026091661|True|89.84%|48.63%|10.4136|16.3854|

预定两候选联合筛查：[{"restore_qk": false, "checks": {"word": false, "lambada": true, "wiki": true, "pg_full": true, "pg_tail": true, "cost": true}, "all": false}, {"restore_qk": true, "checks": {"word": false, "lambada": true, "wiki": true, "pg_full": true, "pg_tail": true, "cost": true}, "all": false}]

成本：[{"seed": 2026091660, "restore_qk": false, "early_seconds_utc_plus_restore": 298.015227, "sparse128_seconds_monotonic_plus_restore": 533.8549788771197, "saving_percent": 44.176744848043136, "still_cheaper_with_one_second_penalty": true}, {"seed": 2026091660, "restore_qk": true, "early_seconds_utc_plus_restore": 298.08350705402347, "sparse128_seconds_monotonic_plus_restore": 533.8549788771197, "saving_percent": 44.16395484762633, "still_cheaper_with_one_second_penalty": true}, {"seed": 2026091661, "restore_qk": false, "early_seconds_utc_plus_restore": 298.144055, "sparse128_seconds_monotonic_plus_restore": 534.5807863301598, "saving_percent": 44.228437941676276, "still_cheaper_with_one_second_penalty": true}, {"seed": 2026091661, "restore_qk": true, "early_seconds_utc_plus_restore": 298.2387021490897, "sparse128_seconds_monotonic_plus_restore": 534.5807863301598, "saving_percent": 44.21073301259728, "still_cheaper_with_one_second_penalty": true}]

每个候选同时满足两个准确率最多低5pp、三个PPL最多高5%、两种子更便宜。两候选使用97.5%区间，条件于已有种子和各数据单位；非独立总体证据。64步时长用训练日志UTC到检查点重载的保守时间估计，包含不必执行的重载；128步用单调时钟实测，加恢复保存；报告1秒额外开销敏感性。不同计时来源不能伪称精确同定义。候选都没通过不代表稀疏已达到最优成本。已知恢复基线非原创，固定128步省时事实与同质量最便宜的策略必须区分。
