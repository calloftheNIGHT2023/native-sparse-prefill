# 稀疏64步与密集64步：补齐同预算成本曲线

相同4090、两种子、64更新，全部按密集注意力评测。使用已保存检查点，未新增训练。原权重与已知QK恢复两个固定候选对比密集64原权重；不是独立确认。

|种子|恢复QK|长题正确率|LAMBADA正确率|Wiki困惑度|PG19困惑度|
|---|---|---:|---:|---:|---:|
|2026091660|False|79.69%|52.54%|10.8070|16.9599|
|2026091660|True|88.28%|47.85%|10.7844|16.6865|
|2026091661|False|89.06%|49.80%|10.8064|16.9871|
|2026091661|True|90.62%|48.44%|10.7611|16.6840|

联合筛查：[{"restore_qk": false, "checks": {"word": false, "lambada": true, "wiki": true, "pg_full": true, "pg_tail": true, "cost": true}, "all": false}, {"restore_qk": true, "checks": {"word": false, "lambada": true, "wiki": true, "pg_full": true, "pg_tail": true, "cost": true}, "all": false}]

成本：[{"seed": 2026091660, "restore_qk": false, "early_seconds_utc_plus_restore": 276.191116, "dense64_seconds_utc": 298.015227, "saving_percent": 7.323152987749837, "still_cheaper_with_one_second_penalty": true}, {"seed": 2026091660, "restore_qk": true, "early_seconds_utc_plus_restore": 276.2868856158439, "dense64_seconds_utc": 298.015227, "saving_percent": 7.291017174822434, "still_cheaper_with_one_second_penalty": true}, {"seed": 2026091661, "restore_qk": false, "early_seconds_utc_plus_restore": 276.750971, "dense64_seconds_utc": 298.144055, "saving_percent": 7.175418607625761, "still_cheaper_with_one_second_penalty": true}, {"seed": 2026091661, "restore_qk": true, "early_seconds_utc_plus_restore": 276.84717216418227, "dense64_seconds_utc": 298.144055, "saving_percent": 7.143151935670067, "still_cheaper_with_one_second_penalty": true}]

两准确率非劣容差5pp、三PPL容差5%，两个候选均用97.5%配对区间；种子先平均，按背景/题目/窗口/书籍重采样。成本双方均按UTC至64检查点重载，稀疏另加实际恢复保存；附加1秒成本仍须更便宜。与128步单调计时不同。没有挑选有利指标，未把未知算相同，未证明最优成本。已知恢复基线不作为原创。
