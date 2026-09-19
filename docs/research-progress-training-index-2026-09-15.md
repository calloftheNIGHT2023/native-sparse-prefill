# 当前AMP主实验：15条训练与日志索引

本附件从三个本地证据镜像的原始 result.json 自动生成。仅统计完成的正式训练，不包括预检、评测或更早的小模型/2K/8K阶段。

所有时间为2026年9月15日UTC；美国东部时间减4小时，北京时间加8小时。表内开始/结束采用模型进程记录，可能比控制器启动晚几秒。每条原始结果含256条逐步记录（损失、梯度范数、学习率、窗口编号、耗时及UTC）。

|序号|注意力|种子|峰值LR|开始UTC|结束UTC|更新|训练秒|进程秒|原始日志|
|---|---|---:|---:|---|---|---:|---:|---:|---|
|1|密集|2026091560|3e-05|18:22:14|18:27:45|256|318.014|330.228|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr3e-05/result.json)|
|2|K4|2026091560|3e-05|18:27:49|18:32:17|256|251.052|267.729|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k4-lr3e-05/result.json)|
|3|K16|2026091560|3e-05|18:32:22|18:37:28|256|289.667|306.326|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k16-lr3e-05/result.json)|
|4|K4|2026091560|0.0001|18:37:33|18:42:00|256|251.502|267.291|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k4-lr0.0001/result.json)|
|5|K16|2026091560|0.0001|18:42:04|18:47:11|256|289.717|307.061|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k16-lr0.0001/result.json)|
|6|密集|2026091560|0.0001|18:47:15|18:52:51|256|319.248|335.304|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr0.0001/result.json)|
|7|K16|2026091560|0.0003|18:52:55|18:58:01|256|289.630|306.134|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k16-lr0.0003/result.json)|
|8|密集|2026091560|0.0003|18:58:05|19:03:39|256|318.232|333.929|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr0.0003/result.json)|
|9|K4|2026091560|0.0003|19:03:43|19:08:09|256|250.672|265.991|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k4-lr0.0003/result.json)|
|10|K16|2026091561|0.0003|19:08:13|19:13:20|256|289.758|306.660|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/repeat-k16/result.json)|
|11|K4|2026091561|0.0003|19:13:24|19:17:51|256|250.933|266.533|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/repeat-k4/result.json)|
|12|密集|2026091561|0.0003|19:17:55|19:23:29|256|318.195|334.142|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/repeat-k0/result.json)|
|13|密集|2026091560|0.001|20:43:21|20:48:49|256|316.566|328.519|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-lr-boundary-evidence-v0/results/amp-recovery-stage-v0/cal-k0-lr0.001/result.json)|
|14|K16|2026091560|0.001|20:48:53|20:53:59|256|288.943|305.685|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-lr-boundary-evidence-v0/results/amp-recovery-stage-v0/cal-k16-lr0.001/result.json)|
|15|K16|2026091561|0.001|21:11:22|21:16:24|256|288.764|301.838|[result.json](D:/ChatGPT/projects/native-sparse-prefill/results/cloud-amp-confirmation-evidence-v1/results/amp-recovery-stage-v0/repeat-k16/result.json)|

## 统计口径与复核

- 正式轨迹：15；正式更新：3840；逐步日志：3840条。
- 训练循环累计：72.181569分钟。
- 训练进程累计：76.056139分钟。
- 密集5条、K4四条、K16六条。K16包括旧LR=0.0003第二种子复跑，不能因最终未选中而从历史开销中删除。
- 43次预检/诊断更新按阶段记录另列，不包含在3840条正式逐步日志中。
- 计时不等于账单；没有累计安装、独立评测、迁移、空闲和存储费用。
- 本次逐条检查了256个连续step编号，并将逐步耗时之和与training_seconds复核；原始结果及使用的报告文件SHA256在清单中。

## 原始结果位置与SHA256

|序号|项目相对路径|SHA256|
|---|---|---|
|1|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr3e-05/result.json`|`62e789853fccf3843c37a58a25366a13e395c68992f3218f41e0c090ff58232e`|
|2|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k4-lr3e-05/result.json`|`fd35b44e317b25887d51b4d2cf2f9356c12f5b3bc50f632af10cab50e108d901`|
|3|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k16-lr3e-05/result.json`|`a6ccd049c6defe3918bddf1d5d1453de6180e0726a16465e712485d6329fb5fd`|
|4|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k4-lr0.0001/result.json`|`ed172822cd37fb65965c26dd8b8581b886b02089715318f9c4c5f1ca18555167`|
|5|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k16-lr0.0001/result.json`|`4e050cb0cd3d073248fbb47f8f9f26f50ec5c209df95f2e7d5793f13e5fc663c`|
|6|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr0.0001/result.json`|`cb94d08a256efb04a9b5fc6063f93717d441cdf53e77dfc39a1c2f24b6cda488`|
|7|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k16-lr0.0003/result.json`|`8019e4e489a133420ad0933f2a087f2b5164aee341b8e5363f65887d7caf0fef`|
|8|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k0-lr0.0003/result.json`|`4409d8445bb3683091e9b5df7c95a04aeab1ebc3bd5fa882d704bedeb527b574`|
|9|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/cal-k4-lr0.0003/result.json`|`6c39e139f92ac7a12cb0263299abd2ecbc8b90a4bcabe310b5b8417b286d77d4`|
|10|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/repeat-k16/result.json`|`f03ee6879dc2df49164ee9ad515a9019a0e27d5dd0494de94595acac2a38ca4d`|
|11|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/repeat-k4/result.json`|`bad025b16a0b4292e282b1507d9911044a4bf6676209e5cd32314118f8fbd91f`|
|12|`results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2/repeat-k0/result.json`|`f426c2b79e74246b361daf49483a55328bdcb08b92f22b74973bbdc7d4393620`|
|13|`results/cloud-amp-lr-boundary-evidence-v0/results/amp-recovery-stage-v0/cal-k0-lr0.001/result.json`|`64f9d0d2d5759a0f16531890e5af66866f34c6f1e57798dd4ea78fc2f0b72a85`|
|14|`results/cloud-amp-lr-boundary-evidence-v0/results/amp-recovery-stage-v0/cal-k16-lr0.001/result.json`|`805e5b4db98768a69cc9a471d5b60af7aa6b0db39f49c3148a8dc85c9c993aa9`|
|15|`results/cloud-amp-confirmation-evidence-v1/results/amp-recovery-stage-v0/repeat-k16/result.json`|`d5d5d7d5d7e366e698bb44800dcca917cff93c04e1a014a02aa65c7dde8fd9c1`|

[回到阶段报告](D:/ChatGPT/projects/native-sparse-prefill/docs/research-progress-report-2026-09-15.md)
