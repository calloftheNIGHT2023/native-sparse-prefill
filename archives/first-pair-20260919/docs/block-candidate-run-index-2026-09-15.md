# 本轮逐次日志与时间轴

UTC 起止来自原始结果；单调计时用于耗时。以下均为 0 次优化更新。

| 运行 | 开始 UTC | 结束 UTC | 秒 | 条件/核对 |
|---|---|---|---:|---:|
| [block-candidate-audit-v0](../results/block-candidate-audit-v0/evaluation.json) | 2026-09-15T01:42:38.170302+00:00 | 2026-09-15T01:43:33.450375+00:00 | 55.280 | 114 |
| [block-candidate-confirmation-v0](../results/block-candidate-confirmation-v0/evaluation.json) | 2026-09-15T01:50:50.183372+00:00 | 2026-09-15T01:51:13.111602+00:00 | 22.928 | 54 |
| [block-candidate-diagnostic-correction-v1](../results/block-candidate-diagnostic-correction-v1/evaluation.json) | 2026-09-15T01:51:21.256775+00:00 | 2026-09-15T01:51:23.919713+00:00 | 2.663 | 0 |
| [block-candidate-prefill-v0](../results/block-candidate-prefill-v0/benchmark.json) | 2026-09-15T01:51:26.957260+00:00 | 2026-09-15T01:51:47.025874+00:00 | 20.069 | 24 |
| [block-candidate-fused-gate-v0](../results/block-candidate-fused-gate-v0/verification.json) | 2026-09-15T01:55:27.899764+00:00 | 2026-09-15T01:55:37.589531+00:00 | 9.690 | 30 |
| [block-candidate-fused-replay-v0](../results/block-candidate-fused-replay-v0/evaluation.json) | 2026-09-15T01:55:48.143951+00:00 | 2026-09-15T01:55:59.289621+00:00 | 11.146 | 54 |
| [block-candidate-fused-prefill-v0](../results/block-candidate-fused-prefill-v0/benchmark.json) | 2026-09-15T01:56:02.038222+00:00 | 2026-09-15T01:56:10.390227+00:00 | 8.352 | 24 |
| [block-candidate-fused-bf16-replay-v0](../results/block-candidate-fused-bf16-replay-v0/evaluation.json) | 2026-09-15T01:58:26.543586+00:00 | 2026-09-15T01:58:37.255633+00:00 | 10.712 | 54 |
| [block-candidate-fused-long-prefill-v1](../results/block-candidate-fused-long-prefill-v1/benchmark.json) | 2026-09-15T01:58:40.228849+00:00 | 2026-09-15T01:59:26.526477+00:00 | 46.298 | 24 |

逐配置起止时间与所有计时采样保存在各目录 `events.jsonl`。stdout 保存在同名 `logs/<运行名>.log`。
CPU 5 项测试日志：`logs/block-candidate-unit-tests-v1.log`；原 4 项版本日志保留。
数据准备开始前冻结时间与结束时间：`data/block-candidate-confirmation-v0/preregistration.json`、`data-audit.json`。
v0 计数诊断修正、早期非共同随机输入计时的限制见主报告。
