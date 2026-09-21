# 本轮日志与时间轴

所有时间均为UTC；每步训练另保留train_nll、梯度范数、耗时。

| 阶段 | 开始 | 结束 | 结果 |
|---|---|---|---|
| 原版8K反向 | 2026-09-15T16:00:01.957996+00:00 | 2026-09-15T16:00:04.904870+00:00 | complete |
| 候选8K反向 | 2026-09-15T16:22:49.142629+00:00 | 2026-09-15T16:22:50.756297+00:00 | complete |
| 8K整合训练 | 2026-09-15T16:23:27.468983+00:00 | 2026-09-15T16:24:12.207401+00:00 | complete |

| 轨迹 | 开始 | 结束 | 更新数 |
|---|---|---|---:|
| fp3232a | 2026-09-15T16:23:40.661597+00:00 | 2026-09-15T16:23:48.795881+00:00 | 4 |
| fp3232b | 2026-09-15T16:23:48.817021+00:00 | 2026-09-15T16:23:56.535510+00:00 | 4 |
| fp3264 | 2026-09-15T16:23:56.559248+00:00 | 2026-09-15T16:24:04.330783+00:00 | 4 |
| fp32128 | 2026-09-15T16:24:04.353050+00:00 | 2026-09-15T16:24:12.081353+00:00 | 4 |

完整日志目录：results/flashmoba-backward-environment-v1至v5、flashmoba-backward-stage-controller-v0、flashmoba-long-backward-original-v0、flashmoba-long-backward-barrier-v0、flashmoba-newpod-standard-original-v0、flashmoba-newpod-standard-barrier-v0、flashmoba-backward-timing-*、flashmoba-backward-training-v0。

## 环境事件

- 2026-09-15T15:51:04.198964+00:00：flashmoba-backward-environment-v1 / command_start venv
- 2026-09-15T15:51:16.701083+00:00：flashmoba-backward-environment-v1 / command_end venv
- 2026-09-15T15:51:16.706163+00:00：flashmoba-backward-environment-v1 / command_start pip-tools
- 2026-09-15T15:51:31.551501+00:00：flashmoba-backward-environment-v1 / command_end pip-tools
- 2026-09-15T15:51:31.553948+00:00：flashmoba-backward-environment-v1 / command_start torch
- 2026-09-15T15:53:39.094620+00:00：flashmoba-backward-environment-v1 / command_end torch
- 2026-09-15T15:58:36.341217+00:00：flashmoba-backward-environment-v2 / command_start venv
- 2026-09-15T15:58:38.964117+00:00：flashmoba-backward-environment-v2 / command_end venv
- 2026-09-15T15:58:38.967782+00:00：flashmoba-backward-environment-v2 / command_start pip-tools
- 2026-09-15T15:58:41.191428+00:00：flashmoba-backward-environment-v2 / command_end pip-tools
- 2026-09-15T15:58:41.194572+00:00：flashmoba-backward-environment-v2 / command_start torch
- 2026-09-15T15:59:40.715629+00:00：flashmoba-backward-environment-v2 / command_end torch
- 2026-09-15T15:59:40.718897+00:00：flashmoba-backward-environment-v2 / command_start deps
- 2026-09-15T15:59:44.145776+00:00：flashmoba-backward-environment-v2 / command_end deps
- 2026-09-15T15:59:44.226430+00:00：flashmoba-backward-environment-v2 / command_start baseline-wheel
- 2026-09-15T15:59:44.994696+00:00：flashmoba-backward-environment-v2 / command_end baseline-wheel
- 2026-09-15T15:59:44.997076+00:00：flashmoba-backward-environment-v2 / command_start baseline-import
- 2026-09-15T15:59:46.718411+00:00：flashmoba-backward-environment-v2 / command_end baseline-import
- 2026-09-15T15:59:46.730455+00:00：flashmoba-backward-environment-v2 / failed 
- 2026-09-15T16:01:35.803448+00:00：flashmoba-backward-environment-v3 / command_start baseline-import
- 2026-09-15T16:01:37.775514+00:00：flashmoba-backward-environment-v3 / command_end baseline-import
- 2026-09-15T16:01:42.308634+00:00：flashmoba-backward-environment-v3 / command_start main-restore-missing
- 2026-09-15T16:01:42.827582+00:00：flashmoba-backward-environment-v3 / command_end main-restore-missing
- 2026-09-15T16:01:46.099162+00:00：flashmoba-backward-environment-v3 / command_start cutlass-restore-missing
- 2026-09-15T16:01:58.741295+00:00：flashmoba-backward-environment-v3 / command_end cutlass-restore-missing
- 2026-09-15T16:03:20.540485+00:00：flashmoba-backward-environment-v3 / command_start apply-check
- 2026-09-15T16:03:20.559447+00:00：flashmoba-backward-environment-v3 / command_end apply-check
- 2026-09-15T16:03:20.561185+00:00：flashmoba-backward-environment-v3 / command_start apply
- 2026-09-15T16:03:20.579427+00:00：flashmoba-backward-environment-v3 / command_end apply
- 2026-09-15T16:03:29.710414+00:00：flashmoba-backward-environment-v3 / command_start candidate-build
- 2026-09-15T16:03:31.798267+00:00：flashmoba-backward-environment-v3 / command_end candidate-build
- 2026-09-15T16:03:31.811611+00:00：flashmoba-backward-environment-v3 / failed 
- 2026-09-15T16:04:45.794943+00:00：flashmoba-backward-environment-v4 / command_start candidate-build
- 2026-09-15T16:08:15.913260+00:00：flashmoba-backward-environment-v4 / command_end candidate-build
- 2026-09-15T16:08:15.929098+00:00：flashmoba-backward-environment-v4 / failed 
- 2026-09-15T16:08:27.405379+00:00：flashmoba-backward-environment-v5 / command_start candidate-build
- 2026-09-15T16:22:41.359616+00:00：flashmoba-backward-environment-v5 / command_end candidate-build
- 2026-09-15T16:22:41.368044+00:00：flashmoba-backward-environment-v5 / command_start candidate-import
- 2026-09-15T16:22:43.096074+00:00：flashmoba-backward-environment-v5 / command_end candidate-import
- 2026-09-15T16:22:44.045292+00:00：flashmoba-backward-environment-v5 / complete 
