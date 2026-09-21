# 第二 Pod 独立 E 启动器

`scripts/run_babylm_sparse_parallel_v0.py` 默认只输出计划。它不执行预飞，不复用预飞权重，不自动恢复，只允许新 Pod 从随机初始化运行一个 sparse 科学子进程。原科学 pair 脚本和训练实现均未修改。

## 冻结协议新增字段

保留第一 Pod 科学协议全部模型、种子、数据、词预算、更新数、优化器、评测窗口/节奏、里程碑和完成边界字段。只修改第二 Pod 身份及其有界成本字段。下面每个证据前缀均需 `<prefix>_path` 和 `<prefix>_sha256`，SHA 是文件原始字节 SHA256：

- `parent_scientific_protocol`
- `numerics`：新 Pod tiny GPU 原阈值通过报告。
- `new_sparse_preflight_protocol`、`new_sparse_preflight_summary`、`new_sparse_preflight_events`：新 Pod 完整候选模型 12 更新工程运行。
- `reference_dense_preflight_protocol`、`reference_dense_preflight_summary`：第一 Pod 已归档 D 工程证据。
- `reference_sparse_preflight_protocol`、`reference_sparse_preflight_summary`、`reference_sparse_preflight_events`：第一 Pod 已归档 E 工程证据。
- `old_sparse_release`：旧 E 调度已解除的固定收据。

其他执行字段：`pod_id` 为新 Pod；`single_pod_ceiling_usd` 和 `paid_ceiling_usd` 相同且不超过 75；`stage_spent_usd` 含新 Pod setup/preflight 已花成本；`hourly_rate_usd` 为正的有限数；`max_wall_seconds <= hard_timeout_seconds_per_run <= 259200`。`launch_allowed=true` 才能 execute。所有路径在执行主机必须可读，旧证据可以保留原绝对 provenance 路径，比较时只消除项目安装根目录差别；新卡 source/data 文件仍逐个实际校验 SHA。

旧 E 解除收据要求：

```json
{
  "schema_version": 1,
  "kind": "old_sparse_schedule_release",
  "parent_scientific_protocol_sha256": "第一 Pod 原科学协议文件 SHA256",
  "old_pod_id": "第一 Pod ID",
  "new_pod_id": "第二 Pod ID",
  "old_sparse_schedule_disabled": true,
  "old_sparse_not_running": true,
  "verified_utc": "实际核验 UTC"
}
```

这是一份被 SHA 固定的调度审计声明，不能冒称密码学签名或实时远端证明；控制器负责在生成声明前实际核验。启动器在预飞验证前后各读一次固定收据。

## 校验与报告

新卡 tiny 报告必须保持 CPU/CUDA FP32 `atol=1e-4, rtol=1e-3`，全支持/replay `atol=1e-6, rtol=1e-5`，顶层及全部内层阈值不变，7 项全部通过，执行计数必须为原 9 F / 9 B / 3 工程更新。

旧 D/E 和新 E 的共同主干初始化、源码内容、数据内容及 12 更新计数一致；新旧 E 的全部索引器初始化也一致。12 个更新窗口顺序、cursor、LR、词位置与 target 数逐条严格相同。逐更新 CE、aux、合并 loss、global/backbone/indexer 梯度范数及裁剪系数共 84 项使用冻结的跨设备 FP32 容差比较并写进 stage.json，失败也保留。这是标量轨迹重放，不证明每个梯度分量一致，更不能把两块 GPU 的耗时混作同一硬件加速结果。

启动前要求新 Pod 环境 ID、GPU UUID 和驱动与新 tiny 收据相同；持有现有 pair/preflight 锁及独立 sparse 锁；输出必须为空。仅启动 `run_babylm_de_v0.py --mode sparse`，不带 resume。墙钟和成本均包含验证及训练期间费用，并扣除已记录 setup；进程退出不会停止 Pod 计费。

## 本机模拟验证

首轮 7/7 通过，补充内层阈值绑定后仅定向复测成功流程和阈值/收据/GPU 拒绝路径，2/2 通过。每次独立日志和 source SHA 位于 `results/babylm-sparse-parallel-mock-v0-*.json` 与同名 logs。最近一次汇总为 `results/babylm-sparse-parallel-mock-v0.json`。

所有子进程调用均被 mock，真实模型 F/B、optimizer 更新、科学训练、GPU 工作和真实训练子进程均为 0。测试包含旧调度未解除、配方/协议 SHA/预算篡改、内层失败及放宽阈值、新卡 UUID 不符、最后一次 loss 超阈值、索引器初始化或数据顺序不同、锁/输出占用以及子进程 timeout 保存证据。
