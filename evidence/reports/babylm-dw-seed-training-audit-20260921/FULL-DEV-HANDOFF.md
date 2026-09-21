# 新种子 D/W 完整开发集独立审计交接

此文件仅准备审计，不运行模型、远端连接、采集器或评测。当前已有 D/W 训练日志审计通过；完整 dev 结果必须另行等原始证据齐全后核验。不得用最终48窗面板代替。

## 固定身份与未来证据入口

- 项目根：`D:/ChatGPT/projects/native-sparse-prefill`。
- 本轮远端逻辑输出根：`results/babylm-dw-seed-confirmation-20260921-v1`。
- 控制器：`scripts/run_babylm_dw_seed_confirmation_v1.py`，SHA `569d183c27a16bae53dee466b5c80987b0706a050dec1668caf9048fc4cef1af`。
- 新主干 seed `20260921`，数据顺序 seed `20260919`；D/W 初始 named-parameter map canonical SHA `3f26ce064cc74c3f9544ec8e2fd3ef6a8489096a37bb1ce3c51701d07cb4792c`。
- 条件是 `D_dense` / `W_fixed_local`。两者 scorer `mode=dense` 是兼容字段；W 必须经 fixed-local factory 且没有 indexer/aux，不可根据该字段误称两个密集模型。
- 精确 master 字节 SHA 以新最终采集收据为准；控制器 `master_protocol_sha256` 使用 canonical JSON SHA，不能和 JSON 文件字节 SHA 混淆。

完成后 root 的已有采集入口是 `logs/collect_dw_arm_backup_20260921.py --arm D` / `--arm W`。该命令会连接远端，本交接不执行。

预计本机采集目录：

- `logs/babylm-dw-seed-D-backup-20260921-v1/`
- `logs/babylm-dw-seed-W-backup-20260921-v1/`

读取各目录 `manifest.json` 获取 immutable `latest_manifest` 与 `latest_manifest_sha256`；先验证 manifest 字节 SHA、`status=collection_complete_all_files_sha_verified`。之后始终用 immutable manifest 的 `receipts[].local_relative_path` 相对该 manifest 父目录定位原始文件，并逐文件核 SHA、size、receipt status。不要读可能过时的普通同名路径；不要把 `.versions/` 中不同采集代的数据随意拼接。

在本次准备时，已有 D checkpoint 采集状态为 `collection_failed_evidence_preserved`，不满足完整备份门槛。root 处理后续采集；不能因本机有训练 summary 就宣称权重备份成功。

每个 arm A=D/W 必需的逻辑文件如下（每个 suffix 恰好匹配一项）：

```text
results/babylm-dw-seed-confirmation-20260921-v1/A/train/run/protocol.json
results/babylm-dw-seed-confirmation-20260921-v1/A/train/run/events.jsonl
results/babylm-dw-seed-confirmation-20260921-v1/A/train/run/summary.json
results/babylm-dw-seed-confirmation-20260921-v1/A/train/audit.json
results/babylm-dw-seed-confirmation-20260921-v1/A/bound-eval.json
results/babylm-dw-seed-confirmation-20260921-v1/A/checkpoint-binding.json
results/babylm-dw-seed-confirmation-20260921-v1/A/eval-worker.json
results/babylm-dw-seed-confirmation-20260921-v1/A/eval-audit.json
results/babylm-dw-seed-confirmation-20260921-v1/A/eval/protocol.json
results/babylm-dw-seed-confirmation-20260921-v1/A/eval/summary.json
results/babylm-dw-seed-confirmation-20260921-v1/A/eval/windows.jsonl
```

唯一 checkpoint 是各 arm 的 `train/run/snapshots/model-u00001413-w000010001709-i000016325414-l000016302816.pt`，以及相应 `-final-epoch_complete.json` receipt。核实已下载文件 SHA 与训练 audit、绑定收据和 scorer checkpoint provenance 全链一致；如果大文件尚未下载，只能报告日志/元数据审计，不得声称权重已本机验证。无需为数学指标复算加载张量。

## 已有可复用代码及调用边界

目前没有可直接覆盖此新 seed 的独立端到端审计 CLI。不要直接运行以下旧脚本 main：它们固定旧 checkpoint、旧 collection id 或旧 W 前缀/剩余分段。

1. `results/babylm-stage-c2-final-20260921/recompute_metrics.py`：可直接复用纯标准库 `Audit.dev(mode, windows_path, summary_path)`，返回 `(metrics, rows, summary)`，重算总量、六来源、四位置桶及来源×位置，同时检查完整性、有限/零目标、原始分数和 summary 对应。也可复用 `compare(a,b)`。**不要调用其 `main()` 或 `Audit.collection()`，它们绑定旧 C2 collection。**
2. `results/babylm-stage-w-complete-20260921/audit_union.py`：可复用纯标准库 `Audit.collection(manifest_path, expected_sha)` 进行 immutable manifest/receipt SHA/size检查；`metric/aggregate/difference` 也是纯统计函数。**不要调用其 `run()` 或 `main()`，它们固定旧 W 12402+6390 分段。**
3. `scripts/run_babylm_dw_seed_confirmation_v1.py:audit_eval` 是本轮生产侧检查，其结果为 `complete_seed_full_dev_audited`，但它不是独立审计替代品。

需要新最终审计时，用 `importlib.util.spec_from_file_location` 载入前两脚本并调用上述已有纯函数；不改冻结文件、不导入模型、不重跑评测。建议只新增本轮结果目录中的小胶合入口，负责下面身份/原始数据/配对检查，数学聚合复用 `Audit.dev`。本交接未创建或运行该入口，故没有假造一个“可运行且已通过”的最终命令。

## 完成后必须检查的项目

1. **完整终态。** 两个 `eval-worker.json.status=complete`；两个 `eval-audit.json.status=complete_seed_full_dev_audited`；scorer `status=evaluation_complete` 且 `partial_metrics_only=false`。stage 最终应为 `complete_pending_independent_local_audit`。若只有一组结束，审计该组并明确另一组未完成，不给完整 D/W 对比结论。
2. **模型/科学身份。** 各 bound-eval 必须等于实际 `eval/protocol.json`；其 canonical SHA 匹配 summary 与 eval-audit。最终模型 checkpoint SHA、train protocol canonical SHA、训练 audit 字节 SHA、1413终点、master/template/source/GDN SHA逐链匹配。新 seed 是20260921，不能错用旧seed20260917模型。D/W共同初始map/data-order/config/LR已在 `pair-training-audit.json`核验，还要与最终采集的原始训练日志 SHA一致。
3. **同一数据。** 两组 dev manifest SHA固定为 `baa53c08d1c26e6dda2f238d2221a1d2cfef4c754ec46327765987f423519619`，数据 artifacts/tokenizer 的 fingerprint 一致。两组原始 JSONL 必须各18792行，IDs严格按 `0..18791`、唯一且完整；按固定 `data/babylm-dev-windows-v0/windows.u64.npy` 逐行校验来源、source_index、segment_index、source_token_start/end、word_exposures、input_tokens、loss_tokens。D/W这些字段还必须逐行全等。
4. **分母/前向。** 每组 `18792F / 0B / 0updates`，attempted=completed=committed，不得漏记在途失败F。每组输入 token `17437534`、目标 token `17418742`、词暴露 `10420962`。行内 `loss_tokens=input_tokens-1`，零目标窗保留 `nll/ppl=null`、`nll_sum=0`；grad_enabled/model_training 均false。训练/面板/旧失败/完整dev分别记账。
5. **位置分母。** 固定 query-history 桶 `1-256,257-512,513-1024,1025-2048`；对targets=T，每桶目标数为 `max(0,min(T,hi)-lo+1)`。来源token起止由冻结窗口索引决定，不能用重打包/截断数据混替。位置CE与scalar CE的浮点reduction区别沿旧audit容差处理，不改模型重放阈值。
6. **token加权复算。** `NLL = math.fsum(row.nll_sum) / sum(row.loss_tokens)`；`PPL=exp(NLL)`。不得平均每窗PPL冒充总体PPL。与summary、生产audit核对，并按六来源、四位置、来源×位置复算（共用既有 `Audit.dev`）。所有NaN/Inf拒绝；空分母指标明确null。
7. **独立配对差值。** 固定方向 `W-D`：NLL差 `NLL_W-NLL_D`，PPL绝对差 `PPL_W-PPL_D`，相对差 `expm1(NLL_W-NLL_D)`。总量及各来源/位置保留真实分母；只做预定描述性比较，不事后挑窗口/种子/断点，不把不显著视为等价。
8. **解释限制。** 两个独立种子并不足以自动成立统计非劣结论；本轮属于已有dev上的追加种子验证。逻辑稀疏和参考完整score分配不是端到端加速证据。unknown-rate继续null，不能给虚构美元成本。新完整dev37584F另记，不混入训练45196F/B或672面板F；v0失败317F/316B（317B attempts）/19updates+48初始panelF另保留。

建议最终输出另建 `results/babylm-dw-seed-full-dev-audit-20260921/`，保存采集manifest身份、原始SHA清单、零模型审计计数、总/分组D-W指标与失败列表。不要覆盖本目录已通过的 D/W 训练报告。
