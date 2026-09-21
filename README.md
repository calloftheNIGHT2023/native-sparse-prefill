# Native sparse prefill 研究快照

当前结果：从随机初始化、全参数、第一步固定局部稀疏注意力开始，完成两个配对初始化种子的 BabyLM 一轮训练。W 在两个种子的完整开发集上都低于密集 D 的 NLL；PPL 分别为 50.07 对 51.33、50.56 对 54.40。这是小型混合模型上固定局部基线的质量证据，不是新路由方法、完整 Qwen 架构复现或论文创新已成立。

另已完成约 44 秒的小型工程诊断：在固定的八窗口样本和共享 GPU 上，GDN 占仪器化前向流耗时约 82.5%（D）与 80.1%（W）。这只是定位开销：不代表全训练前向加反向占比，也不能据此宣称端到端加速或成本降低。诊断共 36 次前向、32 次反向、0 次参数更新，与科学训练分开记账。

Current work: random-initialized, full-parameter BabyLM pretraining in a small 95.391M-parameter hybrid model. D uses dense global attention; W uses the fixed most-recent 64 complete blocks (4 tokens per block) plus the original causal tail. W has no learned indexer or auxiliary routing objective.

## Verified development-set results

Each arm trained for one epoch: 10,001,709 word exposures, 16,325,414 input tokens, 1,413 updates. Evaluation covers all 18,792 fixed dev windows and 17,418,742 target tokens. Lower is better.

| Backbone initialization seed | D NLL | W NLL | D PPL | W PPL | W minus D NLL |
|---|---:|---:|---:|---:|---:|
| 20260917 | 3.938299515 | 3.913459397 | 51.331239 | 50.071871 | -0.024840118 |
| 20260921 | 3.996446633 | 3.923130225 | 54.404487 | 50.558457 | -0.073316408 |

The fixed-local baseline has lower dev NLL in both matched initializations. Data order is held fixed; this is not two fully independent dataset/optimizer sweeps. The dev set was already used during development. Two seeds do not establish statistical equivalence, broad generalization, a novel method, or paper-level sufficiency.

**No acceleration or cost reduction is established.** Both implementations allocate full attention-score reference tensors. Hardware migration and shared-GPU timing are not matched speed evidence. The latest host price is unknown and monetary cost remains null. Failed earlier work and the interrupted initial three-hour allocation remain in the reports.

## Inspect and verify

Run `python scripts/verify_github_evidence_v1.py` from this snapshot. It uses only the Python standard library: hash checks, decompression, original raw-counter and token-weighted metric recomputation, and the unchanged historical verifier. It performs no model calls, downloads, training, or network operations.

- `evidence/seed-20260921/`: 45 hash-verified terminal text artifacts; raw events and dev windows are gzipped.
- `evidence/seed-20260917/`: first-seed D dev rows and the disjoint W prefix/remainder rows.
- `evidence/reports/`: detailed training, quality, execution, and cost audits with their limitations.
- `archives/first-pair-20260919/`: prior published first-pair evidence with its original sources and unchanged verifier. This earlier ten-pass experiment is historical, not the current one-epoch result.
- `EXPORT_MANIFEST.json`: original-byte SHA, exported-byte SHA, and redaction/compression accounting.

## Reproduction boundary

This is an auditable source-and-text-evidence package, **not a complete runnable reproduction bundle**. Raw/tokenized BabyLM corpora, model weights, environment binaries, private keys, API tokens, and live connection settings are excluded. Scientific source bytes are preserved. Exported protocols and metadata may be redacted; embedded source/original-file hashes refer to pre-redaction evidence and must not be treated as hashes of the exported JSON. Cloud launch authorization and connection files are not provided. The public verifier never loads model tensors and does not establish numerical replay.

The small hybrid-model results do not reproduce or establish claims about the full Qwen architecture. Earlier LoRA/CPT experiments are historical and do not answer random-initialized pretraining.
