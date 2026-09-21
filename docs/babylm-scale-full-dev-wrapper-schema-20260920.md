# Scale-only final-epoch full-development worker

This worker evaluates only the prespecified update-1413 model after the continuation count/source audit succeeds. It never selects checkpoints by score and never trains. It derives the child protocol from the byte-identical Stage A `full-dev-sparse.json`, preserving all data, scoring, precision, seed, and original evaluator source pins.

## Master example

Write this as `configs/babylm-stage-c1-scale-epoch-20260920-v0/eval.json`. Replace the indicated administrative hashes and hardware identities before freezing. Expand `source_sha256` with **every** `expected_source_hashes` entry from the original Stage A template, plus the new worker SHA. `training_protocol_sha256` is optional; if supplied it is the canonical child training protocol hash, not the train-master file hash.

```json
{
  "schema_version": 1,
  "launch_allowed": true,
  "base_template": "configs/babylm-optimization-stage-a-20260920-v0/full-dev-sparse.json",
  "base_template_sha256": "7bc1fd79a2007299aaaa828f0e102fc7f556f6fa33b94236ee349a68d8b66802",
  "continuation_master": "configs/babylm-stage-c1-scale-epoch-20260920-v0/train.json",
  "continuation_master_sha256": "REPLACE_WITH_TRAIN_MASTER_FILE_SHA256",
  "continuation_audit_path": "results/babylm-stage-c1-scale-epoch-20260920-v0/train/continuation-audit.json",
  "training_run_dir": "results/babylm-stage-c1-scale-epoch-20260920-v0/train/run",
  "index_score_scale": 0.08838834764831843,
  "output_dir": "results/babylm-stage-c1-scale-epoch-20260920-v0/full-dev",
  "soft_timeout_seconds": 3500,
  "hard_timeout_seconds": 3600,
  "hourly_rate_usd": 1.4,
  "stage_cost_cap_usd": 1.4,
  "execution_hardware": {
    "mig_uuid": "MIG-REPLACE_WITH_UUID",
    "physical_uuid": "GPU-REPLACE_WITH_UUID",
    "mig_profile": "2g.48gb",
    "driver_version": "REPLACE_WITH_FROZEN_DRIVER"
  },
  "source_sha256": {
    "EXPAND_ALL_ORIGINAL_TEMPLATE_SOURCE_PINS": "REQUIRED",
    "scripts/run_babylm_scale_full_dev_v0.py": "REPLACE_WITH_WORKER_FILE_SHA256"
  }
}
```

The scale value is exactly Python `1.0 / math.sqrt(128)`. The stage has a one-hour cost envelope at the conservative combined rate; this is a bound, not an invoice. The overall train/evaluation queue separately enforces its batch budget.

## Launch and process ownership

`--protocol <eval.json> --protocol-sha256 <file SHA>` performs source/template validation with zero model calls. Add `--execute` only under the root queue's external 3600-second process-group deadline. The worker takes `/tmp/babylm-one-epoch-<MIG UUID>.lock` itself; the root queue must not hold the same lock. The old evaluator inherits the worker's process group. On a soft timeout or signal, the worker terminates and, if necessary, kills its child. The outer deadline remains mandatory to bound uninterruptible or wrapper failures.

## Fixed input and dynamic binding

The only accepted final model is `train/run/snapshots/model-u00001413-w000010001709-i000016325414-l000016302816.pt`, with its `-final-epoch_complete.json` receipt. Before loading it, the worker verifies `epoch_complete`, the continuation audit, 1413 cumulative scientific updates and 22598 training forwards/backwards, 913 incremental updates and 14598 incremental forwards/backwards, and 336 cumulative monitoring-panel forwards. The final checkpoint SHA is computed after completion and must agree with both final receipt and continuation audit. All binding inputs receive immutable hashes in `full-dev/source-binding-receipt.json`.

The derived child changes only the model's score scale and administrative checkpoint/protocol/output/wall-time bindings. Child output is `full-dev/evaluation`; the worker records the exact field diff. The full-development set remains 18792 ordered windows, 17437534 input tokens, 17418742 loss targets and six sources, with the original position-scoring protocol. No training occurs. The successful audit checks all window IDs, raw-row SHA, counts, source totals, token-weighted NLL/PPL and zero backwards/optimizer updates.

Primary outputs are `stage.json`, `source-binding-receipt.json`, `evaluation-protocol.json`, `evaluation/summary.json`, `evaluation/windows.jsonl`, and `full-dev-audit.json`. Existing output directories are rejected. Failure evidence is retained; there is no retry, resume, checkpoint fallback, or threshold adjustment. Full dev is development evidence, not held-out confirmation or a sparse-kernel speed measurement.
