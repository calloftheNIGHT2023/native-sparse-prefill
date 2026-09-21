"""Tiny CPU verification of transparent administrative resume; no real corpus/GPU."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from scripts import run_babylm_scale_epoch_continuation_v0 as continuation
from src.babylm_hybrid import evaluation, training
from src.babylm_hybrid.config import HybridConfig

ENGINEERING_COUNTS = {"model_forward_calls": 0, "training_forward_calls": 0,
                      "evaluation_forward_calls": 0, "backward_calls": 0,
                      "optimizer_updates": 0, "scientific_model_forward_calls": 0, "gpu_calls": 0}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_administrative_wall_and_cost_fields_can_change(self):
        parent = {"learning_rate": 3e-4, "indexer_learning_rate": 1e-3, "max_updates": 1413,
                  "max_word_exposures": 10001709, "warmup_word_exposures": 1000000,
                  "model_config": {"index_score_scale": 1 / math.sqrt(128)}, "max_wall_seconds": 5300}
        before = copy.deepcopy(parent)
        child = continuation.derive_child(parent, {"max_wall_seconds": 12500, "paid_ceiling_usd": 20})
        self.assertEqual(parent, before)
        self.assertEqual({k: v for k, v in child.items() if k not in continuation.ADMIN_FIELDS},
                         {k: v for k, v in parent.items() if k not in continuation.ADMIN_FIELDS})
        for override in ({"indexer_learning_rate": 1e-4}, {"max_updates": 500}, {"model_config": {}}, {"data_order_seed": 2}):
            with self.assertRaisesRegex(ValueError, "Only administrative"):
                continuation.derive_child(parent, override)

    def test_state_hash_supports_scalar_optimizer_step_and_shape_dtype(self):
        state = {"step": torch.tensor(2.0), "exp_avg": torch.arange(4, dtype=torch.float32)}
        self.assertEqual(continuation.tree_sha(state), continuation.tree_sha(copy.deepcopy(state)))
        for altered in ({**state, "step": torch.tensor(3.0)}, {**state, "step": torch.tensor([2.0])},
                        {**state, "step": torch.tensor(2.0, dtype=torch.float64)}):
            self.assertNotEqual(continuation.tree_sha(state), continuation.tree_sha(altered))

    def test_command_uses_existing_checkpoint_without_new_update_limit(self):
        run = Path("new-directory/run")
        args = continuation.command(Path("child-protocol.json"), run)
        self.assertEqual(args[args.index("--resume-checkpoint") + 1], str(run / "checkpoint.pt"))
        self.assertNotIn("--stop-after-updates", args)
        self.assertEqual(args[args.index("--mode") + 1], "sparse")

    def test_parent_hash_failure_occurs_before_checkpoint_loading(self):
        source = self.root / "source"
        source.mkdir()
        for name in ("events.jsonl", "protocol.json", "summary.json", "checkpoint.pt"):
            (source / name).write_bytes(b"not a checkpoint")
        with mock.patch.object(torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "Parent evidence SHA mismatch"):
                continuation.inspect_parent(source, {name: "0" * 64 for name in
                    ("events.jsonl", "protocol.json", "summary.json", "checkpoint.pt")})
        loader.assert_not_called()

    def test_tiny_administrative_branch_resume_equals_uninterrupted_state(self):
        torch.set_num_threads(1)
        ledger = self.root / "synthetic-ledger.json"
        ledger.write_text('{"synthetic":true}')
        manifest = self.root / "synthetic-dev.json"
        manifest.write_text(json.dumps({"total_windows": 1, "source_summaries": [{"source_index": 0, "source": "synthetic"}]}))
        class Dataset:
            fingerprint = {"synthetic_engineering_only": True}
            def __len__(self): return 6
            def window(self, index):
                return {"window_index": index, "input_ids": np.asarray([(index + j) % 17 for j in range(8)], dtype=np.uint32),
                        "word_exposures": 4, "input_tokens": 8, "next_token_loss_positions": 7, "loss_tokens": 7,
                        "source_index": 0, "segment_index_in_source": index, "source_token_start": 0,
                        "source_token_end": 8, "single_segment": True, "reset_model_state_before": True}
        dataset = Dataset()
        cfg = HybridConfig(vocab_size=17, hidden_size=16, intermediate_size=32, layer_types=("global",),
                           global_q_heads=2, global_kv_heads=1, selected_complete_blocks=1,
                           index_head_dim=128, index_score_scale=1 / math.sqrt(128))
        p = {"schema_version": 1, "scope": "engineering_smoke", "launch_allowed": True,
             "device": "cpu", "dtype": "float32", "model_config": cfg.to_dict(), "backbone_seed": 13,
             "indexer_seed": 17, "data_order_seed": 19, "train_manifest": str(ledger),
             "train_manifest_sha256": continuation.sha(ledger), "max_epochs": 1, "max_word_exposures": 24,
             "max_updates": 6, "max_wall_seconds": 60, "windows_per_update": 1,
             "learning_rate": 3e-4, "indexer_learning_rate": 1e-3, "weight_decay": 0.1,
             "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip_norm": 1,
             "gradient_clip_scope": "separate_backbone_indexer", "aux_weight": 1,
             "warmup_word_exposures": 4, "min_lr_ratio": 0.1, "checkpoint_every_updates": 1,
             "milestone_word_exposures": [4, 8, 16, 24],
             "eval_every_updates": 1, "eval_initial": True, "eval_final": True,
             "eval_position_diagnostics": True, "eval_manifest": str(manifest),
             "eval_manifest_sha256": continuation.sha(manifest), "eval_window_indices": [0]}
        p = json.loads(json.dumps(p))
        child = continuation.derive_child(p, {"max_wall_seconds": 120})
        source, branch, whole = self.root / "source/run", self.root / "branch/run", self.root / "whole/run"
        def run(protocol, directory, previous=None, **kwargs):
            with mock.patch.object(training, "load_window_dataset", return_value=dataset), \
                 mock.patch.object(evaluation, "iter_windows", side_effect=lambda *a, **kw: iter([dataset.window(0)])):
                result = training.train_run(protocol, "sparse", directory, **kwargs)
            old_train = previous["counts"]["forward_calls"] if previous else 0
            old_eval = previous["eval_counts"]["forward_calls"] if previous else 0
            old_back = previous["counts"]["backward_calls"] if previous else 0
            old_updates = previous["counts"]["updates"] if previous else 0
            train_f = result["counts"]["forward_calls"] - old_train
            eval_f = result["eval_counts"]["forward_calls"] - old_eval
            ENGINEERING_COUNTS["training_forward_calls"] += train_f
            ENGINEERING_COUNTS["evaluation_forward_calls"] += eval_f
            ENGINEERING_COUNTS["model_forward_calls"] += train_f + eval_f
            ENGINEERING_COUNTS["backward_calls"] += result["counts"]["backward_calls"] - old_back
            ENGINEERING_COUNTS["optimizer_updates"] += result["counts"]["updates"] - old_updates
            return result
        partial = run(p, source, stop_after_updates=2)
        files = {path.relative_to(source).as_posix(): continuation.sha(path) for path in source.rglob("*") if path.is_file()}
        raw_prefix = (source / "events.jsonl").read_bytes()
        receipt = continuation.prepare_branch(source, branch, child, files, expected_updates=2)
        self.assertEqual(receipt["status"], "administrative_branch_verified")
        self.assertEqual(receipt["changed_protocol_fields"], {"max_wall_seconds": {"before": 60, "after": 120}})
        self.assertFalse((branch / "summary.json").exists())
        self.assertEqual((branch / "events.jsonl").read_bytes(), raw_prefix)
        for name, value in receipt["historical_copied_file_sha256"].items():
            self.assertEqual(continuation.sha(branch / name), value)
        with self.assertRaisesRegex(ValueError, "Scientific|scientific"):
            continuation.prepare_branch(source, self.root / "forbidden/run", {**child, "learning_rate": 9e-4}, files, expected_updates=2)
        resumed = run(child, branch, previous=partial, resume_checkpoint=branch / "checkpoint.pt")
        continuous = run(child, whole)
        self.assertEqual(resumed["status"], "epoch_complete")
        left = torch.load(branch / "checkpoint.pt", map_location="cpu", weights_only=False)
        right = torch.load(whole / "checkpoint.pt", map_location="cpu", weights_only=False)
        for key in ("model_state", "optimizer_state", "python_rng", "numpy_rng", "torch_rng", "cuda_rng",
                    "counts", "counters", "cursor", "last_lr", "last_eval_point", "initial_parameter_hashes", "eval_counts"):
            self.assertEqual(continuation.tree_sha(left[key]), continuation.tree_sha(right[key]), key)
        continuation.verify_files(source, files)
        self.assertEqual((branch / "events.jsonl").read_bytes()[:len(raw_prefix)], raw_prefix)
        events = [json.loads(line) for line in (branch / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["counts"]["updates"] for e in events if e["type"] == "update"], list(range(1, 7)))
        self.assertEqual([e["counts"]["updates"] for e in events if e["type"] == "evaluation"], list(range(7)))
        self.assertEqual(sum(e["type"] == "resume" for e in events), 1)
        self.assertEqual([e["status"] for e in events if e["type"] == "run_stop"], ["stopped_by_update_limit", "epoch_complete"])
        self.assertEqual(resumed["counts"], continuous["counts"])


if __name__ == "__main__":
    unittest.main()
