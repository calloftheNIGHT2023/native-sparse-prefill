"""Bounded score-scale launcher controls plus one tiny CPU prefix; no real corpus/GPU."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from scripts import run_babylm_score_scale_probe_v0 as probe
from src.babylm_hybrid import training, evaluation
from src.babylm_hybrid.config import HybridConfig

ENGINEERING_COUNTS = {"model_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0,
                      "scientific_model_forward_calls": 0, "gpu_calls": 0}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = probe.load(probe.ROOT / "configs/babylm-one-epoch-blackwell-mig-20260919-v3.json")
        self.master = {"base_protocol": "unused-for-pure-derivation", "base_protocol_sha256": "a" * 64,
                       "indexer_learning_rate": 1e-3, "index_score_scale": 1.0 / math.sqrt(128), "stop_after_updates": 500,
                       "administrative_overrides": {"max_wall_seconds": 3600, "paid_ceiling_usd": 2,
                                                    "hourly_rate_usd": 1, "stage_spent_usd": 0}}

    def test_only_nested_score_scale_and_named_administrative_values_change(self):
        before = copy.deepcopy(self.base)
        child = probe.derive_child(self.base, self.master)
        self.assertEqual(self.base, before)
        for key in self.base:
            if key != "model_config" and key not in self.master["administrative_overrides"]:
                self.assertEqual(child[key], self.base[key], key)
        self.assertEqual(child["indexer_learning_rate"], 1e-3)
        self.assertEqual(child["model_config"]["index_score_scale"], 1.0 / math.sqrt(128))
        self.assertEqual({k for k in child["model_config"] if child["model_config"][k] != self.base["model_config"][k]}, {"index_score_scale"})
        self.assertEqual(child["learning_rate"], 3e-4)
        self.assertEqual(child["max_updates"], 1413)
        self.assertEqual(child["max_word_exposures"], 10001709)
        self.assertTrue(child["probe_execution"]["epoch_is_not_complete"])

    def test_scientific_overrides_or_altered_boundary_are_rejected(self):
        self.master["administrative_overrides"]["learning_rate"] = 1e-4
        with self.assertRaisesRegex(ValueError, "Scientific parameter"):
            probe.derive_child(self.base, self.master)
        del self.master["administrative_overrides"]["learning_rate"]
        self.master["stop_after_updates"] = 1413
        with self.assertRaisesRegex(ValueError, "500 updates"):
            probe.derive_child(self.base, self.master)

    def test_wrong_scale_or_indexer_lr_is_rejected(self):
        self.master["index_score_scale"] = 0.1
        with self.assertRaisesRegex(ValueError, "1/sqrt"):
            probe.derive_child(self.base, self.master)
        self.master["index_score_scale"] = 1.0 / math.sqrt(128)
        self.master["indexer_learning_rate"] = 1e-4
        with self.assertRaisesRegex(ValueError, "learning rates"):
            probe.derive_child(self.base, self.master)

    def test_scale_preserves_fresh_parameter_initialization_without_forward(self):
        from dataclasses import replace
        cfg = HybridConfig(vocab_size=17, hidden_size=16, intermediate_size=32,
                           layer_types=("global",), global_q_heads=2, global_kv_heads=1,
                           index_head_dim=128, selected_complete_blocks=1)
        baseline = training.build_model(cfg, "sparse", 13, 17)
        scaled = training.build_model(replace(cfg, index_score_scale=1.0 / math.sqrt(128)), "sparse", 13, 17)
        self.assertEqual(set(baseline.state_dict()), set(scaled.state_dict()))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, scaled.state_dict()[key]), key)
        self.assertEqual(scaled.layers[0].mixer.indexer.score_scale, 1.0 / math.sqrt(128))

    def test_command_requests_fresh_sparse_prefix_without_resume(self):
        command = probe.command(Path("protocol.json"), Path("new-run"))
        self.assertEqual(command[-2:], ["--stop-after-updates", "500"])
        self.assertEqual(command[command.index("--mode") + 1], "sparse")
        self.assertNotIn("--resume-checkpoint", command)

    def test_timeout_terminates_then_kills_child_and_keeps_same_group(self):
        child = mock.Mock()
        child.pid = 999
        child.wait.side_effect = [subprocess.TimeoutExpired("test", 1), subprocess.TimeoutExpired("test", 15), 0]
        child.poll.side_effect = [None, -9]
        with mock.patch.object(probe.subprocess, "Popen", return_value=child) as popen:
            with self.assertRaises(subprocess.TimeoutExpired):
                probe.supervise(["not-executed"], self.root, 1, {})
        self.assertIs(popen.call_args.kwargs["start_new_session"], False)
        child.terminate.assert_called_once(); child.kill.assert_called_once()

    def test_wrong_gate_pod_is_rejected_without_hardware_probe(self):
        gate = self.root / "gate.json"
        gate.write_text(json.dumps({"schema_version": 1, "status": "passed", "pod_id": "another-pod"}))
        with mock.patch.object(probe.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "another Pod"):
                probe.migration_gate({"gate_receipt": str(gate), "gate_receipt_sha256": probe.sha(gate)}, "current-pod")
        run.assert_not_called()

    def test_tiny_engine_prefix_saves_optimizer_rng_panel_and_full_lr_horizon(self):
        torch.set_num_threads(1)
        ledger = self.root / "synthetic-ledger.json"
        ledger.write_text('{"synthetic":true}')
        manifest = self.root / "dev.json"
        manifest.write_text(json.dumps({"total_windows": 1, "source_summaries": [{"source_index": 0, "source": "synthetic"}]}))
        class Dataset:
            fingerprint = {"synthetic_engineering_only": True}
            def __len__(self): return 6
            def window(self, index):
                ids = np.asarray([(index + j) % 17 for j in range(8)], dtype=np.uint32)
                return {"window_index": index, "input_ids": ids, "word_exposures": 4, "input_tokens": 8,
                        "next_token_loss_positions": 7, "loss_tokens": 7, "source_index": 0,
                        "segment_index_in_source": index, "source_token_start": 0, "source_token_end": 8,
                        "single_segment": True, "reset_model_state_before": True}
        dataset = Dataset()
        cfg = HybridConfig(vocab_size=17, hidden_size=16, intermediate_size=32, layer_types=("global",),
                           global_q_heads=2, global_kv_heads=1, selected_complete_blocks=1,
                           index_head_dim=128, index_score_scale=1.0 / math.sqrt(128))
        p = {"schema_version": 1, "scope": "engineering_smoke", "launch_allowed": True,
             "device": "cpu", "dtype": "float32", "model_config": cfg.to_dict(), "backbone_seed": 13,
             "indexer_seed": 17, "data_order_seed": 19, "train_manifest": str(ledger),
             "train_manifest_sha256": probe.sha(ledger), "max_epochs": 1, "max_word_exposures": 24,
             "max_updates": 6, "max_wall_seconds": 60, "windows_per_update": 1,
             "learning_rate": 3e-4, "indexer_learning_rate": 1e-3, "weight_decay": 0.1,
             "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip_norm": 1,
             "gradient_clip_scope": "separate_backbone_indexer", "aux_weight": 1,
             "warmup_word_exposures": 4, "min_lr_ratio": 0.1, "checkpoint_every_updates": 1,
             "eval_every_updates": 1, "eval_initial": True, "eval_final": True,
             "eval_position_diagnostics": True, "eval_manifest": str(manifest),
             "eval_manifest_sha256": probe.sha(manifest), "eval_window_indices": [0]}
        p = json.loads(json.dumps(p))
        output = self.root / "run"
        with mock.patch.object(training, "load_window_dataset", return_value=dataset), \
             mock.patch.object(evaluation, "iter_windows", side_effect=lambda *a, **kw: iter([dataset.window(0)])):
            result = training.train_run(p, "sparse", output, stop_after_updates=2)
        ENGINEERING_COUNTS["model_forward_calls"] += result["counts"]["forward_calls"] + result["eval_counts"]["forward_calls"]
        ENGINEERING_COUNTS["backward_calls"] += result["counts"]["backward_calls"]
        ENGINEERING_COUNTS["optimizer_updates"] += result["counts"]["updates"]
        order = training.epoch_permutation(19, 0, len(dataset))[:2]
        plan = {"child_protocol": p, "expected_prefix_counts": {"updates": 2, "windows": 2,
                "input_tokens": 16, "loss_tokens": 14, "word_exposures": 8,
                "forward_calls": 2, "backward_calls": 2, "engineering_updates": 2, "scientific_updates": 0},
                "window_ids": [int(i) for i in order], "expected_cursor": {"epoch": 0, "position": 2},
                "expected_evaluation_updates": [0, 1, 2], "expected_evaluation_forwards": 3}
        audit = probe.audit_prefix(output, plan)
        self.assertEqual(audit["status"], "complete_prefix_audited")
        self.assertEqual(audit["counts"]["updates"], 2)
        self.assertEqual(audit["eval_counts"]["forward_calls"], 3)
        self.assertTrue(all(audit["checkpoint_components_present"].values()))
        self.assertEqual(result["status"], "stopped_by_update_limit")
        multiplier = training.word_lr_multiplier(8, p)
        self.assertGreater(multiplier, p["min_lr_ratio"])
        self.assertEqual(result["last_lr"]["backbone"], 3e-4 * multiplier)
        self.assertEqual(result["last_lr"]["indexer"], 1e-3 * multiplier)
        self.assertTrue(all(receipt["protocol_completion_boundary"] is False for receipt in result["snapshot_state"]["final_receipts"]))


if __name__ == "__main__":
    unittest.main()
