"""Pure JSON/metadata tests: zero model forwards, backwards, or GPU calls."""
import copy
import math
from pathlib import Path
import unittest

from scripts import run_babylm_scale_full_dev_v0 as wrapper


class FullDevWrapperTests(unittest.TestCase):
    def setUp(self):
        self.template = wrapper.load(wrapper.resolve(wrapper.BASE_TEMPLATE))
        self.master = {"index_score_scale": wrapper.SCALE, "output_dir": "results/test-scale-full-dev",
                       "continuation_master_sha256": "a" * 64}
        self.checkpoint = "results/test-scale-train/run/snapshots/" + wrapper.FINAL_STEM + ".pt"
        self.train_protocol = {"model_config": copy.deepcopy(self.template["model_config"]),
            "backbone_seed": self.template["backbone_seed"], "indexer_seed": self.template["indexer_seed"],
            "max_epochs": 1, "max_updates": 1413}
        self.train_protocol["model_config"]["index_score_scale"] = wrapper.SCALE
        self.training_sha = wrapper.canonical_sha(self.train_protocol)
        self.summary = {"status": "epoch_complete", "mode": "sparse", "counts": copy.deepcopy(wrapper.TRAIN_COUNTS),
                        "eval_counts": {"forward_calls": 336}, "protocol_sha256": self.training_sha,
                        "source_hashes": {"frozen-model.py": "f" * 64}}
        self.audit = {"status": "complete_epoch_continuation_audited", "master_protocol_sha256": "a" * 64,
                      "counts": copy.deepcopy(wrapper.TRAIN_COUNTS), "incremental_counts": copy.deepcopy(wrapper.NEW_TRAIN_COUNTS),
                      "eval_counts": {"forward_calls": 336}, "protocol_sha256": self.training_sha,
                      "final_model_checkpoint_path": self.checkpoint, "final_model_checkpoint_sha256": "b" * 64}
        self.receipt = {"counts": copy.deepcopy(wrapper.TRAIN_COUNTS), "stop_reason": "epoch_complete",
                        "protocol_completion_boundary": True,
                        "point": {key: wrapper.TRAIN_COUNTS[key] for key in ("updates", "word_exposures", "input_tokens", "loss_tokens")},
                        "model_path": "snapshots/" + wrapper.FINAL_STEM + ".pt", "model_sha256": "b" * 64,
                        "protocol_sha256": self.training_sha, "source_hashes": {"frozen-model.py": "f" * 64}}

    def gate(self):
        return wrapper.validate_completion(self.summary, self.audit, self.receipt, self.train_protocol,
                                           self.master, self.template, self.checkpoint, "b" * 64)

    def test_only_scale_checkpoint_admin_changes(self):
        child, changes = wrapper.derive_child(self.template, self.master, self.checkpoint, "b" * 64, self.training_sha)
        self.assertTrue(set(changes) <= wrapper.ALLOWED_CHILD_CHANGES)
        self.assertEqual(child["window_indices"], None)
        self.assertEqual(child["max_wall_seconds"], 3500)
        for key in set(self.template) - wrapper.ALLOWED_CHILD_CHANGES:
            self.assertEqual(child[key], self.template[key])
        self.assertEqual(self.template["model_config"]["index_score_scale"], 1)

    def test_rejects_wrong_scale_or_panel_template(self):
        self.master["index_score_scale"] = 1
        with self.assertRaisesRegex(ValueError, "score scale"):
            wrapper.derive_child(self.template, self.master, self.checkpoint, "b" * 64, self.training_sha)
        self.template["window_indices"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "Full development"):
            wrapper.validate_template(self.template)

    def test_completed_prespecified_point_passes(self):
        self.assertEqual(self.gate(), self.training_sha)

    def test_rejects_incomplete_or_different_point(self):
        self.summary["status"] = "stopped_by_update_limit"
        with self.assertRaisesRegex(ValueError, "completed sparse epoch"):
            self.gate()
        self.summary["status"] = "epoch_complete"
        self.receipt["point"]["updates"] = 1400
        with self.assertRaisesRegex(ValueError, "point differs"):
            self.gate()

    def test_rejects_stale_master_checkpoint_or_partial_counts(self):
        for target, key, bad_value in ((self.audit, "master_protocol_sha256", "c" * 64),
                                       (self.audit, "final_model_checkpoint_sha256", "c" * 64),
                                       (self.audit["incremental_counts"], "updates", 912)):
            old = target[key]
            target[key] = bad_value
            with self.assertRaises(ValueError):
                self.gate()
            target[key] = old

    def test_rejects_unexpected_model_or_source_changes(self):
        self.train_protocol["model_config"]["hidden_size"] += 1
        with self.assertRaises(ValueError):
            self.gate()
        self.train_protocol["model_config"]["hidden_size"] -= 1
        self.receipt["source_hashes"]["frozen-model.py"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "source receipts"):
            self.gate()

    def full_eval_fixture(self):
        manifest = wrapper.load(wrapper.resolve(wrapper.DEV_MANIFEST))
        rows, per_source = [], {}
        for source in manifest["source_summaries"]:
            n = source["windows"]
            input_base, input_remainder = divmod(source["input_tokens"], n)
            word_base, word_remainder = divmod(source["window_word_exposures"], n)
            for index in range(n):
                tokens = input_base + (index < input_remainder)
                rows.append({"window_index": len(rows), "input_tokens": tokens, "loss_tokens": tokens - 1,
                             "word_exposures": word_base + (index < word_remainder), "forward_calls": 1,
                             "grad_enabled": False, "model_training": False, "routing_policy": "learned", "nll_sum": 2.5 * (tokens - 1)})
            per_source[source["source"]] = {"windows": n, "input_tokens": source["input_tokens"],
                "loss_tokens": source["next_token_loss_positions"], "word_exposures": source["window_word_exposures"]}
        summary = {"status": "evaluation_complete", "partial_metrics_only": False, "optimizer_updates": 0, "backward_calls": 0,
                   "counts": {"optimizer_updates": 0, "backward_calls": 0}, "requested_windows": 18792, "full_manifest_requested": True,
                   "window_indices": list(range(18792)), "observed_window_indices": list(range(18792)),
                   "checkpoint": {"sha256": "b" * 64, "optimizer_state_loaded": False},
                   "total": {**wrapper.DEV_COUNTS, "nll": 2.5, "ppl": math.exp(2.5)}, "per_source": per_source,
                   "position_diagnostics": {"protocol": "babylm-dev-position-v0", "extra_model_forwards": 0,
                                            "per_source_query_history_bins": {name: {} for name in per_source}}}
        return summary, rows, manifest, {"checkpoint_sha256": "b" * 64}

    def test_full_dev_audit_requires_exact_complete_ledger(self):
        summary, rows, manifest, child = self.full_eval_fixture()
        result = wrapper.audit_evaluation(summary, rows, manifest, child)
        self.assertEqual(result["status"], "complete_full_dev_audited")
        self.assertEqual(result["nll"], 2.5)
        rows.pop()
        with self.assertRaisesRegex(ValueError, "window order"):
            wrapper.audit_evaluation(summary, rows, manifest, child)

    def test_partial_full_dev_or_extra_training_fails(self):
        summary, rows, manifest, child = self.full_eval_fixture()
        summary["partial_metrics_only"] = True
        with self.assertRaisesRegex(ValueError, "incomplete"):
            wrapper.audit_evaluation(summary, rows, manifest, child)
        summary["partial_metrics_only"] = False
        summary["backward_calls"] = 1
        with self.assertRaisesRegex(ValueError, "training work"):
            wrapper.audit_evaluation(summary, rows, manifest, child)


if __name__ == "__main__":
    unittest.main()
