"""CPU tiny checkpoint evaluation tests; synthetic tokens, no optimizer/backward."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts import run_babylm_checkpoint_eval_v0 as runner
from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.model import build_model
from src.babylm_hybrid.evaluation import evaluate_windows


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokens = self.root / "tokens"
        self.tokens.mkdir()
        token_path = self.tokens / "synthetic.ids.u32"
        np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="<u4").tofile(token_path)
        windows = self.root / "windows"
        windows.mkdir()
        rows = [[0, 0, 0, 5, 0, 0, 0, 5, 3, 5, 4],
                [0, 1, 5, 8, 1, 1, 0, 3, 2, 3, 2],
                [0, 2, 8, 9, 2, 2, 0, 1, 1, 1, 0]]
        index = windows / "windows.u64.npy"
        np.save(index, np.asarray(rows, dtype=np.uint64))
        self.manifest_path = windows / "manifest.json"
        manifest = {"status": "window_index_word_accounting_and_coverage_verified",
                    "columns": runner.window_reader.COLUMNS, "total_windows": 3,
                    "tokenizer_sha256": "a" * 64,
                    "artifacts": [{"path": index.name, "sha256": runner.sha(index)}],
                    "source_summaries": [{"source_index": 0, "source": "synthetic.dev",
                                          "token_ids_path_relative_to_project": "tokens/synthetic.ids.u32",
                                          "token_ids_sha256": runner.sha(token_path)}]}
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.patch_root = mock.patch.object(runner.window_reader, "ROOT", self.root)
        self.patch_tok = mock.patch.object(runner.window_reader, "TOK", self.tokens)
        self.patch_dev = mock.patch.object(runner.window_reader, "DEV_TOK", self.tokens)
        for patch in (self.patch_root, self.patch_tok, self.patch_dev):
            patch.start(); self.addCleanup(patch.stop)
        self.cfg = HybridConfig(vocab_size=17, hidden_size=16, intermediate_size=32,
                                layer_types=("global",), global_q_heads=2, global_kv_heads=1,
                                selected_complete_blocks=1)
        self.checkpoint = self.root / "model.pt"
        self.protocol = {"schema_version": 1, "scope": "engineering", "mode": "dense",
                         "model_config": self.cfg.to_dict(), "backbone_seed": 13, "indexer_seed": 17,
                         "checkpoint_path": str(self.checkpoint), "checkpoint_format": "model_only",
                         "dev_manifest": str(self.manifest_path), "dev_manifest_sha256": runner.sha(self.manifest_path),
                         "output_dir": str(self.root / "output"), "device": "cpu", "dtype": "float32",
                         "max_wall_seconds": 60, "window_indices": None, "position_diagnostics": True,
                         "expected_source_hashes": {name: runner.sha(runner.ROOT / name) for name in runner.REQUIRED_SOURCES}}
        self.write_checkpoint()

    def write_checkpoint(self, mode="dense", mutate=None, full=False):
        model = build_model(self.cfg, mode, 13, 17)
        dependency = Path(inspect.getfile(runner.Qwen3NextGatedDeltaNet))
        payload = {"mode": mode, "model_state": model.state_dict(),
                   "protocol": {"model_config": self.cfg.to_dict(), "backbone_seed": 13, "indexer_seed": 17},
                   "source_hashes": {str(runner.ROOT / name): runner.sha(runner.ROOT / name) for name in runner.REQUIRED_SOURCES[1:]}}
        payload["source_hashes"][str(dependency)] = runner.sha(dependency)
        if full:
            payload["optimizer_state"] = {"dummy_unused": "must_not_be_loaded"}
        if mutate:
            mutate(payload)
        torch.save(payload, self.checkpoint)
        self.protocol["checkpoint_sha256"] = runner.sha(self.checkpoint)
        return model

    def read_rows(self):
        return [json.loads(line) for line in (self.root / "output/windows.jsonl").read_text().splitlines()]

    def test_full_dev_matches_existing_evaluator_and_is_read_only(self):
        before = self.checkpoint.read_bytes()
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_complete")
        model = build_model(self.cfg, "dense", 13, 17)
        expected = evaluate_windows(model, self.manifest_path, None, position_diagnostics=True)
        for key in ("nll", "nll_sum", "forward_calls", "windows", "input_tokens", "loss_tokens",
                    "word_exposures", "zero_target_windows", "attention_counts"):
            self.assertEqual(result["total"][key], expected["total"][key])
        self.assertEqual(result["position_diagnostics"]["query_history_bins"], expected["position_diagnostics"]["query_history_bins"])
        rows = self.read_rows()
        self.assertEqual([row["window_index"] for row in rows], [0, 1, 2])
        self.assertTrue(all(not row["grad_enabled"] and not row["model_training"] for row in rows))
        self.assertIsNone(rows[-1]["nll"])
        self.assertEqual(result["backward_calls"], 0)
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(result["counts"]["forward_calls"], 3)
        self.assertEqual(result["requested_window_indices_sha256"], result["observed_window_indices_sha256"])
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_subset_preserves_frozen_order_and_refuses_existing_output(self):
        self.protocol["window_indices"] = [1, 0]
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["observed_window_indices"], [1, 0])
        self.assertEqual(result["counts"]["forward_calls"], 2)
        self.assertFalse(result["full_manifest_requested"])
        with self.assertRaises(FileExistsError):
            runner.run_evaluation(self.protocol)

    def test_wrong_checkpoint_sha_and_source_sha_fail_before_forward(self):
        self.protocol["checkpoint_sha256"] = "0" * 64
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertIn("Checkpoint SHA", result["error"])
        self.assertEqual(result["counts"]["forward_calls"], 0)
        self.protocol["output_dir"] = str(self.root / "bad-source")
        self.protocol["expected_source_hashes"][runner.REQUIRED_SOURCES[0]] = "0" * 64
        result = runner.run_evaluation(self.protocol)
        self.assertIn("Source SHA", result["error"])
        self.assertEqual(result["counts"]["forward_calls"], 0)

    def test_strict_state_dict_rejects_missing_parameter(self):
        self.write_checkpoint(mutate=lambda payload: payload["model_state"].pop("embedding.weight"))
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertIn("Missing key", result["error"])
        self.assertEqual(result["counts"]["forward_calls"], 0)

    def test_window_token_hash_failure_is_persisted(self):
        (self.tokens / "synthetic.ids.u32").write_bytes(b"changed")
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertIn("Token stream SHA", result["error"])
        saved = json.loads((self.root / "output/summary.json").read_text())
        self.assertTrue(saved["partial_metrics_only"])
        self.assertEqual(saved["counts"]["forward_calls"], 0)

    def test_wall_ceiling_fails_without_extra_model_call(self):
        self.protocol["max_wall_seconds"] = 1e-12
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_failed")
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertEqual(result["counts"]["forward_attempts"], 0)

    def test_full_checkpoint_and_sparse_model_are_supported_without_optimizer(self):
        self.protocol.update(mode="sparse", checkpoint_format="full", window_indices=[0])
        self.write_checkpoint(mode="sparse", full=True)
        with mock.patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer allowed")):
            result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_complete")
        self.assertFalse(result["checkpoint"]["optimizer_state_loaded"])
        self.assertEqual(result["counts"]["forward_calls"], 1)

    def test_duplicate_indices_and_runtime_mismatch_are_rejected(self):
        self.protocol["window_indices"] = [0, 0]
        result = runner.run_evaluation(self.protocol)
        self.assertIn("repeated evaluation", result["error"])
        self.protocol.update(output_dir=str(self.root / "runtime-failure"), window_indices=[0],
                             expected_runtime={"torch_num_threads": 999})
        result = runner.run_evaluation(self.protocol)
        self.assertIn("runtime mismatch", result["error"])
        self.assertEqual(result["counts"]["forward_attempts"], 0)

    def test_routing_context_integrates_with_runner_in_one_forward(self):
        self.protocol.update(mode="sparse", window_indices=[0], enable_routing_diagnostics=True,
                             routing_policy="learned", routing_seed=20260920)
        name = "src/babylm_hybrid/routing_diagnostics.py"
        self.protocol["expected_source_hashes"][name] = runner.sha(runner.ROOT / name)
        self.write_checkpoint(mode="sparse")
        result = runner.run_evaluation(self.protocol)
        self.assertEqual(result["status"], "evaluation_complete", result.get("error"))
        self.assertEqual(result["counts"]["forward_calls"], 1)
        self.assertEqual(result["counts"]["backward_calls"], 0)
        self.assertEqual(result["routing_summary"]["schema"], "babylm-routing-diagnostics-v0")
        self.assertEqual(result["routing_summary"]["intervention_policy"], "learned")
        self.assertTrue(result["routing_summary"]["layers"])


if __name__ == "__main__":
    unittest.main()
