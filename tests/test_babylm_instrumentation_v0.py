"""Targeted independent CPU checks for milestone and gradient instrumentation.

Synthetic fixtures only. At most four engine updates plus two independent
optimizer-oracle updates. No historical suite is inherited or executed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.model import HybridLM, build_model
from src.babylm_hybrid import training


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("babylm_previous_fixture_helpers", ROOT / "tests/test_babylm_training_v0.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
ARTIFACT_ROOT = ROOT / "logs/babylm-instrumentation-fixtures-current"
COUNTS = {"model_forward_calls": 0, "backward_calls": 0,
          "engineering_optimizer_steps": 0, "scientific_optimizer_steps": 0,
          "real_training_tokens": 0, "gpu_calls": 0}
MEASUREMENTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_path(directory, entry):
    path = Path(entry["path"])
    return path if path.is_absolute() else directory / path


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class InstrumentationTests(unittest.TestCase):
    protocol = fixtures.TrainingEngineTests.protocol
    run_engine = fixtures.TrainingEngineTests.run_engine

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.directory = ARTIFACT_ROOT / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=True)

    def dataset(self):
        return fixtures.SyntheticWindowDataset(self.directory / "fixture", ((24, 7), (24, 7), (24, 7)))

    def assert_oracle_update(self, mode, dataset, protocol, event, actual_state):
        """Independent flat-vector norm oracle and ordinary optimizer update."""
        model = build_model(HybridConfig(), mode, 731, 991)
        backbone = [p for n, p in model.named_parameters() if ".indexer." not in n]
        indexer = [p for n, p in model.named_parameters() if ".indexer." in n]
        groups = [{"params": backbone, "lr": protocol["learning_rate"]}]
        if indexer:
            groups.append({"params": indexer, "lr": protocol["indexer_learning_rate"]})
        optimizer = torch.optim.AdamW(groups, betas=tuple(protocol["betas"]), eps=protocol["eps"],
                                      weight_decay=protocol["weight_decay"])
        index = int(np.random.default_rng(np.random.SeedSequence([137, 0])).permutation(len(dataset))[0])
        ids = torch.tensor(dataset.rows[index]["input_ids"].astype(np.int64)).unsqueeze(0)
        output = model(ids, aux_weight=protocol["aux_weight"] if mode == "sparse" else 0.0)
        output.loss.backward()

        def flat_norm(parameters):
            gradients = [p.grad.detach().flatten().double() for p in parameters if p.grad is not None]
            return torch.cat(gradients).norm().item() if gradients else 0.0

        norms = {"backbone_grad_norm": flat_norm(backbone), "indexer_grad_norm": flat_norm(indexer),
                 "grad_norm": flat_norm(list(model.parameters()))}
        coefficient = min(1.0, protocol["grad_clip_norm"] / (norms["grad_norm"] + 1e-6))
        self.assertLess(coefficient, 1.0, "The oracle must actually exercise clipping")
        for field, expected in norms.items():
            self.assertAlmostEqual(event[field], expected, delta=max(1e-6, expected * 2e-6))
        self.assertAlmostEqual(event["grad_clip_coefficient"], coefficient, delta=2e-7)
        self.assertEqual(event["grad_clip_max_norm"], protocol["grad_clip_norm"])
        self.assertEqual(event["aux_weight"], protocol["aux_weight"])
        expected_aux_weight = protocol["aux_weight"] if mode == "sparse" else 0.0
        self.assertEqual(event["aux_weight_applied"], expected_aux_weight)
        self.assertAlmostEqual(event["weighted_aux_loss"], output.aux_loss.item()*expected_aux_weight, places=6)
        self.assertEqual(event["gradient_scope"], "combined_objective_before_global_clipping")
        # Standard PyTorch is the independent update oracle, not an engine helper.
        torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["grad_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        worst = 0.0
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(actual_state[name], expected, atol=1e-7, rtol=1e-6)
            worst = max(worst, (actual_state[name] - expected).abs().max().item())
        return {**norms, "expected_coefficient": coefficient,
                "parameter_update_max_abs_difference": worst,
                "oracle_updates": 1, "engine_updates_compared": 1}

    def test_01_sparse_snapshots_threshold_sharing_resume_and_oracle(self):
        dataset = self.dataset()
        protocol = self.protocol(dataset, max_updates=10, max_word_exposures=22,
                                 windows_per_update=1, grad_clip_norm=0.02,
                                 milestone_word_exposures=[5, 6, 14, 22])
        partial, directory = self.run_engine(protocol, "sparse", "run", dataset, stop_after_updates=1)
        state = partial["snapshot_state"]
        self.assertEqual(state["crossed_milestones"], [5, 6])
        self.assertEqual(len(state["models"]), 1)
        first = state["models"][0]
        self.assertEqual(first["nominal_crossed_milestones"], [5, 6])
        self.assertEqual(first["point"], {"updates": 1, "word_exposures": 7, "input_tokens": 24, "loss_tokens": 23})
        first_path = snapshot_path(directory, first)
        self.assertEqual(sha(first_path), first["sha256"])
        original_hash = sha(first_path)
        original_mtime = first_path.stat().st_mtime_ns
        snapshot = load(first_path)
        self.assertIn("model_state", snapshot)
        for forbidden in ("optimizer_state", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
            self.assertNotIn(forbidden, snapshot)
        event = next(e for e in fixtures.read_events(directory) if e["type"] == "update")
        MEASUREMENTS["sparse_gradient_oracle"] = self.assert_oracle_update(
            "sparse", dataset, protocol, event, snapshot["model_state"])
        resumed, _ = self.run_engine(protocol, "sparse", "run", dataset,
                                     resume_checkpoint=directory / "checkpoint.pt")
        self.assertEqual(resumed["counts"]["updates"], 3)
        self.assertEqual(resumed["counts"]["word_exposures"], 21)
        self.assertEqual(resumed["status"], "word_budget_reached")
        state = resumed["snapshot_state"]
        self.assertEqual(state["crossed_milestones"], [5, 6, 14])
        self.assertEqual(sha(first_path), original_hash)
        self.assertEqual(first_path.stat().st_mtime_ns, original_mtime)
        points = [entry["point"]["word_exposures"] for entry in state["models"]]
        self.assertEqual(sorted(points), [7, 14, 21])
        self.assertEqual(len({entry["path"] for entry in state["models"]}), 3)
        for entry in state["models"]:
            self.assertEqual(sha(snapshot_path(directory, entry)), entry["sha256"])
            self.assertNotIn(22, entry["nominal_crossed_milestones"])
        final = state["final_receipts"][-1]
        final_path = snapshot_path(directory, final)
        receipt = json.loads(final_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["stop_reason"], "word_budget_reached")
        self.assertEqual(receipt["counts"]["word_exposures"], 21)
        self.assertNotIn(22, state["crossed_milestones"])
        final_sha = sha(final_path)
        after_stop, _ = self.run_engine(protocol, "sparse", "run", dataset,
                                        resume_checkpoint=directory / "checkpoint.pt")
        self.assertEqual(after_stop["snapshot_state"], state)
        self.assertEqual(sha(final_path), final_sha)
        # A stale/damaged supposedly immutable model cannot be silently accepted.
        original = first_path.read_bytes()
        first_path.write_bytes(original + b"tampered-for-test")
        with self.assertRaisesRegex(Exception, "(?i)(snapshot|hash|sha)"):
            self.run_engine(protocol, "sparse", "run", dataset,
                            resume_checkpoint=directory / "checkpoint.pt")
        first_path.with_suffix(".tampered-test-evidence").write_bytes(first_path.read_bytes())
        first_path.write_bytes(original)
        MEASUREMENTS["milestones"] = {"thresholds": [5, 6, 14, 22], "actual_words": [7, 14, 21],
                                     "first_two_thresholds_share_weights": True, "old_snapshot_not_rewritten": True,
                                     "resume_deduplication": True, "word_cap": 22, "final_actual_words": 21,
                                     "did_not_claim_unreached_threshold": True, "tamper_rejected": True,
                                     "engine_updates": 3}

    def test_02_dense_gradient_oracle_and_missing_indexer(self):
        from src.babylm_hybrid import evaluation
        dataset = self.dataset()
        manifest = self.directory / "eval-fixture.json"
        manifest.write_text(json.dumps({"total_windows": 1,
            "source_summaries": [{"source_index": 0, "source": "synthetic.dev"}]}), encoding="utf-8")
        evaluation_item = {"window_index": 0, "source_index": 0,
                           "input_ids": np.arange(12, dtype=np.int64)+1,
                           "word_exposures": 6, "input_tokens": 12, "loss_tokens": 11,
                           "single_segment": True, "reset_model_state_before": True}
        protocol = self.protocol(dataset, max_updates=1, windows_per_update=1, grad_clip_norm=0.02,
                                 eval_manifest=str(manifest), eval_manifest_sha256=sha(manifest),
                                 eval_window_indices=[0], eval_final=True, eval_position_diagnostics=True)
        with mock.patch.object(evaluation, "iter_windows", side_effect=lambda *args, **kwargs: iter([evaluation_item])):
            summary, directory = self.run_engine(protocol, "dense", "run", dataset)
        event = next(e for e in fixtures.read_events(directory) if e["type"] == "update")
        self.assertEqual(event["indexer_grad_norm"], 0.0)
        actual = load(directory / "checkpoint.pt")["model_state"]
        MEASUREMENTS["dense_gradient_oracle"] = self.assert_oracle_update(
            "dense", dataset, protocol, event, actual)
        eval_event = next(e for e in fixtures.read_events(directory) if e["type"] == "evaluation")
        diagnostics = eval_event["metrics"]["position_diagnostics"]
        self.assertEqual(diagnostics["extra_model_forwards"], 0)
        self.assertEqual(diagnostics["query_history_bins"]["1-256"]["loss_tokens"], 11)
        self.assertEqual(summary["eval_counts"]["forward_calls"], 1)
        MEASUREMENTS["real_tiny_eval_position_integration"] = {
            "evaluation_forwards": 1, "additional_forward_for_diagnostics": 0,
            "position_loss_tokens": 11, "extra_optimizer_updates": 0}

    def test_03_gradient_helper_independent_pythagorean_and_no_grad_cases(self):
        first = nn.Parameter(torch.ones(2))
        second = nn.Parameter(torch.ones(1))
        missing = nn.Parameter(torch.ones(3))
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = torch.tensor([12.0])
        before = [first.grad.clone(), second.grad.clone()]
        self.assertEqual(training._gradient_l2_norm([first]), 5.0)
        self.assertEqual(training._gradient_l2_norm([second]), 12.0)
        self.assertEqual(training._gradient_l2_norm([first, second, missing]), 13.0)
        self.assertEqual(training._gradient_l2_norm([]), 0.0)
        self.assertEqual(training._gradient_l2_norm([missing]), 0.0)
        self.assertTrue(torch.equal(first.grad, before[0]))
        self.assertTrue(torch.equal(second.grad, before[1]))
        self.assertIsNone(missing.grad)
        MEASUREMENTS["helper_edge_cases"] = {"three_four_twelve_norm": 13.0,
                                            "missing_and_empty_norm": 0.0,
                                            "gradient_buffers_unchanged": True,
                                            "full_model_calls": 0, "optimizer_updates": 0}

    def test_04_invalid_thresholds_rejected_without_model_work(self):
        dataset = self.dataset()
        invalid = [[1, 1], [3, 2], [0, 1], [-1], [True], [1.5], "1,2"]
        before = dict(COUNTS)
        for i, value in enumerate(invalid):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_engine(self.protocol(dataset, milestone_word_exposures=value), "dense", f"bad-{i}", dataset)
        self.assertEqual(COUNTS, before)
        MEASUREMENTS["invalid_thresholds"] = {"rejected": invalid, "full_model_calls": 0, "optimizer_updates": 0}

    def test_05_immutable_publish_cannot_overwrite_and_preserves_failed_temp(self):
        for filename, first, second, use_torch in (
            ("receipt.json", {"actual_words": 7}, {"actual_words": 99}, False),
            ("model.pt", {"weight": torch.tensor([3.0])}, {"weight": torch.tensor([9.0])}, True),
        ):
            with self.subTest(filename=filename):
                path = self.directory / filename
                training._immutable_save(path, first, torch_format=use_torch)
                original = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    training._immutable_save(path, second, torch_format=use_torch)
                self.assertEqual(path.read_bytes(), original)
                pending = list(self.directory.glob(filename+".tmp-*"))
                self.assertEqual(len(pending), 1)
                recovered = load(pending[0]) if use_torch else json.loads(pending[0].read_text(encoding="utf-8"))
                fixtures.assert_nested_equal(self, recovered, second)
        MEASUREMENTS["immutable_publication"] = {"json_and_torch_existing_targets_unchanged": True,
                                                "failed_publication_payload_retained": True,
                                                "full_model_calls": 0, "optimizer_updates": 0}


if __name__ == "__main__":
    unittest.main(verbosity=2)
