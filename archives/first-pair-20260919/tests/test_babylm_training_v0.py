"""Independent synthetic CPU training-engine tests; not BabyLM pretraining.

Fixtures remain under the engineering audit log directory, including failed
attempts and their checkpoints. Production data path allowlists are not edited.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest import mock

import numpy as np
import torch

from src.babylm_hybrid.config import HybridConfig
from src.babylm_hybrid.model import HybridLM, build_model
from src.babylm_hybrid import training


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "logs/babylm-training-engine-test-artifacts-current"
COUNTS = {"model_forward_calls": 0, "backward_calls": 0,
          "engineering_optimizer_steps": 0, "scientific_optimizer_steps": 0,
          "real_training_tokens": 0, "gpu_calls": 0}
MEASUREMENTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SyntheticWindowDataset:
    """Explicit token/word/loss accounting with stable on-disk provenance."""

    def __init__(self, directory, specs, fail_index=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fail_index = fail_index
        self.rows = []
        for i, (length, words) in enumerate(specs):
            ids = (np.arange(length, dtype=np.uint32) * 7 + 11 + i * 13) % 97
            file = self.directory / f"synthetic-{i}.npy"
            np.save(file, ids)
            self.rows.append({"window_index": i, "input_ids": ids,
                              "word_exposures": words, "input_tokens": length,
                              "next_token_loss_positions": max(length - 1, 0),
                              "loss_tokens": max(length - 1, 0), "single_segment": True,
                              "reset_model_state_before": True, "source_index": i,
                              "segment_index_in_source": 0, "source_token_start": 0,
                              "source_token_end": length})
        self.ledger = self.directory / "synthetic-ledger.json"
        self.ledger.write_text(json.dumps({
            "synthetic_engineering_only": True,
            "rows": [{k: v for k, v in row.items() if k != "input_ids"} for row in self.rows],
            "arrays": {f"synthetic-{i}.npy": sha(self.directory / f"synthetic-{i}.npy")
                       for i in range(len(self.rows))},
        }, indent=2), encoding="utf-8")
        self.fingerprint = {"synthetic_engineering_only": True,
                            "ledger_sha256": sha(self.ledger), "windows": len(self.rows)}

    def __len__(self):
        return len(self.rows)

    def window(self, index):
        if index == self.fail_index:
            raise RuntimeError("synthetic injected window-read failure; preserve this exact reason")
        row = copy.deepcopy(self.rows[index])
        row["input_ids"].flags.writeable = False
        return row


def read_events(directory):
    path = Path(directory) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def event_kind(event):
    return event.get("type", event.get("event"))


def assert_nested_equal(testcase, a, b):
    if torch.is_tensor(a):
        testcase.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        testcase.assertTrue(np.array_equal(a, b))
    elif isinstance(a, dict):
        testcase.assertEqual(set(a), set(b))
        for key in a:
            assert_nested_equal(testcase, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        testcase.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_nested_equal(testcase, x, y)
    else:
        testcase.assertEqual(a, b)


class TrainingEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.directory = ARTIFACT_ROOT / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=True)

    def dataset(self, specs=((12, 7), (16, 9), (20, 13), (24, 15)), **kwargs):
        return SyntheticWindowDataset(self.directory / "fixture", specs, **kwargs)

    def run_engine(self, protocol, mode, name, dataset, **kwargs):
        directory = self.directory / name
        with mock.patch.object(training, "load_window_dataset", return_value=dataset):
            result = training.train_run(protocol, mode, directory, **kwargs)
        return result, directory

    def protocol(self, dataset, **changes):
        p = {"schema_version": 1, "scope": "engineering_smoke", "launch_allowed": True,
             "device": "cpu", "dtype": "float32", "model_config": HybridConfig().to_dict(),
             "backbone_seed": 731, "indexer_seed": 991, "data_order_seed": 137,
             "train_manifest": str(dataset.ledger), "train_manifest_sha256": sha(dataset.ledger),
             "max_word_exposures": 1000, "max_updates": 3, "max_wall_seconds": 60,
             "windows_per_update": 2, "learning_rate": 1e-3, "indexer_learning_rate": 3e-3,
             "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8,
             "grad_clip_norm": 1000.0, "aux_weight": 0.5, "warmup_word_exposures": 0,
             "min_lr_ratio": 1.0, "checkpoint_every_updates": 1, "eval_every_updates": 0}
        p.update(changes)
        return p

    def load_checkpoint(self, directory, name="checkpoint.pt"):
        return torch.load(directory / name, map_location="cpu", weights_only=False)

    def test_01_paired_data_order_counts_and_optimizer_groups(self):
        dataset = self.dataset()
        protocol = self.protocol(dataset)
        dense, dense_dir = self.run_engine(protocol, "dense", "dense", dataset)
        sparse, sparse_dir = self.run_engine(protocol, "sparse", "sparse", dataset)
        d = [e for e in read_events(dense_dir) if event_kind(e) == "update"]
        e = [e for e in read_events(sparse_dir) if event_kind(e) == "update"]
        self.assertEqual(len(d), 3)
        self.assertEqual([row["window_ids"] for row in d], [row["window_ids"] for row in e])
        used = [row for event in d for row in event["window_ids"]]
        expected_words = sum(dataset.rows[row["window_index"]]["word_exposures"] for row in used)
        expected_inputs = sum(dataset.rows[row["window_index"]]["input_tokens"] for row in used)
        expected_losses = sum(dataset.rows[row["window_index"]]["loss_tokens"] for row in used)
        for summary in (dense, sparse):
            self.assertEqual(summary["counts"]["word_exposures"], expected_words)
            self.assertEqual(summary["counts"]["input_tokens"], expected_inputs)
            self.assertEqual(summary["counts"]["loss_tokens"], expected_losses)
            self.assertEqual(summary["counts"]["updates"], 3)
        dc, ec = self.load_checkpoint(dense_dir), self.load_checkpoint(sparse_dir)
        dgroups, egroups = dc["optimizer_state"]["param_groups"], ec["optimizer_state"]["param_groups"]
        self.assertEqual(len(dgroups), 1)
        self.assertEqual(len(egroups), 2)
        self.assertEqual(sorted(g["lr"] for g in dgroups), [1e-3])
        self.assertEqual(sorted(g["lr"] for g in egroups), [1e-3, 3e-3])
        MEASUREMENTS["paired_order"] = {"window_ids": used, "word_exposures": expected_words,
                                       "input_tokens": expected_inputs, "loss_tokens": expected_losses,
                                       "dense_optimizer_groups": len(dgroups), "sparse_optimizer_groups": len(egroups)}

    def test_02_word_cap_stops_before_large_window_without_skipping(self):
        order = np.random.default_rng(np.random.SeedSequence([137, 0])).permutation(3).tolist()
        specs = [(8, 1)] * 3
        specs[order[0]], specs[order[1]] = (8, 4), (24, 20)
        dataset = self.dataset(specs)
        protocol = self.protocol(dataset, max_word_exposures=6, windows_per_update=3)
        summary, directory = self.run_engine(protocol, "dense", "cap", dataset)
        self.assertEqual(summary["counts"]["word_exposures"], 4)
        self.assertEqual(summary["counts"]["windows"], 1)
        self.assertEqual(summary["counts"]["updates"], 1)
        self.assertEqual(summary["cursor"], {"epoch": 0, "position": 1})
        updates = [e for e in read_events(directory) if event_kind(e) == "update"]
        self.assertEqual(updates[0]["window_ids"], [{"epoch": 0, "window_index": order[0]}])
        MEASUREMENTS["word_cap"] = {"cap": 6, "consumed": 4, "blocked_window_words": 20,
                                    "did_not_skip_blocked_window": True, "cursor": summary["cursor"]}

    def test_03_loss_weighted_accumulation_and_zero_target_window(self):
        dataset = self.dataset(((3, 2), (6, 4), (1, 1)))
        protocol = self.protocol(dataset, max_word_exposures=7, max_updates=1, windows_per_update=3)
        summary, directory = self.run_engine(protocol, "dense", "weighted", dataset)
        self.assertEqual(summary["counts"]["word_exposures"], 7)
        self.assertEqual(summary["counts"]["input_tokens"], 10)
        self.assertEqual(summary["counts"]["loss_tokens"], 7)
        self.assertEqual(summary["counts"]["forward_calls"], 2)
        cfg = HybridConfig()
        model = build_model(cfg, "dense", 731, 991)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
        total = 0
        token_sum = None
        individual_means = []
        order = np.random.default_rng(np.random.SeedSequence([137, 0])).permutation(3)
        for i in order:
            row = dataset.rows[i]
            if row["loss_tokens"] == 0:
                continue
            out = model(torch.tensor(row["input_ids"].astype(np.int64)).unsqueeze(0))
            weighted = out.lm_loss * out.token_loss_count
            token_sum = weighted if token_sum is None else token_sum + weighted
            total += out.token_loss_count
            individual_means.append(out.lm_loss.item())
        expected_loss = token_sum / total
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        event = next(e for e in read_events(directory) if event_kind(e) == "update")
        self.assertAlmostEqual(event["loss_token_weighted_ce"], expected_loss.item(), places=6)
        saved = self.load_checkpoint(directory)["model_state"]
        max_abs = 0
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=2e-5, atol=2e-6)
            max_abs = max(max_abs, (value - saved[name]).abs().max().item())
        MEASUREMENTS["loss_weighting"] = {"expected_token_weighted_ce": expected_loss.item(),
                                          "unweighted_window_mean_ce": sum(individual_means)/len(individual_means),
                                          "engine_ce": event["loss_token_weighted_ce"],
                                          "max_post_update_parameter_abs_difference": max_abs,
                                          "zero_target_window_still_counted": True}

    def test_04_all_zero_target_consumes_budget_without_update(self):
        dataset = self.dataset(((1, 1),))
        protocol = self.protocol(dataset, max_word_exposures=1, windows_per_update=1)
        summary, directory = self.run_engine(protocol, "sparse", "zero_targets", dataset)
        self.assertEqual(summary["counts"]["word_exposures"], 1)
        self.assertEqual(summary["counts"]["input_tokens"], 1)
        for name in ("updates", "forward_calls", "backward_calls", "loss_tokens"):
            self.assertEqual(summary["counts"][name], 0)
        self.assertTrue(any(event_kind(e) == "consumed_without_update" for e in read_events(directory)))

    def test_05_stop_resume_matches_uninterrupted_run_exactly(self):
        dataset = self.dataset(((12, 7), (16, 9), (20, 13)))
        protocol = self.protocol(dataset, max_updates=4)
        counted_forward = HybridLM.forward

        def consume_all_rngs(model, *args, **kwargs):
            # The production tiny model has no dropout. Advancing all three
            # streams ensures restore is tested beyond storing untouched seeds.
            random.random()
            np.random.random()
            torch.rand(1)
            return counted_forward(model, *args, **kwargs)

        with mock.patch.object(HybridLM, "forward", consume_all_rngs):
            continuous, continuous_dir = self.run_engine(protocol, "sparse", "continuous", dataset)
            partial, partial_dir = self.run_engine(protocol, "sparse", "resumed", dataset, stop_after_updates=2)
            self.assertEqual(partial["counts"]["updates"], 2)
            random.seed(989)
            np.random.seed(989)
            torch.manual_seed(989)
            resumed, _ = self.run_engine(protocol, "sparse", "resumed", dataset,
                                         resume_checkpoint=partial_dir / "checkpoint.pt")
        a, b = self.load_checkpoint(continuous_dir), self.load_checkpoint(partial_dir)
        for key in ("model_state", "optimizer_state", "counters", "cursor", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
            assert_nested_equal(self, a[key], b[key])
        self.assertEqual(continuous["counts"], resumed["counts"])
        expected = [e for e in read_events(continuous_dir) if event_kind(e) == "update"]
        actual = [e for e in read_events(partial_dir) if event_kind(e) == "update"]
        self.assertEqual(len(actual), 4)
        self.assertEqual([e["window_ids"] for e in actual], [e["window_ids"] for e in expected])
        self.assertEqual([e["loss_token_weighted_ce"] for e in actual], [e["loss_token_weighted_ce"] for e in expected])
        MEASUREMENTS["resume"] = {"continuous_updates": 4, "stopped_updates": 2,
                                   "resumed_new_updates": 2, "bitwise_exact_model_optimizer_rng_counters_cursor": True,
                                   "all_rng_streams_advanced_and_perturbed_before_restore": True,
                                   "cursor": resumed["cursor"]}

    def test_06_reject_unauthorized_or_invalid_protocol_before_work(self):
        dataset = self.dataset()
        invalid = [{"scope": "scientific"}, {"scope": "scientific_pretraining"}, {"device": "cuda"},
                   {"launch_allowed": False}, {"max_updates": 11},
                   {"windows_per_update": 0}, {"max_word_exposures": 0}]
        before = dict(COUNTS)
        for i, change in enumerate(invalid):
            with self.subTest(change=change), self.assertRaises(Exception):
                self.run_engine(self.protocol(dataset, **change), "dense", f"invalid-{i}", dataset)
        self.assertEqual(COUNTS, before)
        MEASUREMENTS["invalid_protocols_rejected_before_compute"] = invalid

    def test_07_resume_rejects_protocol_data_source_and_log_tamper(self):
        dataset = self.dataset()
        protocol = self.protocol(dataset, max_updates=2)
        summary, directory = self.run_engine(protocol, "sparse", "tamper", dataset, stop_after_updates=1)
        checkpoint = directory / "checkpoint.pt"
        original_bytes = checkpoint.read_bytes()
        events = directory / "events.jsonl"
        original_events = events.read_bytes()
        before = dict(COUNTS)
        with self.assertRaises(Exception):
            self.run_engine({**protocol, "learning_rate": 9e-3}, "sparse", "tamper", dataset,
                            resume_checkpoint=checkpoint)
        dataset.fingerprint["ledger_sha256"] = "f"*64
        with self.assertRaises(Exception):
            self.run_engine(protocol, "sparse", "tamper", dataset, resume_checkpoint=checkpoint)
        dataset.fingerprint["ledger_sha256"] = sha(dataset.ledger)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertTrue(saved["source_hashes"])
        source = next(iter(saved["source_hashes"]))
        saved["source_hashes"][source] = "0"*64
        torch.save(saved, checkpoint)
        with self.assertRaises(Exception):
            self.run_engine(protocol, "sparse", "tamper", dataset, resume_checkpoint=checkpoint)
        checkpoint.write_bytes(original_bytes)
        events.write_bytes(original_events + b'{"type":"tampered"}\n')
        with self.assertRaises(Exception):
            self.run_engine(protocol, "sparse", "tamper", dataset, resume_checkpoint=checkpoint)
        self.assertEqual(COUNTS, before)
        # Preserve the malformed evidence as well as restoring original artifacts for inspection.
        (directory / "tampered-events.jsonl").write_bytes(events.read_bytes())
        events.write_bytes(original_events)
        MEASUREMENTS["resume_tamper_rejections"] = ["protocol", "data_fingerprint", "source_hashes", "events_log"]

    def test_08_window_failure_preserves_reason_and_last_success(self):
        order = np.random.default_rng(np.random.SeedSequence([137, 0])).permutation(4).tolist()
        dataset = self.dataset(fail_index=order[1])
        protocol = self.protocol(dataset, windows_per_update=1, max_updates=3)
        directory = self.directory / "injected_failure"
        with self.assertRaises(Exception):
            self.run_engine(protocol, "sparse", "injected_failure", dataset)
        failure_files = list((directory / "failures").glob("*.json"))
        self.assertTrue(failure_files)
        failure_text = "\n".join(path.read_text(encoding="utf-8") for path in failure_files)
        self.assertIn("synthetic injected window-read failure; preserve this exact reason", failure_text)
        self.assertIn("Traceback", failure_text)
        successful = self.load_checkpoint(directory)
        self.assertEqual(successful["counters"]["updates"], 1)
        failure_checkpoint = failure_files[-1].with_suffix(".pt")
        self.assertTrue(failure_checkpoint.exists())
        failed = torch.load(failure_checkpoint, map_location="cpu", weights_only=False)
        self.assertFalse(failed["resume_supported"])
        MEASUREMENTS["failure_preservation"] = {"last_successful_updates": 1,
                                                "raw_reason_and_traceback_preserved": True,
                                                "failure_checkpoint_not_silently_resumable": True}

    def test_09_evaluation_counters_boundaries_and_resume_deduplication(self):
        from src.babylm_hybrid import evaluation
        dataset = self.dataset()
        eval_manifest = self.directory / "synthetic-eval-manifest.json"
        eval_manifest.write_text('{"synthetic_engineering_evaluator_stub":true}', encoding="utf-8")
        protocol = self.protocol(dataset, max_updates=2, eval_manifest=str(eval_manifest),
                                 eval_manifest_sha256=sha(eval_manifest), eval_every_updates=1,
                                 eval_initial=True, eval_final=True, eval_window_indices=[0, 1, 2])
        fake_metrics = {"status": "nll_evaluation_complete",
                        "total": {"word_exposures": 17, "input_tokens": 23, "loss_tokens": 19,
                                  "forward_calls": 3, "windows": 3, "nll": 1.25},
                        "per_source": {}, "optimizer_updates": 0}
        with mock.patch.object(evaluation, "evaluate_windows", return_value=copy.deepcopy(fake_metrics)) as evaluator:
            continuous, continuous_dir = self.run_engine(protocol, "dense", "eval-continuous", dataset)
            self.assertEqual(evaluator.call_count, 3)  # initial + update1 + update2, not duplicate final
            partial, partial_dir = self.run_engine(protocol, "dense", "eval-resume", dataset, stop_after_updates=1)
            self.assertEqual(partial["eval_counts"]["evaluations"], 2)
            resumed, _ = self.run_engine(protocol, "dense", "eval-resume", dataset,
                                         resume_checkpoint=partial_dir / "checkpoint.pt")
            self.assertEqual(evaluator.call_count, 6)
            zero, _ = self.run_engine(protocol, "dense", "eval-zero", dataset, stop_after_updates=0)
            self.assertEqual(evaluator.call_count, 7)  # initial/final both at update0 evaluated once
        expected = {"word_exposures": 51, "input_tokens": 69, "loss_tokens": 57,
                    "forward_calls": 9, "evaluations": 3}
        self.assertEqual(continuous["eval_counts"], expected)
        self.assertEqual(resumed["eval_counts"], expected)
        self.assertEqual(zero["counts"]["updates"], 0)
        self.assertEqual(zero["eval_counts"]["evaluations"], 1)
        self.assertEqual(zero["eval_counts"]["word_exposures"], 17)
        for directory in (continuous_dir, partial_dir):
            events = [e for e in read_events(directory) if event_kind(e) == "evaluation"]
            self.assertEqual(len(events), 3)
        MEASUREMENTS["evaluation_engine_integration"] = {
            "nested_total_counters_propagated": True,
            "continuous_equals_resumed_evaluation_counts": expected,
            "initial_periodic_final_deduplicated": True,
            "zero_update_initial_and_final_deduplicated": True,
            "mocked_evaluator_calls": 7, "actual_evaluation_model_forwards": 0,
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
