"""Synthetic log-only endpoint audits: zero model calls and no real training."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import summarize_babylm_one_epoch_v0 as metrics


def fixture():
    protocol = {"max_epochs": 1, "expected_stop_reason": "epoch_complete", "scope": "scientific",
                "windows_per_update": 2,
                "expected_epoch_counts": {"windows": 5, "updates": 3, "word_exposures": 11,
                                          "input_tokens": 26, "loss_tokens": 21,
                                          "forward_calls": 5, "backward_calls": 5,
                                          "final_update_windows": 1}}
    fingerprint = {"total_windows": 5}
    source_hashes = {"synthetic-training.py": "a" * 64}
    counts = {key: 0 for key in ("windows", "updates", "word_exposures", "input_tokens", "loss_tokens",
                                 "forward_calls", "backward_calls", "scientific_updates", "engineering_updates",
                                 "forward_attempts", "backward_attempts", "forward_input_tokens",
                                 "forward_word_exposures", "skipped_zero_target_input_tokens",
                                 "skipped_zero_target_word_exposures")}
    events = [{"event_id": 1, "type": "run_start", "protocol_sha256": metrics.canonical_sha(protocol),
               "source_hashes": source_hashes, "data_fingerprint": fingerprint,
               "counts": dict(counts), "cursor": {"epoch": 0, "position": 0}}]
    words, lengths = [1, 2, 1, 3, 4], [3, 5, 4, 6, 8]
    for update, indices in enumerate(([2, 0], [4, 1], [3]), 1):
        windows = []
        for index in indices:
            row = {"epoch": 0, "window_index": index, "word_exposures": words[index],
                   "input_tokens": lengths[index], "loss_tokens": lengths[index] - 1,
                   "forward_started": True, "forward_completed": True,
                   "backward_started": True, "backward_completed": True}
            windows.append(row)
            for key in ("word_exposures", "input_tokens", "loss_tokens"):
                counts[key] += row[key]
            for key in ("windows", "forward_calls", "backward_calls", "forward_attempts", "backward_attempts"):
                counts[key] += 1
            counts["forward_input_tokens"] += row["input_tokens"]
            counts["forward_word_exposures"] += row["word_exposures"]
        counts["updates"] += 1
        counts["scientific_updates"] += 1
        cursor = {"epoch": 1, "position": 0} if update == 3 else {"epoch": 0, "position": 2 * update}
        events.append({"event_id": len(events) + 1, "type": "update", "windows": windows,
                       "window_ids": [{"epoch": 0, "window_index": i} for i in indices],
                       "counts": dict(counts), "cursor": cursor,
                       "loss_tokens_this_update": sum(w["loss_tokens"] for w in windows),
                       "loss_token_weighted_ce": float(update), "loss_token_weighted_aux": 0.1 * update,
                       "lr": {"backbone": 0.001 / update, "indexer": 0.002 / update},
                       "gradient_clip_scope": "separate_groups",
                       "backbone_grad_clip_coefficient": 0.8, "indexer_grad_clip_coefficient": 0.4})
    evaluations = {"evaluations": 0, "forward_calls": 0, "input_tokens": 0,
                   "loss_tokens": 0, "word_exposures": 0}
    events.append({"event_id": len(events) + 1, "type": "run_stop", "status": "epoch_complete",
                   "counts": dict(counts), "cursor": cursor, "eval_counts": evaluations})
    summary = {"status": "epoch_complete", "scope": "scientific", "mode": "sparse",
               "protocol_sha256": metrics.canonical_sha(protocol), "source_hashes": source_hashes,
               "data_fingerprint": fingerprint, "counts": dict(counts), "cursor": cursor,
               "eval_counts": evaluations, "last_lr": events[-2]["lr"]}
    return protocol, summary, events


def write_fixture(path, protocol, summary, events):
    (path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


class OneEpochMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol, self.summary, self.events = fixture()

    def run_audit(self, tail=2):
        write_fixture(self.root, self.protocol, self.summary, self.events)
        return metrics.summarize(self.root, tail)

    def test_weighted_ppl_step_ppl_and_partial_batch_are_distinct(self):
        result = self.run_audit()
        tail = result["tail"]
        self.assertEqual((tail["first_update"], tail["last_update"], tail["loss_tokens"]), (2, 3, 16))
        observed = tail["metrics"]
        self.assertAlmostEqual(observed["token_weighted_lm_nll"], 37 / 16)
        self.assertAlmostEqual(observed["token_weighted_lm_ppl"], math.exp(37 / 16))
        self.assertEqual(observed["arithmetic_mean_step_lm_nll"], 2.5)
        self.assertAlmostEqual(observed["arithmetic_mean_step_lm_ppl"], (math.exp(2) + math.exp(3)) / 2)
        self.assertAlmostEqual(observed["exp_arithmetic_mean_step_lm_nll"], math.exp(2.5))
        self.assertAlmostEqual(observed["auxiliary_loss_separate"]["token_weighted_mean"], 3.7 / 16)
        self.assertEqual(result["epoch_coverage"]["final_update_windows"], 1)
        self.assertTrue(result["epoch_coverage"]["every_window_exactly_once"])
        self.assertEqual(result["analysis_counts"], {"forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0})
        self.assertIn("not_heldout", result["scope"])
        self.assertEqual(tail["controls"]["clip_coefficients"]["indexer"]["min"], 0.4)
        self.assertEqual(set(result["input_file_sha256"]), {"events.jsonl", "summary.json", "protocol.json"})

    def test_old_global_clip_is_explicitly_compatible(self):
        for event in self.events[1:-1]:
            del event["backbone_grad_clip_coefficient"], event["indexer_grad_clip_coefficient"]
            event["gradient_clip_scope"] = "global"
            event["grad_clip_coefficient"] = 0.7
        result = self.run_audit()
        controls = result["tail"]["controls"]
        self.assertEqual(controls["clip_coefficients"]["backbone"]["min"], 0.7)
        self.assertEqual(controls["clip_coefficients"]["indexer"]["min"], 0.7)
        self.assertEqual(controls["clip_coefficient_sources"], ["legacy_global_coefficient"])

    def test_clip_fields_without_grad_alias_are_accepted(self):
        for event in self.events[1:-1]:
            for group in ("backbone", "indexer"):
                event[f"{group}_clip_coefficient"] = event.pop(f"{group}_grad_clip_coefficient")
        self.assertEqual(self.run_audit()["status"], "audited_epoch_complete")

    def test_unfinished_summary_rejected(self):
        self.summary["status"] = "max_updates_reached"
        with self.assertRaisesRegex(metrics.AuditError, "not epoch_complete"):
            self.run_audit()

    def test_incomplete_terminal_cursor_rejected(self):
        self.summary["cursor"] = {"epoch": 0, "position": 5}
        with self.assertRaisesRegex(metrics.AuditError, "incomplete epoch cursor"):
            self.run_audit()

    def test_duplicate_window_rejected(self):
        self.events[2]["windows"][0]["window_index"] = 0
        with self.assertRaisesRegex(metrics.AuditError, "Duplicate"):
            self.run_audit()

    def test_second_epoch_rejected(self):
        self.events[2]["windows"][0]["epoch"] = 1
        with self.assertRaisesRegex(metrics.AuditError, "multiple epoch"):
            self.run_audit()

    def test_extra_last_batch_window_rejected(self):
        self.events[-2]["windows"].append(dict(self.events[-2]["windows"][0]))
        with self.assertRaisesRegex(metrics.AuditError, "wrong batch size"):
            self.run_audit()

    def test_incomplete_backward_rejected(self):
        self.events[1]["windows"][0]["backward_completed"] = False
        with self.assertRaisesRegex(metrics.AuditError, "Incomplete backward_completed"):
            self.run_audit()

    def test_cumulative_word_tampering_rejected(self):
        self.events[2]["counts"]["word_exposures"] += 1
        with self.assertRaisesRegex(metrics.AuditError, "Cumulative count mismatch"):
            self.run_audit()

    def test_protocol_hash_rejected(self):
        self.protocol["unexpected_edit"] = True
        with self.assertRaisesRegex(metrics.AuditError, "protocol SHA mismatch"):
            self.run_audit()

    def test_terminal_status_and_missing_stop_rejected(self):
        self.events[-1]["status"] = "max_updates_reached"
        with self.assertRaisesRegex(metrics.AuditError, "Terminal event"):
            self.run_audit()
        self.events.pop()
        with self.assertRaisesRegex(metrics.AuditError, "Missing explicit"):
            self.run_audit()

    def test_nonfinite_json_and_ppl_overflow_rejected(self):
        for value, message in ((float("nan"), "Nonfinite JSON"), (float("inf"), "Nonfinite JSON"),
                               (1000.0, "PPL overflow")):
            with self.subTest(value=value):
                self.events[-2]["loss_token_weighted_ce"] = value
                with self.assertRaisesRegex(metrics.AuditError, message):
                    self.run_audit()

    def test_partial_run_cannot_emit_success_file(self):
        self.summary["status"] = "interrupted"
        write_fixture(self.root, self.protocol, self.summary, self.events)
        output = self.root / "not-created.json"
        args = ["summarize", "--run-dir", str(self.root), "--output", str(output), "--tail-updates", "2"]
        with mock.patch("sys.argv", args), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 2)
        self.assertFalse(output.exists())

    def test_cli_success_and_refusal_to_overwrite_evidence(self):
        write_fixture(self.root, self.protocol, self.summary, self.events)
        output = self.root / "result.json"
        args = ["summarize", "--run-dir", str(self.root), "--output", str(output), "--tail-updates", "2"]
        with mock.patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(metrics.main(), 0)
        before = output.read_bytes()
        with mock.patch("sys.argv", args), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(metrics.main(), 2)
        self.assertEqual(output.read_bytes(), before)

    def test_insufficient_tail_and_missing_group_clip_rejected(self):
        with self.assertRaisesRegex(metrics.AuditError, "Not enough updates"):
            self.run_audit(tail=4)
        del self.events[1]["indexer_grad_clip_coefficient"]
        with self.assertRaisesRegex(metrics.AuditError, "Missing separate-group"):
            self.run_audit()

    def freeze_fixture_lr(self):
        self.protocol.update({"max_word_exposures": 11, "warmup_word_exposures": 4,
                              "min_lr_ratio": 0.1, "learning_rate": 0.001,
                              "indexer_learning_rate": 0.002})
        digest = metrics.canonical_sha(self.protocol)
        self.summary["protocol_sha256"] = self.events[0]["protocol_sha256"] = digest
        for event in self.events[1:-1]:
            words = event["counts"]["word_exposures"]
            multiplier = words / 4 if words < 4 else 0.1 + 0.9 * (1 + math.cos(math.pi * (words - 4) / 7)) / 2
            event["lr_word_position"] = words
            event["lr_multiplier"] = multiplier
            event["lr"] = {"backbone": 0.001 * multiplier, "indexer": 0.002 * multiplier}
        self.summary["last_lr"] = dict(self.events[-2]["lr"])

    def test_actual_resume_event_is_rejected(self):
        self.events.insert(2, {"type": "resume"})
        for index, event in enumerate(self.events, 1):
            event["event_id"] = index
        with self.assertRaisesRegex(metrics.AuditError, "failure/resume"):
            self.run_audit()

    def test_frozen_lr_schedule_is_verified_and_tampering_rejected(self):
        self.freeze_fixture_lr()
        result = self.run_audit()
        self.assertTrue(result["lr_schedule_audit"]["all_update_lrs_verified_against_protocol"])
        self.assertEqual(result["lr_schedule_audit"]["verified_updates"], 3)
        self.events[1]["lr"]["indexer"] *= 1.01
        with self.assertRaisesRegex(metrics.AuditError, "indexer LR does not match"):
            self.run_audit()

    def test_lr_word_position_and_multiplier_must_match_schedule(self):
        self.freeze_fixture_lr()
        self.events[1]["lr_word_position"] += 1
        with self.assertRaisesRegex(metrics.AuditError, "LR word position"):
            self.run_audit()
        self.events[1]["lr_word_position"] -= 1
        self.events[1]["lr_multiplier"] *= 1.01
        with self.assertRaisesRegex(metrics.AuditError, "LR multiplier"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
