"""Synthetic-only task schema/ranking audit against pinned official AST oracles.

No benchmark record is read, no tokenizer/model is constructed, and no model
forward/backward/update runs. Only pure official functions are extracted; their
module-level dataset imports and CLI entrypoints are never executed.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import pathlib
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.babylm_hybrid import official_task_records as tasks


ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "literature/babylm-causal-adapter-2026-09-17"
SOURCE = PINNED / "strict/evaluation_pipeline/sentence_zero_shot"
ARTIFACT_ROOT = ROOT / "logs/babylm-task-record-fixtures-current"
COUNTS = {"official_pure_function_calls": 0, "synthetic_records_normalized": 0,
          "full_model_forward_calls": 0, "backward_calls": 0, "optimizer_updates": 0,
          "scientific_updates": 0, "real_benchmark_records_read": 0, "gpu_calls": 0}
MEASUREMENTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_function(file_name, function_name):
    manifest = json.loads((PINNED / "manifest.json").read_text(encoding="utf-8"))
    path = SOURCE / file_name
    expected = next(item["sha256"] for item in manifest["sources"] if item["path"] == str(path.relative_to(PINNED)).replace("\\", "/"))
    if sha(path) != expected:
        raise RuntimeError("Pinned upstream oracle hash mismatch")
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    executable = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {"torch": torch, "Counter": Counter, "defaultdict": defaultdict, "pathlib": pathlib}
    exec(compile(executable, str(path), "exec"), namespace)
    function = namespace[function_name]

    def counted(*args, **kwargs):
        COUNTS["official_pure_function_calls"] += 1
        return function(*args, **kwargs)
    return counted


def official_ranking(records, scores, temperatures, seed, task, batch_sizes=None):
    initialize = official_function("compute_results.py", "update_subset_to_stats")
    rank = official_function("compute_results.py", "rank_and_evaluate")
    summarize = official_function("run.py", "process_results")
    stats = {temp: {} for temp in temperatures}
    predictions = {temp: defaultdict(list) for temp in temperatures}
    args = SimpleNamespace(task=task, save_predictions=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        offset = 0
        for size in batch_sizes or [len(records)]:
            current = records[offset:offset+size]
            metadata = [record["metadata"] for record in current]
            initialize(stats, metadata)
            log_probs = {temp: [torch.tensor([row[column] for row in scores[t][offset:offset+size]], dtype=torch.float64)
                                for column in range(len(records[0]["sentences"]))]
                         for t, temp in enumerate(temperatures)}
            rank(args, stats, log_probs, current, [r["label"] for r in current],
                 metadata, [r["UID"] for r in current], predictions)
            offset += size
    accuracies, macro = summarize(args, stats)
    return stats, {t: {uid: {"predictions": rows} for uid, rows in predictions[t].items()}
                   for t in predictions}, accuracies, macro


def score_payload(records, scores, temperatures=(1.0,), target_counts=None):
    n, k = len(records), len(records[0]["sentences"])
    target_counts = target_counts or [[2]*k for _ in range(n)]
    return {"task": records[0]["task"], "upstream_commit": "6f825c291e2c4c78ad33b1935fd64d45f52642dc",
            "temperatures": list(temperatures), "scores": scores,
            "scored_target_counts": target_counts,
            "scorable_candidates": [[value > 0 for value in row] for row in target_counts],
            "ranking_allowed_per_record": [all(value > 0 for value in row) for row in target_counts],
            "record_ids": [record["record_id"] for record in records],
            "normalization": "completion_token_sum", "official_benchmark_executed": False}


class TaskRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.directory = ARTIFACT_ROOT / self._testMethodName
        self.directory.mkdir(parents=True, exist_ok=True)

    def normalize(self, raw, task="blimp", filename="fixture.jsonl", line_number=1, **kwargs):
        line = json.dumps(raw, ensure_ascii=False)+"\n"
        path = self.directory / filename
        # This synthetic artifact is the complete source of one test record.
        # Preserve the exact physical newline supplied to normalize_record.
        # Windows text-mode translation would otherwise change file provenance.
        path.write_bytes(line.encode("utf-8"))
        normalized = tasks.normalize_record(raw, task=task, source_path=str(path), line_number=line_number,
                                            source_sha256=sha(path), raw_line=line, **kwargs)
        COUNTS["synthetic_records_normalized"] += 1
        return normalized

    @staticmethod
    def blimp(uid="A", number=0):
        return {"sentence_good": f"A child sees {number} birds.", "sentence_bad": f"A child see {number} birds.",
                "field": "syntax_semantics", "UID": uid, "linguistics_term": "agreement",
                "pairID": str(number), "unrelated_provenance": {"synthetic": True}}

    @staticmethod
    def ewok():
        return {"Context1": "The red cup is on the table.", "Context2": "The red cup is under the table.",
                "Target1": "It is above the floor.", "Target2": "A deliberately unused second target.",
                "Domain": "synthetic_spatial", "ContextType": "assertion", "ContextDiff": "on/under",
                "TargetDiff": "above/below", "item_id": "synthetic-ewok-1"}

    def test_01_blimp_and_supplement_match_pinned_decoder(self):
        oracle = official_function("read_files.py", "decode_blimp")
        examples = [(self.blimp(), "normal.jsonl"),
                    ({"sentence_good": "The fox sleeps.", "sentence_bad": "The fox sleep.", "extra": 19}, "synthetic_supplement.jsonl")]
        for raw, filename in examples:
            with self.subTest(filename=filename):
                actual = self.normalize(raw, filename=filename)
                expected = oracle(copy.deepcopy(raw), Path(filename))
                for key, value in expected.items():
                    self.assertEqual(actual[key], value)
                self.assertEqual(actual["metadata"], {k: expected[k] for k in ("field", "UID", "linguistics_term")})
                self.assertEqual(actual["source"]["file_sha256"], sha(self.directory / filename))
                self.assertEqual(actual["source"]["line_number"], 1)
                self.assertEqual(actual["source"]["raw_line_sha256"], hashlib.sha256((self.directory / filename).read_bytes()).hexdigest())
        raw = {**self.blimp(), "sentence_good": "The fox\u2028sleeps calmly."}
        physical_line = json.dumps(raw, ensure_ascii=False)+"\r\n"
        physical_path = self.directory / "unicode-crlf.jsonl"
        physical_path.write_bytes(physical_line.encode("utf-8"))
        decoded = tasks.normalize_record(raw, task="blimp", source_path=str(physical_path), line_number=1,
                                         source_sha256=sha(physical_path), raw_line=physical_line)
        COUNTS["synthetic_records_normalized"] += 1
        self.assertEqual(decoded["source"]["raw_line_sha256"], sha(physical_path))
        self.assertEqual(decoded["sentences"][0], raw["sentence_good"])
        MEASUREMENTS["blimp_normalization"] = {"pinned_decoder_equivalent": True,
                                                "supplement_uses_filename_uid": True,
                                                "syntax_semantics_standardization": True}

    def test_02_ewok_matches_target1_only_conditional_official_contract(self):
        raw = self.ewok()
        oracle = official_function("read_files.py", "decode_ewok")
        for full in (False, True):
            actual = self.normalize(raw, task="ewok", full_sentence_scores=full)
            expected = oracle(copy.deepcopy(raw), full)
            for key, value in expected.items():
                self.assertEqual(actual[key], value)
            self.assertEqual(actual["completions"], [" "+raw["Target1"]]*2)
            self.assertFalse(any(raw["Target2"] in sentence for sentence in actual["sentences"]))
        MEASUREMENTS["ewok_normalization"] = {"target1_under_two_contexts": True,
                                               "no_invented_four_way_scoring": True,
                                               "full_sentence_scores_flag_ignored_by_upstream_decoder": True}

    def test_03_identity_provenance_stability_and_raw_schema_rejections(self):
        raw = self.blimp()
        before = copy.deepcopy(raw)
        first = self.normalize(raw)
        same = self.normalize(raw)
        other_line = self.normalize(raw, line_number=2)
        other_file = self.normalize(raw, filename="different.jsonl")
        changed = self.normalize({**raw, "pairID": "changed-extra-field"})
        self.assertEqual(raw, before)
        self.assertEqual(first["record_id"], same["record_id"])
        self.assertEqual(len({first["record_id"], other_line["record_id"], other_file["record_id"], changed["record_id"]}), 4)
        self.assertNotEqual(first["source"]["canonical_raw_sha256"], changed["source"]["canonical_raw_sha256"])
        invalid = [{k: v for k, v in raw.items() if k != "sentence_bad"},
                   {**raw, "sentence_good": ""}, {**raw, "sentence_bad": None},
                   {**raw, "field": 1}, {**raw, "UID": []}]
        for i, bad in enumerate(invalid):
            with self.subTest(bad=i), self.assertRaises((ValueError, TypeError)):
                self.normalize(bad, filename=f"bad-{i}.jsonl")
        with self.assertRaises((ValueError, TypeError)):
            self.normalize(raw, line_number=0)
        with self.assertRaises((ValueError, TypeError)):
            tasks.normalize_record(raw, task="blimp", source_path=str(self.directory / "fixture.jsonl"),
                                   line_number=1, source_sha256="z"*64, raw_line=json.dumps(raw))
        with self.assertRaises((ValueError, TypeError)):
            tasks.normalize_record(raw, task="blimp", source_path=str(self.directory / "fixture.jsonl"),
                                   line_number=1, source_sha256="0"*64,
                                   raw_line=json.dumps({**raw, "sentence_good": "Changed provenance."}))
        MEASUREMENTS["identity_provenance"] = {"input_not_mutated": True, "stable_identity": True,
                                              "source_line_and_raw_changes_disambiguated": True,
                                              "required_schema_rejected": True}

    def test_04_rank_all_candidates_and_macro_vs_micro_match_official(self):
        records = [self.normalize(self.blimp("A", i), filename=f"a{i}.jsonl") for i in range(3)]
        records += [self.normalize(self.blimp("B", 3), filename="b.jsonl")]
        scores = [[[-1.0, -3.0], [-2.0, -5.0], [-1.0, -7.0], [-9.0, -1.0]]]
        payload = score_payload(records, scores)
        expected_stats, expected_predictions, expected_accuracy, expected_macro = official_ranking(records, scores, [1.0], 17, "blimp")
        actual = tasks.aggregate_scores(records, payload, tie_seed=17)["results"][0]
        self.assertEqual(actual["counts_by_metadata"], expected_stats[1.0])
        self.assertEqual(actual["accuracy_by_metadata_pct"], expected_accuracy[1.0])
        self.assertEqual(actual["official_uid_mean_accuracy_pct"], expected_macro[1.0])
        self.assertEqual(actual["official_uid_mean_accuracy_pct"], 50.0)
        self.assertEqual(actual["diagnostic_item_accuracy_pct"], 75.0)
        self.assertEqual(actual["official_predictions"], expected_predictions[1.0])
        MEASUREMENTS["aggregation_oracle"] = {"uid_macro_percent": 50.0, "diagnostic_micro_percent": 75.0,
                                               "all_candidates_and_metadata_counts_match_pinned_official": True}

    def test_05_seeded_ties_match_official_without_touching_global_rng(self):
        records = [self.normalize(self.blimp("ties", i), filename=f"tie{i}.jsonl") for i in range(4)]
        temperatures = [1.0, 2.0]
        scores = [[[1, 1], [-1, -2], [4, 4], [-7, -7]], [[2, 2], [-4, -1], [0, 0], [5, 5]]]
        payload = score_payload(records, scores, temperatures)
        payload["batch_sizes"] = [2, 2]
        torch_before, py_before, np_before = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        first = tasks.aggregate_scores(records, payload, tie_seed=31)
        second = tasks.aggregate_scores(records, payload, tie_seed=31)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(py_before, random.getstate())
        self.assertEqual(np_before[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))
        stats, predictions, accuracy, macro = official_ranking(records, scores, temperatures, 31, "blimp", batch_sizes=[2, 2])
        for temp, result in zip(temperatures, first["results"]):
            self.assertEqual(result["official_predictions"], predictions[temp])
            self.assertEqual(result["counts_by_metadata"], stats[temp])
            self.assertEqual(result["official_uid_mean_accuracy_pct"], macro[temp])
            self.assertEqual(result["tie_count"], 3)
            expected_bounds = ({"minimum": 25.0, "maximum": 100.0, "expected": 62.5} if temp == 1.0 else
                               {"minimum": 0.0, "maximum": 75.0, "expected": 37.5})
            self.assertEqual(result["tie_uncertainty"]["item_accuracy_pct"], expected_bounds)
            self.assertEqual(result["tie_uncertainty"]["uid_mean_accuracy_pct"], expected_bounds)
        self.assertTrue(all(result["tie_uncertainty"] for result in first["results"]))
        MEASUREMENTS["ties"] = {"two_batch_two_temperature_official_sequence_matches": True,
                                 "local_seed_deterministic": True, "global_python_numpy_torch_rng_unchanged": True,
                                 "tied_records_per_temperature": 3, "no_temperature_selected": True}

    def test_06_invalid_or_incomplete_scores_fail_closed(self):
        records = [self.normalize(self.blimp())]
        base = score_payload(records, [[[-1, -2]]])
        mutations = [lambda p: p["scores"][0][0].__setitem__(0, float("nan")),
                     lambda p: p["scores"][0][0].__setitem__(1, float("inf")),
                     lambda p: p["scores"][0].pop(), lambda p: p["scores"][0][0].pop(),
                     lambda p: p.__setitem__("temperatures", [1.0, 2.0]),
                     lambda p: p.__setitem__("upstream_commit", "wrong"),
                     lambda p: p.__setitem__("task", "ewok"),
                     lambda p: p["scored_target_counts"][0].__setitem__(0, 0),
                     lambda p: p["scorable_candidates"][0].__setitem__(0, False),
                     lambda p: p["ranking_allowed_per_record"].__setitem__(0, False),
                     lambda p: p["scorable_candidates"][0].__setitem__(0, 1),
                     lambda p: p["ranking_allowed_per_record"].__setitem__(0, 1),
                     lambda p: p.__setitem__("batch_sizes", [2]),
                     lambda p: p.__setitem__("batch_sizes", [0, 1]),
                     lambda p: p.__setitem__("temperatures", [float("nan")]),
                     lambda p: p.__setitem__("record_ids", ["wrong-record"]),
                     lambda p: p.__setitem__("normalization", "completion_token_mean"),
                     lambda p: p.pop("normalization")]
        for index, mutate in enumerate(mutations):
            bad = copy.deepcopy(base)
            mutate(bad)
            with self.subTest(case=index), self.assertRaises((ValueError, TypeError)):
                tasks.aggregate_scores(records, bad, tie_seed=0)
        with self.assertRaises((ValueError, TypeError)):
            tasks.aggregate_scores(records * 2, score_payload(records * 2, [[[-1, -2], [-1, -2]]]))
        with self.assertRaises((ValueError, TypeError)):
            tasks.aggregate_scores([], base)
        MEASUREMENTS["invalid_scores"] = {"nan_inf_empty_targets_incomplete_and_duplicate_records_rejected": True,
                                          "sum_normalization_required_mean_and_missing_rejected": True,
                                          "intentional_divergence_from_upstream_nan_to_negative_infinity": True}

    def test_07_ewok_prediction_reports_chosen_sentence_not_shared_target(self):
        records = [self.normalize(self.ewok(), task="ewok", filename="ewok0.jsonl"),
                   self.normalize({**self.ewok(), "Domain": "synthetic_time"}, task="ewok", filename="ewok1.jsonl")]
        scores = [[[-5, -1], [-1, -6]]]
        _, predictions, accuracy, macro = official_ranking(records, scores, [1.0], 7, "ewok")
        actual = tasks.aggregate_scores(records, score_payload(records, scores), tie_seed=7)["results"][0]
        self.assertEqual(actual["official_predictions"], predictions[1.0])
        self.assertEqual(actual["accuracy_by_metadata_pct"], accuracy[1.0])
        self.assertEqual(actual["official_uid_mean_accuracy_pct"], macro[1.0])
        self.assertEqual(actual["official_uid_mean_accuracy_pct"], 50.0)
        MEASUREMENTS["ewok_ranking"] = {"chosen_full_sentence_matches_official": True,
                                        "shared_target_not_mistaken_for_prediction": True,
                                        "uid_macro_percent": 50.0}


if __name__ == "__main__":
    unittest.main(verbosity=2)
