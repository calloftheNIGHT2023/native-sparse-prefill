"""Synthetic-only conformance to pinned official BabyLM causal source.

Running this file directly writes append-only test history and actual counters.
No official benchmark questions/answers or real data splits are loaded.
"""
import ast
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from src.babylm_hybrid.official_causal_scoring import (
    CausalScoringError, FixedLocalTokenizer, HybridCausalAdapter, LENGTH_NORMALIZED_TASKS,
    UPSTREAM_COMMIT, collate_records, prepare_record, score_records,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "literature/babylm-causal-adapter-2026-09-17"
OFFICIAL = SNAPSHOT / "strict/evaluation_pipeline/sentence_zero_shot"
COUNTS = {"fake_model_forward_calls": 0, "tiny_cpu_model_forward_calls": 0,
          "scientific_model_forward_calls": 0, "backward_calls": 0,
          "optimizer_updates": 0, "gpu_model_forward_calls": 0,
          "official_benchmark_items_read": 0}


def official_functions():
    """Load only selected code ASTs; never import official data/remote loaders."""
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == UPSTREAM_COMMIT
    for entry in manifest["sources"]:
        assert hashlib.sha256((SNAPSHOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    tree = ast.parse((OFFICIAL / "dataset.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CompletionRankingDataset":
            method = next(m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == "process_causal_sentences")
            selected.append(ast.ClassDef(name="CompletionRankingDataset", bases=[], keywords=[], body=[method], decorator_list=[]))
        elif isinstance(node, ast.FunctionDef) and node.name == "get_causal_collate_fn":
            selected.append(node)
    for filename, names in [("compute_results.py", {"compute_causal_results", "update_subset_to_stats"}),
                            ("run.py", {"get_temperatures"})]:
        tree = ast.parse((OFFICIAL / filename).read_text(encoding="utf-8"))
        selected.extend(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)
    captured = []
    def capture(args, stats, values, *ignored):
        captured.append({temp: torch.stack(scores, dim=1).tolist() for temp, scores in values.items()})
    env = {"torch": torch, "pad_sequence": pad_sequence, "F": F,
           "Counter": Counter, "defaultdict": defaultdict, "DEVICE": torch.device("cpu"),
           "LENGTH_NORMALIZED_TASKS": set(LENGTH_NORMALIZED_TASKS), "tqdm": lambda data: data,
           "rank_and_evaluate": capture}
    code = compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), "pinned-official-selected-code", "exec")
    exec(code, env)
    return env, captured


class CharProcessor:
    pad_token_id = 0
    def __call__(self, text, return_offsets_mapping=True):
        return {"input_ids": [3 + ord(c) % 90 for c in text], "attention_mask": [1] * len(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class FakeLM(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.child = nn.Dropout(0.5)
        self.fail = fail
        self.segments = []

    def forward(self, input_ids, segment_ids=None, aux_weight=0.0):
        COUNTS["fake_model_forward_calls"] += 1
        assert not torch.is_grad_enabled()
        assert not self.training and not self.child.training
        assert aux_weight == 0.0
        self.segments.append(segment_ids.clone())
        if self.fail:
            raise RuntimeError("fixture failure")
        vocab = torch.arange(97, dtype=torch.float32, device=input_ids.device)
        # Context- and position-dependent logits make masking/shift mistakes visible.
        center = ((input_ids.cumsum(1) + torch.arange(input_ids.shape[1])) % 97).unsqueeze(-1)
        return SimpleNamespace(logits=-(vocab - center).abs() / 23.0)


class CausalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.oracle, cls.captured = official_functions()
        cls.processor = CharProcessor()
        cls.records = [
            {"sentences": ["red bird flies", "red bird sleeps"], "completions": ["flies", "sleeps"]},
            {"sentences": ["the small bird sings", "the bird rests"], "completions": ["sings", "rests"]},
        ]

    def oracle_batch(self, records, processor):
        dataset = self.oracle["CompletionRankingDataset"]()
        dataset.processor = processor
        processed = [dataset.process_causal_sentences(r, None) for r in records]
        rows = [(r, p, 0, {"fixture": "synthetic"}, f"fixture-{i}", None)
                for i, (r, p) in enumerate(zip(records, processed))]
        return processed, self.oracle["get_causal_collate_fn"](processor.pad_token_id)(rows)

    def test_preparation_and_padding_match_official_code(self):
        expected, official_batch = self.oracle_batch(self.records, self.processor)
        actual = [prepare_record(r, self.processor) for r in self.records]
        for left, right in zip(actual, expected):
            self.assertEqual(set(left), set(right))
            for key in left:
                if left[key] is None:
                    self.assertIsNone(right[key])
                else:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        ours = collate_records(actual, self.processor.pad_token_id)
        for key in ours:
            torch.testing.assert_close(ours[key], official_batch[1][key], rtol=0, atol=0)

    def test_scores_sum_mean_temperature_and_counts_against_official(self):
        _, batch = self.oracle_batch(self.records, self.processor)
        for task in ["blimp", "ewok", "comps", "entity_tracking", "global_piqa_parallel", "global_piqa_nonparallel"]:
            model = FakeLM()
            model.train()
            model.child.eval()  # Restore mixed module modes, not just top-level.
            result = score_records(model, self.processor, self.records, task, temperatures=(1.0, 0.7))
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            wrapper = HybridCausalAdapter(FakeLM()).eval()
            with torch.no_grad():
                self.oracle["compute_causal_results"](SimpleNamespace(task=task, images_path=None, save_predictions=False), wrapper, [batch], [1.0, 0.7])
            expected = self.captured[-1]
            for i, temp in enumerate((1.0, 0.7)):
                torch.testing.assert_close(torch.tensor(result["scores"][i]), torch.tensor(expected[temp]), atol=0, rtol=0)
            self.assertEqual(result["counters"]["model_forward_calls"], 2)
            self.assertEqual(result["counters"]["candidate_sequences"], 4)
            self.assertEqual(result["counters"]["source_tokens"], sum(len(s) for r in self.records for s in r["sentences"]))
            self.assertEqual(result["counters"]["scored_target_tokens"], sum(len(c) for r in self.records for c in r["completions"]))
            self.assertGreaterEqual(result["counters"]["padded_forward_positions"], result["counters"]["forward_input_tokens"])
            self.assertFalse(result["official_benchmark_executed"])

    def test_fixed_tokenizer_no_bos_eos_and_unicode_offsets(self):
        tokenizer = FixedLocalTokenizer(ROOT / "data/babylm-2026-tokenizer-16k-v0/tokenizer.json")
        records = [{"sentences": ["A cafe\u0301 bird says hello.", "A café bird says goodbye."],
                    "completions": ["hello.", "goodbye."]}]
        expected, _ = self.oracle_batch(records, tokenizer)
        ours = prepare_record(records[0], tokenizer)
        for key in ours:
            if ours[key] is not None:
                torch.testing.assert_close(ours[key], expected[0][key], rtol=0, atol=0)
        for sentence in records[0]["sentences"]:
            encoded = tokenizer(text=sentence)
            self.assertNotIn(tokenizer.bos_token_id, encoded["input_ids"])
            self.assertNotIn(tokenizer.eos_token_id, encoded["input_ids"])
            self.assertEqual(tokenizer.backend.decode(encoded["input_ids"]), sentence)

    def test_whole_sentence_excludes_first_token_and_overlap_is_included(self):
        record = {"sentences": ["abc"], "completions": ["abc"]}
        prepared = collate_records([prepare_record(record, self.processor)], 0)
        self.assertEqual(prepared["sentence_0_phrase_mask"].sum().item(), 2)
        def merged(text, return_offsets_mapping=True):
            return {"input_ids": [3, 4], "attention_mask": [1, 1], "offset_mapping": [(0, 3), (3, 7)]}
        phrase = prepare_record({"sentences": ["red car"], "completions": ["ar"]}, merged)
        self.assertEqual(phrase["sentence_0_phrase_mask"].tolist(), [0, 1])

    def test_empty_completion_is_zero_not_nan(self):
        result = score_records(FakeLM(), self.processor, [{"sentences": ["two tokens"], "completions": [""]}], "global_piqa_parallel")
        self.assertEqual(result["scores"], [[[0.0]]])
        self.assertEqual(result["counters"]["scored_target_tokens"], 0)
        self.assertEqual(result["scorable_candidates"], [[False]])
        self.assertEqual(result["ranking_allowed_per_record"], [False])

    def test_invalid_domain_fails_before_forward_and_failure_restores_modes(self):
        before = COUNTS["fake_model_forward_calls"]
        for record in [{"sentences": [""], "completions": [""]}, {"sentences": ["x"], "completions": ["x"]},
                       {"sentences": ["abc"], "completions": ["b"]},
                       {"sentences": ["abc"], "completions": ["c"], "image": "no"}]:
            with self.assertRaises(ValueError):
                prepare_record(record, self.processor)
        with self.assertRaises(ValueError):
            prepare_record(self.records[0], self.processor, max_input_tokens=2)
        for temperatures in [(), (0,), (float("nan"),), (1, 1)]:
            with self.assertRaises(ValueError):
                score_records(FakeLM(), self.processor, self.records, "blimp", temperatures=temperatures)
        with self.assertRaises(ValueError):
            score_records(FakeLM(), self.processor, self.records, "reading")
        self.assertEqual(before, COUNTS["fake_model_forward_calls"])
        model = FakeLM(fail=True)
        model.train()
        model.child.eval()
        with self.assertRaisesRegex(CausalScoringError, "fixture failure") as caught:
            score_records(model, self.processor, self.records, "blimp")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(caught.exception.candidate_index, 0)
        self.assertEqual(caught.exception.task, "blimp")
        self.assertEqual(caught.exception.counters["model_forward_attempts"], 1)
        self.assertEqual(caught.exception.counters["model_forward_calls"], 0)
        self.assertGreater(caught.exception.counters["submitted_forward_input_tokens"], 0)
        self.assertEqual(caught.exception.counters["forward_input_tokens"], 0)
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)

    def test_temperature_numerical_overflow_retains_completed_forward(self):
        with self.assertRaisesRegex(CausalScoringError, "Nonfinite temperature") as caught:
            score_records(FakeLM(), self.processor, self.records, "blimp", temperatures=(1e-300,))
        self.assertEqual(caught.exception.counters["model_forward_attempts"], 1)
        self.assertEqual(caught.exception.counters["model_forward_calls"], 1)
        self.assertGreater(caught.exception.counters["forward_input_tokens"], 0)
        self.assertEqual(caught.exception.counters["scored_target_tokens"], 0)

    def test_wrapper_rejects_left_padding_or_nonbinary(self):
        adapter = HybridCausalAdapter(FakeLM())
        for mask in [torch.tensor([[0, 1]]), torch.tensor([[1, 2]]), torch.tensor([[0, 0]])]:
            with self.assertRaises(ValueError):
                adapter(torch.tensor([[2, 3]]), mask)

    def test_official_default_temperature_is_one(self):
        values = self.oracle["get_temperatures"](SimpleNamespace(min_temperature=1.0, max_temperature=None, temperature_interval=0.05))
        self.assertEqual(values, [1.0])

    def test_tiny_hybrid_sparse_padding_invariance_three_forwards(self):
        from src.babylm_hybrid.config import HybridConfig
        from src.babylm_hybrid.model import build_model
        model = build_model(HybridConfig(), "sparse", backbone_seed=1717, indexer_seed=37)
        def count(module, inputs):
            COUNTS["tiny_cpu_model_forward_calls"] += 1
            assert COUNTS["tiny_cpu_model_forward_calls"] <= 3
            assert not torch.is_grad_enabled()
        handle = model.register_forward_pre_hook(count)
        records = [{"sentences": ["a tiny model reads this text."], "completions": ["this text."]},
                   {"sentences": ["a short line."], "completions": ["line."]}]
        try:
            batched = score_records(model, self.processor, records, "blimp")
            for i, record in enumerate(records):
                individual = score_records(model, self.processor, [record], "blimp")
                torch.testing.assert_close(torch.tensor(batched["scores"][0][i]),
                                           torch.tensor(individual["scores"][0][0]), atol=5e-5, rtol=1e-5)
            self.assertTrue(model.training)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        finally:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tiny", action="store_true", help="Recheck fixture-only changes without spending additional real forwards")
    args = parser.parse_args()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CausalContractTests)
    if args.skip_tiny:
        suite = unittest.TestSuite(t for t in suite if "test_tiny_hybrid" not in t.id())
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    paths = [Path(__file__), ROOT / "src/babylm_hybrid/official_causal_scoring.py",
             ROOT / "src/babylm_hybrid/model.py", ROOT / "src/babylm_hybrid/config.py",
             ROOT / "src/babylm_hybrid/attention.py", SNAPSHOT / "manifest.json",
             ROOT / "data/babylm-2026-tokenizer-16k-v0/tokenizer.json"]
    record = {"utc": datetime.now(timezone.utc).isoformat(), "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "failures": [str(x) for x in result.failures],
              "errors": [str(x) for x in result.errors], "counters": COUNTS,
              "elapsed_seconds": time.perf_counter() - started, "upstream_commit": UPSTREAM_COMMIT,
              "source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "torch_version": torch.__version__, "device": "cpu",
              "skip_tiny": args.skip_tiny,
              "scope": "synthetic engineering conformance only; no official task scores"}
    out = ROOT / "results/babylm-official-causal-scoring-v0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
