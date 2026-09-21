"""No-model AST/pure-administrative tests of the isolated time-bound engine."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'src/babylm_hybrid/training.py'
NEW = ROOT / 'src/babylm_hybrid/training_timebound_v0.py'
ORIGINAL_SHA = '3c2af088027a689fe6281b7ee1c4ab8d27dcf156278449df5c553e95a85ab98f'


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8'))


def function(parsed, name):
    return next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == name)


def validation(path):
    namespace = {'math': math}
    module = ast.Module(body=[copy.deepcopy(function(tree(path), '_validate_protocol'))], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['_validate_protocol']


def budget_helpers(path, protocol, seconds, log_path=None):
    # Compile only copied administrative closures, never import torch or the engine.
    training = function(tree(path), 'train_run')
    wanted = ('estimated_paid_cost', 'cost_time_reason', 'append')
    functions = [copy.deepcopy(node) for node in training.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    outer = ast.parse('def factory(p, log_path):\n    event_id = 0\n    mode = "sparse"\n    pass\n    return estimated_paid_cost, cost_time_reason, append\n')
    outer.body[0].body[2:3] = functions
    namespace = {'elapsed': lambda: seconds, '_utc': lambda: 'fixed-test-utc', 'os': os,
                 '_canonical': lambda value: json.dumps(value, allow_nan=False).encode('utf-8')}
    exec(compile(ast.fix_missing_locations(outer), str(path), 'exec'), namespace)
    return namespace['factory'](protocol, log_path)


def protocol(**changes):
    result = dict(schema_version=1, scope='scientific', launch_allowed=True, device='cuda', dtype='float32',
                  model_config={}, backbone_seed=1, indexer_seed=2, data_order_seed=3,
                  train_manifest='not-read-by-validation.json', train_manifest_sha256='0' * 64,
                  max_word_exposures=1000, max_updates=20, max_wall_seconds=3600,
                  windows_per_update=16, learning_rate=3e-4, indexer_learning_rate=1e-3,
                  weight_decay=0.1, betas=[0.9, 0.95], eps=1e-8, grad_clip_norm=1,
                  aux_weight=1, warmup_word_exposures=100, min_lr_ratio=0.1,
                  checkpoint_every_updates=10, eval_every_updates=0,
                  hourly_rate_usd=0.53, paid_ceiling_usd=10, stage_spent_usd=0)
    result.update(changes)
    return result


def unknown(**changes):
    return protocol(**dict(cost_mode='time_bounded_unknown_rate', hourly_rate_usd=None,
                           paid_ceiling_usd=None, **changes))


class TimeboundAdministrativeTests(unittest.TestCase):
    def test_frozen_original_sha(self):
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), ORIGINAL_SHA)

    def test_unknown_cuda_protocol_validates(self):
        validation(NEW)(unknown())
        validation(NEW)(unknown(scope='gpu_preflight', max_updates=12, max_wall_seconds=1800))

    def test_unknown_requires_finite_positive_nonboolean_time(self):
        for value in (None, False, True, 0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                validation(NEW)(unknown(max_wall_seconds=value))

    def test_unknown_requires_explicit_null_rate_and_ceiling(self):
        for key in ('hourly_rate_usd', 'paid_ceiling_usd'):
            for value in (0, 1, 'unknown'):
                bad = unknown(); bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validation(NEW)(bad)
            bad = unknown(); del bad[key]
            with self.assertRaises(ValueError): validation(NEW)(bad)

    def test_unknown_does_not_bypass_scope_or_preflight_caps(self):
        for changes in ({'device': 'cpu'}, {'scope': 'engineering_smoke'},
                        {'scope': 'gpu_preflight', 'max_updates': 13, 'max_wall_seconds': 1800},
                        {'scope': 'gpu_preflight', 'max_updates': 12, 'max_wall_seconds': 1801}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): validation(NEW)(unknown(**changes))

    def test_known_paid_validation_matches_original(self):
        cases = [protocol(), protocol(scope='engineering_smoke', device='cpu', max_updates=5)]
        for key in ('hourly_rate_usd', 'paid_ceiling_usd', 'stage_spent_usd', 'max_wall_seconds'):
            for value in (None, 0, -1, float('nan'), float('inf'), 10, 20):
                p = protocol(); p[key] = value; cases.append(p)
        def result(validate, p):
            try: validate(p); return 'passed'
            except (ValueError, TypeError) as error: return type(error).__name__, str(error)
        for p in cases:
            with self.subTest(p=p): self.assertEqual(result(validation(ORIGINAL), p), result(validation(NEW), p))

    def test_known_paid_estimate_and_bound_match_original(self):
        for seconds in (0, 100, 3599, 3600, 100000):
            for spent in (0, 9.5, 9.999):
                p = protocol(stage_spent_usd=spent)
                a = budget_helpers(ORIGINAL, p, seconds); b = budget_helpers(NEW, p, seconds)
                self.assertEqual(a[0](), b[0]()); self.assertEqual(a[1](), b[1]())

    def test_unknown_returns_none_and_time_limit_still_stops(self):
        p = unknown(stage_spent_usd=None)
        for seconds, expected in ((0, None), (3599.9, None), (3600, 'max_wall_seconds_reached'), (99999, 'max_wall_seconds_reached')):
            estimate, reason, _ = budget_helpers(NEW, p, seconds)
            self.assertIsNone(estimate()); self.assertEqual(reason(), expected)

    def test_unknown_event_serializes_explicit_nulls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            estimate, _, append = budget_helpers(NEW, unknown(), 10, path)
            append('update', counts={'updates': 1}, estimated_gpu_usd=estimate())
            row = json.loads(path.read_text())
            self.assertIsNone(row['estimated_gpu_usd']); self.assertIsNone(row['hourly_rate_usd'])
            self.assertIsNone(row['paid_ceiling_usd']); self.assertEqual(row['cost_mode'], 'time_bounded_unknown_rate')
            self.assertEqual(row['counts']['updates'], 1)

    def test_known_event_structure_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('old.jsonl', 'new.jsonl')]
            for engine, path in zip((ORIGINAL, NEW), paths):
                estimate, _, append = budget_helpers(engine, protocol(), 10, path)
                append('update', counts={'updates': 1}, estimated_gpu_usd=estimate())
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_all_nonadministrative_top_level_code_is_identical(self):
        def reduced(parsed):
            return [node for node in parsed.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
                    and not (isinstance(node, ast.FunctionDef) and node.name in ('_validate_protocol', 'train_run'))]
        self.assertEqual(ast.dump(ast.Module(body=reduced(tree(ORIGINAL)), type_ignores=[])),
                         ast.dump(ast.Module(body=reduced(tree(NEW)), type_ignores=[])))

    def test_scientific_train_body_identical_outside_budget_and_cost_metadata(self):
        class Normalize(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name in ('estimated_paid_cost', 'cost_time_reason', 'append'):
                    return ast.FunctionDef(name=node.name, args=node.args, body=[ast.Pass()], decorator_list=[])
                return self.generic_visit(node)
            def visit_Dict(self, node):
                self.generic_visit(node)
                pairs = [(k, v) for k, v in zip(node.keys, node.values)
                         if not (isinstance(k, ast.Constant) and k.value in ('cost_mode', 'cost_limitations'))]
                node.keys = [k for k, _ in pairs]; node.values = [v for _, v in pairs]
                return node
        original = Normalize().visit(copy.deepcopy(function(tree(ORIGINAL), 'train_run')))
        new = Normalize().visit(copy.deepcopy(function(tree(NEW), 'train_run')))
        self.assertEqual(ast.dump(original), ast.dump(new))


if __name__ == '__main__': unittest.main()
