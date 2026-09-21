"""Isolated W fixed-local training entry; frozen D/E engine is not edited.

The legacy engine's mode=dense means backbone-only optimizer compatibility.
The scientific condition is W_fixed_local, never a dense or learned-router run.
Default CLI validates only. The enclosing queue owns the GPU lock and hard limit.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITION = 'W_fixed_local'
ADMIN = {'max_wall_seconds', 'paid_ceiling_usd', 'hourly_rate_usd', 'stage_spent_usd', 'hard_timeout_seconds',
         'execution_hardware', 'actual_compute_hourly_rate_usd', 'protocol_document'}
LOCAL_SOURCE = 'src/babylm_hybrid/local_attention_v0.py'
SELF = 'scripts/run_babylm_local_train_v0.py'
REQUIRED = (SELF, 'scripts/run_babylm_local_eval_v0.py', LOCAL_SOURCE,
            'src/babylm_hybrid/config.py', 'src/babylm_hybrid/model.py', 'src/babylm_hybrid/attention.py',
            'src/babylm_hybrid/training.py', 'src/babylm_hybrid/evaluation.py',
            'scripts/prepare_babylm_windows_v0.py', 'scripts/run_babylm_de_v0.py',
            'scripts/run_babylm_checkpoint_eval_v0.py')
EXPECTED = {'windows': 22598, 'updates': 1413, 'scientific_updates': 1413, 'engineering_updates': 0,
            'forward_calls': 22598, 'backward_calls': 22598, 'input_tokens': 16325414,
            'loss_tokens': 16302816, 'word_exposures': 10001709}
FINAL_MODEL = 'model-u00001413-w000010001709-i000016325414-l000016302816.pt'


def require(ok, message):
    if not ok: raise ValueError(message)


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def verify_sources(master):
    sources = master['source_sha256']
    require(all(k in sources for k in REQUIRED), 'W source pins missing')
    for name, digest in sources.items():
        path = resolve(name)
        require(path.is_relative_to(ROOT) and sha(path) == digest, 'W source SHA differs: ' + name)
    return sources


def derive_child(master):
    base = resolve(master['reference_dense_protocol'])
    require(sha(base) == master['reference_dense_protocol_sha256'], 'Original dense protocol file differs')
    p = load(base)
    require(p['scope'] == 'scientific' and p['device'] == 'cuda' and p['dtype'] == 'float32', 'Original scientific FP32 CUDA protocol required')
    require(p['max_epochs'] == 1 and p['max_updates'] == 1413 and p['max_word_exposures'] == 10001709
            and p['windows_per_update'] == 16 and p['learning_rate'] == 3e-4, 'Original one-epoch schedule differs')
    require(p['model_config']['block_size'] == 4 and p['model_config']['selected_complete_blocks'] == 64
            and p['model_config']['index_score_scale'] == 1.0, 'Original unscaled D model config required')
    require(p.get('scientific_condition') is None, 'Reference must be the original D protocol')
    overrides = master['administrative_overrides']
    require(set(overrides) <= ADMIN, 'Only administrative overrides allowed')
    require(all(k in overrides for k in ['execution_hardware', 'actual_compute_hourly_rate_usd', 'protocol_document']),
            'Current W hardware/rate/protocol provenance must replace historical D metadata')
    child = copy.deepcopy(p); child.update(overrides)
    require(0 < child['max_wall_seconds'] <= 6900, 'W training cooperative wall exceeds allocation')
    child.update(scientific_condition=CONDITION, engine_compatibility_mode='dense', actual_policy='local', aux_weight=0.0,
                 engine_compatibility_note='Legacy dense mode selects backbone-only optimizer; model is fixed-local W, not dense attention.',
                 local_attention_contract={'selection': 'most_recent_complete_blocks_plus_original_causal_tail',
                                           'block_size': 4, 'selected_complete_blocks': 64,
                                           'indexer_present': False, 'auxiliary_objective_applied': False,
                                           'persistent_identity_buffer': 'local_attention_contract_v0'})
    purpose = master.get('purpose', 'scientific')
    require(purpose in {'scientific', 'gpu_preflight'}, 'Unknown W launch purpose')
    if purpose == 'gpu_preflight':
        require(0 < child['max_wall_seconds'] <= 600, 'W preflight wall exceeds 600 seconds')
        require(master.get('preflight_updates', 6) == 6, 'Only the fixed six-update preflight is permitted')
        child.update(scope='gpu_preflight', max_updates=6, checkpoint_every_updates=6,
                     milestone_word_exposures=[], eval_every_updates=0, eval_initial=False, eval_final=False,
                     preflight_not_scientific_training=True, preflight_discard_weights=True)
    return child


def validate(master):
    require(master['schema_version'] == 1 and master['condition'] == CONDITION
            and master['engine_compat_mode'] == 'dense', 'Explicit W scientific identity is required')
    verify_sources(master)
    require(len(master['expected_initial_parameter_hashes_sha256']) == 64, 'Frozen original D initialization hash required')
    child = derive_child(master)
    output = resolve(master['output_dir'])
    require(output.is_relative_to(ROOT / 'results'), 'W output must be under project results')
    return child


def configure_runtime():
    import torch
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def parameter_hashes(model):
    result = {}
    for name, parameter in model.named_parameters():
        meta = json.dumps({'shape': list(parameter.shape), 'dtype': str(parameter.dtype)},
                          sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
        result[name] = hashlib.sha256(meta + parameter.detach().contiguous().numpy().tobytes()).hexdigest()
    return result


def expected_lr(words, p):
    warm, cap, minimum = p['warmup_word_exposures'], p['max_word_exposures'], p['min_lr_ratio']
    multiplier = words / warm if warm and words < warm else minimum + (1 - minimum) * .5 * (
        1 + math.cos(math.pi * min(1., max(0., (words - warm) / max(cap - warm, 1)))))
    return p['learning_rate'] * multiplier


def expected_window_order(p):
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence([int(p['data_order_seed']), 0])).permutation(22598).tolist()


@contextmanager
def injected_factory(engine, expected_initial_hash=None, source_pins=None):
    """Process-local, explicit dependency injection; restore both symbols finally."""
    from src.babylm_hybrid.local_attention_v0 import build_local_model
    original_factory, original_sources = engine.build_model, engine.source_hashes
    def factory(cfg, mode, backbone_seed, indexer_seed):
        require(mode == 'dense', 'W uses only the backbone-only engine compatibility branch')
        model = build_local_model(cfg, mode, backbone_seed, indexer_seed)
        require(not any('.indexer.' in n for n, _ in model.named_parameters()), 'W unexpectedly created trainable indexer parameters')
        require('local_attention_contract_v0' in model.state_dict(), 'Persistent W checkpoint identity is missing')
        if expected_initial_hash is not None:
            require(canonical(parameter_hashes(model)) == expected_initial_hash, 'W initial backbone differs from original D')
        return model
    def sources():
        result = original_sources()
        for name, digest in (source_pins or {}).items():
            require(sha(resolve(name)) == digest, 'W source changed after launch: ' + name)
            result[str(resolve(name))] = digest
        return result
    engine.build_model, engine.source_hashes = factory, sources
    try: yield
    finally: engine.build_model, engine.source_hashes = original_factory, original_sources


def audit_training(run_dir, child, master, master_sha):
    summary = load(run_dir / 'summary.json')
    raw = (run_dir / 'events.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'Incomplete final W log')
    events = [json.loads(line) for line in raw.splitlines()]
    require([e['event_id'] for e in events] == list(range(1, len(events) + 1)), 'W event IDs differ')
    require(summary['status'] == 'epoch_complete' and summary['cursor'] == {'epoch': 1, 'position': 0}, 'W epoch incomplete')
    require(summary['protocol_sha256'] == canonical(child) and load(run_dir / 'protocol.json') == child, 'Executed W protocol differs')
    require(summary['optimizer_groups'] == ['backbone'] and summary['parameter_counts']['indexer'] == 0, 'W optimizer includes indexer')
    require(all(summary['counts'][k] == v for k, v in EXPECTED.items()), 'W epoch counts differ')
    require(canonical(summary['initial_parameter_hashes']) == master['expected_initial_parameter_hashes_sha256'], 'W initial backbone differs')
    require(not any(e['type'] in {'resume', 'run_resume', 'failure', 'consumed_without_update', 'eval_skipped_budget'} for e in events), 'Unexpected W resume/failure/skipped work')
    updates = [e for e in events if e['type'] == 'update']
    require([e['counts']['updates'] for e in updates] == list(range(1, 1414)), 'W update coverage differs')
    seen = []; totals = {k: 0 for k in ['word_exposures', 'input_tokens', 'loss_tokens']}
    for i, e in enumerate(updates, 1):
        require(set(e['lr']) == {'backbone'} and e['aux_weight_applied'] == 0
                and e['loss_token_weighted_aux'] == 0 and math.isfinite(e['loss_token_weighted_ce']), 'W is not pure LM backbone-only training')
        require(e['lr_word_position'] == e['counts']['word_exposures']
                and abs(e['lr']['backbone'] - expected_lr(e['lr_word_position'], child)) <= 1e-15, 'W learning-rate trajectory differs')
        require(len(e['windows']) == (6 if i == 1413 else 16), 'W accumulation count differs')
        for w in e['windows']:
            require(w['epoch'] == 0 and all(w[k] is True for k in ['forward_started', 'forward_completed', 'backward_started', 'backward_completed']), 'W incomplete training window')
            seen.append(w['window_index'])
            for k in totals: totals[k] += w[k]
        require(all(e['counts'][k] == v for k, v in totals.items()), 'W cumulative window ledger differs')
    require(len(seen) == len(set(seen)) == 22598 and set(seen) == set(range(22598)), 'W full window coverage differs')
    require(seen == expected_window_order(child), 'W frozen window order differs')
    panels = [e for e in events if e['type'] == 'evaluation']
    require([e['counts']['updates'] for e in panels] == [0, 250, 500, 750, 1000, 1250, 1413]
            and summary['eval_counts']['evaluations'] == 7 and summary['eval_counts']['forward_calls'] == 336, 'W frozen panel cadence differs')
    final = run_dir / 'snapshots' / FINAL_MODEL
    final_receipt = final.with_name(final.stem + '-final-epoch_complete.json')
    r = load(final_receipt)
    require(final.is_file() and r['stop_reason'] == 'epoch_complete' and r['point']['updates'] == 1413, 'W unique final1413 snapshot missing')
    # Model/receipt digest correspondence is checked against both engine snapshot records.
    entries = [e for e in summary['snapshot_state']['models'] if e['point']['updates'] == 1413]
    require(len(entries) == 1 and sha(final) == entries[0]['sha256'], 'W final model SHA differs')
    final_entries = [e for e in summary['snapshot_state']['final_receipts'] if e['point']['updates'] == 1413 and e['stop_reason'] == 'epoch_complete']
    require(len(final_entries) == 1 and final_entries[0]['sha256'] == sha(final_receipt)
            and r['model_sha256'] == entries[0]['sha256'] and r['model_path'] == entries[0]['path']
            and r['protocol_sha256'] == canonical(child) and r['counts'] == summary['counts']
            and r['point'] == entries[0]['point'], 'W final receipt/checkpoint/summary binding differs')
    return {'status': 'complete_local_epoch_audited', 'condition': CONDITION, 'engine_compat_mode': 'dense',
            'master_protocol_sha256': master_sha, 'protocol_sha256': canonical(child), 'source_sha256': master['source_sha256'],
            'counts': summary['counts'], 'eval_counts': summary['eval_counts'], 'initial_parameter_hashes_sha256': canonical(summary['initial_parameter_hashes']),
            'final_model_checkpoint_path': str(final), 'final_model_checkpoint_sha256': sha(final),
            'final_model_receipt_path': str(final_receipt), 'final_model_receipt_sha256': sha(final_receipt),
            'events_sha256': sha(run_dir / 'events.jsonl'), 'summary_sha256': sha(run_dir / 'summary.json'),
            'model_factory': LOCAL_SOURCE + ':build_local_model', 'scientific_claim': 'One W baseline; no quality or speed claim from completion.'}


def audit_preflight(run_dir, child, master, master_sha):
    s = load(run_dir / 'summary.json')
    events = [json.loads(line) for line in (run_dir / 'events.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['type'] == 'update']
    require(s['status'] == 'max_updates_reached' and [e['counts']['updates'] for e in updates] == list(range(1, 7)), 'W preflight did not finish its six engineering updates')
    require(s['counts']['engineering_updates'] == 6 and s['counts']['scientific_updates'] == 0
            and s['counts']['forward_calls'] == s['counts']['backward_calls'] == 96
            and s['eval_counts']['forward_calls'] == 0, 'W preflight compute ledger differs')
    require(all(len(e['windows']) == 16 and all(w['forward_completed'] and w['backward_completed'] for w in e['windows']) for e in updates)
            and sum(len(e['windows']) for e in updates) == 96, 'Raw W preflight F/B count differs')
    require([w['window_index'] for e in updates for w in e['windows']] == expected_window_order(child)[:96], 'W preflight frozen data prefix differs')
    require(all(e['lr_word_position'] == e['counts']['word_exposures'] and
                abs(e['lr']['backbone'] - expected_lr(e['lr_word_position'], child)) <= 1e-15 for e in updates), 'W preflight LR prefix differs')
    require(s['optimizer_groups'] == ['backbone'] and s['parameter_counts']['indexer'] == 0
            and canonical(s['initial_parameter_hashes']) == master['expected_initial_parameter_hashes_sha256'], 'W preflight initialization/optimizer differs')
    require(all(math.isfinite(e['loss_token_weighted_ce']) and e['loss_token_weighted_aux'] == 0 and e['aux_weight_applied'] == 0 for e in updates), 'W preflight loss is nonfinite or has auxiliary objective')
    return {'status': 'complete_local_gpu_preflight_audited', 'condition': CONDITION, 'purpose': 'gpu_preflight',
            'engine_compat_mode': 'dense', 'actual_policy': 'local', 'master_protocol_sha256': master_sha,
            'protocol_sha256': canonical(child), 'source_sha256': master['source_sha256'],
            'counts': s['counts'], 'eval_counts': s['eval_counts'],
            'initial_parameter_hashes_sha256': canonical(s['initial_parameter_hashes']),
            'elapsed_wall_seconds': s['elapsed_wall_seconds'], 'update_seconds': [e['update_seconds'] for e in updates],
            'not_scientific_training': True, 'discard_weights_for_fresh_scientific_start': True,
            'events_sha256': sha(run_dir / 'events.jsonl'), 'summary_sha256': sha(run_dir / 'summary.json')}


def execute(path, expected_sha):
    path = resolve(path); require(sha(path) == expected_sha, 'Frozen W train master SHA differs')
    master = load(path); require(master.get('launch_allowed') is True, 'W launch not frozen')
    child = validate(master); output = resolve(master['output_dir'])
    require(not output.exists(), 'Fresh W output required; no automatic resume')
    output.mkdir(parents=True); write(output / 'child-protocol.json', child)
    receipt = {'status': 'running', 'condition': CONDITION, 'engine_compat_mode': 'dense',
               'master_protocol_sha256': expected_sha, 'protocol_sha256': canonical(child), 'source_sha256': master['source_sha256'],
               'started_utc': datetime.now(timezone.utc).isoformat(), 'mode_warning': child['engine_compatibility_note'],
               'gpu_lock_owner': 'enclosing finite queue', 'automatic_resume': False}
    write(output / 'local-training-audit.json', receipt)
    try:
        from src.babylm_hybrid import training
        configure_runtime()
        with injected_factory(training, master['expected_initial_parameter_hashes_sha256'], master['source_sha256']):
            training.train_run(child, 'dense', output / 'run')
        audit = audit_preflight if master.get('purpose', 'scientific') == 'gpu_preflight' else audit_training
        receipt.update(audit(output / 'run', child, master, expected_sha))
    except BaseException as e:
        receipt.update(status='local_training_failed', error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        if (output / 'run/summary.json').exists():
            s = load(output / 'run/summary.json'); receipt.update(counts=s.get('counts'), eval_counts=s.get('eval_counts'))
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat(); write(output / 'local-training-audit.json', receipt)
    print(json.dumps({k: receipt.get(k) for k in ['status', 'condition', 'counts', 'eval_counts']}, allow_nan=False))
    return 0 if receipt['status'] in {'complete_local_epoch_audited', 'complete_local_gpu_preflight_audited'} else 2


def tiny_cpu_self_test(output_dir):
    """Synthetic two-window optimizer update and one-window scorer integration."""
    from unittest import mock
    import numpy as np
    import torch
    from src.babylm_hybrid import training
    from src.babylm_hybrid.config import HybridConfig
    from src.babylm_hybrid.model import build_model
    from scripts import run_babylm_checkpoint_eval_v0 as evaluator
    from scripts import run_babylm_local_eval_v0 as local_eval
    output = resolve(output_dir)
    require(output.is_relative_to(ROOT / 'logs') and not output.exists(), 'Fresh local engineering log directory required')
    output.mkdir(parents=True); configure_runtime()
    counts = {'training_forward_calls': 0, 'evaluation_forward_calls': 0, 'backward_calls': 0,
              'optimizer_updates': 0, 'scientific_forward_calls': 0, 'cuda_calls': 0}
    receipt = {'status': 'running', 'engineering_counts': counts, 'source_sha256': {name: sha(resolve(name)) for name in REQUIRED}}
    try:
        class Dataset:
            fingerprint = {'synthetic_engineering_only': True}
            def __len__(self): return 2
            def window(self, index):
                return {'window_index': index, 'input_ids': np.asarray([(index + i) % 17 for i in range(8)], dtype=np.uint32),
                        'word_exposures': 4, 'input_tokens': 8, 'next_token_loss_positions': 7, 'loss_tokens': 7,
                        'source_index': 0, 'source': 'synthetic', 'segment_index_in_source': index,
                        'source_token_start': 0, 'source_token_end': 8, 'single_segment': True, 'reset_model_state_before': True}
        dataset = Dataset()
        manifest_path = output / 'synthetic-manifest.json'
        manifest = {'total_windows': 1, 'source_summaries': [{'source_index': 0, 'source': 'synthetic'}]}
        write(manifest_path, manifest)
        cfg = HybridConfig(vocab_size=17, hidden_size=16, intermediate_size=32, layer_types=('global',),
                           global_q_heads=2, global_kv_heads=1, selected_complete_blocks=1)
        dense = build_model(cfg, 'dense', 13, 17)
        init_hash = canonical(parameter_hashes(dense))
        p = {'schema_version': 1, 'scope': 'engineering_smoke', 'launch_allowed': True, 'device': 'cpu', 'dtype': 'float32',
             'model_config': cfg.to_dict(), 'backbone_seed': 13, 'indexer_seed': 17, 'data_order_seed': 19,
             'train_manifest': str(manifest_path), 'train_manifest_sha256': sha(manifest_path), 'max_epochs': 1,
             'max_word_exposures': 8, 'max_updates': 1, 'max_wall_seconds': 60, 'windows_per_update': 2,
             'learning_rate': 3e-4, 'indexer_learning_rate': 1e-3, 'weight_decay': .1, 'betas': [.9, .95], 'eps': 1e-8,
             'grad_clip_norm': 1.0, 'gradient_clip_scope': 'separate_backbone_indexer', 'aux_weight': 0.0,
             'warmup_word_exposures': 4, 'min_lr_ratio': .1, 'checkpoint_every_updates': 1, 'eval_every_updates': 0,
             'eval_initial': False, 'eval_final': False, 'milestone_word_exposures': [],
             'scientific_condition': CONDITION, 'engine_compatibility_mode': 'dense', 'actual_policy': 'local'}
        before_factory, before_sources = training.build_model, training.source_hashes
        with mock.patch.object(training, 'load_window_dataset', return_value=dataset), \
             injected_factory(training, init_hash, receipt['source_sha256']):
            result = training.train_run(p, 'dense', output / 'train')
        counts.update(training_forward_calls=result['counts']['forward_calls'], backward_calls=result['counts']['backward_calls'],
                      optimizer_updates=result['counts']['updates'])
        require(training.build_model is before_factory and training.source_hashes is before_sources, 'Training injection was not restored')
        require(result['status'] == 'epoch_complete' and result['optimizer_groups'] == ['backbone']
                and result['parameter_counts']['indexer'] == 0 and counts['training_forward_calls'] == counts['backward_calls'] == 2, 'Tiny optimizer integration failed')
        final = result['snapshot_state']['models'][-1]
        cp = output / 'train' / final['path']
        state = torch.load(cp, map_location='cpu', weights_only=True)
        rejected = False
        try: dense.load_state_dict(state['model_state'], strict=True)
        except RuntimeError: rejected = True
        require(rejected, 'Old dense model accepted W checkpoint')
        ev = {'schema_version': 1, 'scope': 'engineering', 'mode': 'dense', 'device': 'cpu', 'dtype': 'float32',
              'checkpoint_path': str(cp), 'checkpoint_sha256': sha(cp), 'checkpoint_format': 'model_only',
              'checkpoint_protocol_sha256': canonical(p), 'model_config': cfg.to_dict(), 'backbone_seed': 13, 'indexer_seed': 17,
              'dev_manifest': str(manifest_path), 'dev_manifest_sha256': sha(manifest_path), 'window_indices': [0],
              'position_diagnostics': True, 'enable_routing_diagnostics': False, 'torch_num_threads': 1,
              'expected_source_hashes': receipt['source_sha256'], 'max_wall_seconds': 60, 'output_dir': str(output / 'evaluation')}
        old_eval_factory = evaluator.build_model
        with mock.patch.object(evaluator, 'verify_manifest', return_value=(manifest, {'synthetic_engineering_only': True})), \
             mock.patch.object(evaluator.window_reader, 'iter_windows', side_effect=lambda *a, **k: iter([dataset.window(0)])), \
             local_eval.injected_factory(evaluator):
            scored = evaluator.run_evaluation(ev)
        counts['evaluation_forward_calls'] = scored['counts']['forward_calls']
        require(evaluator.build_model is old_eval_factory, 'Evaluation injection was not restored')
        require(scored['status'] == 'evaluation_complete' and counts['evaluation_forward_calls'] == 1
                and scored['backward_calls'] == scored['optimizer_updates'] == 0, 'Tiny scorer integration failed: ' + str(scored.get('error')))
        counts['model_forward_calls'] = counts['training_forward_calls'] + counts['evaluation_forward_calls']
        receipt.update(status='passed', original_dense_strict_load_rejected=True, shared_initial_parameters_verified=True,
                       train_factory_restored=True, eval_factory_restored=True, last_learning_rates=result.get('last_lr'),
                       one_window_heldout_NLL=scored['total']['nll'])
    except BaseException as e:
        receipt.update(status='failed', error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
    write(output / 'wrapper-integration-receipt.json', receipt)
    print(json.dumps(receipt, allow_nan=False))
    return 0 if receipt['status'] == 'passed' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol'); p.add_argument('--protocol-sha256'); p.add_argument('--execute', action='store_true')
    p.add_argument('--tiny-cpu-self-test', action='store_true'); p.add_argument('--test-output')
    args = p.parse_args()
    if args.tiny_cpu_self_test:
        require(args.test_output and not args.execute and not args.protocol, 'Tiny test needs only its fresh --test-output')
        return tiny_cpu_self_test(args.test_output)
    require(bool(args.protocol), '--protocol required outside the explicit engineering self-test')
    if args.execute:
        require(bool(args.protocol_sha256), 'Explicit frozen protocol SHA required')
        return execute(args.protocol, args.protocol_sha256)
    child = validate(load(resolve(args.protocol)))
    print(json.dumps({'status': 'static_validation_passed', 'condition': CONDITION, 'child_protocol_sha256': canonical(child), 'model_calls': 0}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
