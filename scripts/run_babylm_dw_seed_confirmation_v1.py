"""One frozen fresh-seed D/W epoch pair and full dev; no retry or resume.

The outer timeout owns a dedicated process group. This parent holds the GPU
flock; all workers inherit that group. Default validation is model-free.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_babylm_local_train_v0 as common

SELF = 'scripts/run_babylm_dw_seed_confirmation_v1.py'
ENGINE = 'src/babylm_hybrid/training_timebound_v0.py'
CONDITIONS = {'D': 'D_dense', 'W': 'W_fixed_local'}
STAGES = ('train_D', 'train_W', 'eval_D', 'eval_W')
require, sha, canonical, load, write = common.require, common.sha, common.canonical, common.load, common.write
ADMIN = set(common.ADMIN) | {
    'created_utc', 'frozen_utc', 'notes', 'status', 'source_sha256', 'cost_mode',
    'budget_reconciled', 'execution_amendment', 'execution_order',
    'execution_parent_protocol', 'execution_parent_protocol_sha256',
    'execution_service_exception', 'parent_protocol', 'parent_protocol_sha256',
    'pair_stage_ceiling_usd', 'authorization', 'authorization_sha256',
    'condition_pair', 'seed_scope',
}
IDENTITY = {'scientific_condition', 'engine_compatibility_mode', 'engine_compatibility_note',
            'actual_policy', 'local_attention_contract'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    p = Path(value)
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    require(p.is_relative_to(ROOT), 'Path outside project')
    return p


def verify_pins(pins):
    require(all(n in pins for n in (SELF, ENGINE, *common.REQUIRED,
            'scripts/summarize_babylm_one_epoch_v0.py')), 'Required source pins missing')
    for name, digest in pins.items():
        require(sha(resolve(name)) == digest, 'Source SHA changed: ' + name)


def validate_training(p, reference, master, arm):
    allowed = ADMIN | IDENTITY | {'backbone_seed'} | ({'aux_weight'} if arm == 'W' else set())
    differences = [k for k in set(p) | set(reference)
                   if k not in allowed and p.get(k) != reference.get(k)]
    require(not differences, 'Unapproved scientific changes: ' + repr(differences))
    require(p['scope'] == 'scientific' and p['device'] == 'cuda' and p['dtype'] == 'float32', 'Scientific CUDA FP32 required')
    require(p['launch_allowed'] is True and p['backbone_seed'] == master['backbone_seed'], 'Fresh shared seed required')
    require(p['scientific_condition'] == CONDITIONS[arm] and p['actual_policy'] == ('dense' if arm == 'D' else 'local')
            and p['engine_compatibility_mode'] == 'dense', 'Incorrect condition/factory identity')
    require(p['source_sha256'] == master['source_sha256'], 'Training source pins differ')
    require(p.get('condition_pair') == 'D_W_new_initialization_confirmation'
            and p.get('seed_scope') == 'initialization_only_data_order_unchanged', 'Pair identity differs')
    require(p['max_epochs'] == 1 and p['max_updates'] == 1413 and p['max_word_exposures'] == 10001709
            and p['training_initialization'] == 'fresh_random_shared_backbone_no_resume', 'Only fresh complete one epoch allowed')
    require(0 < p['max_wall_seconds'] <= master['train_timeout_seconds'] - 100, 'Training soft/hard margin missing')
    require(p['cost_mode'] == 'time_bounded_unknown_rate' and p['hourly_rate_usd'] is None
            and p['actual_compute_hourly_rate_usd'] is None, 'Unknown rate must remain explicit')
    require(p['execution_hardware']['physical_uuid'] == master['hardware']['physical_uuid']
            and p['execution_hardware']['driver_version'] == master['hardware']['driver_version'], 'Training hardware differs')
    if arm == 'W':
        require(p['aux_weight'] == 0 and p['local_attention_contract']['block_size'] == 4
                and p['local_attention_contract']['selected_complete_blocks'] == 64
                and p['local_attention_contract']['selection'] == 'most_recent_complete_blocks_plus_original_causal_tail'
                and p['local_attention_contract']['indexer_present'] is False, 'Original pure local W required')


def validate(master):
    require(master['schema_version'] == 1 and master['launch_allowed'] is True, 'Frozen master required')
    require(master['authorized_conditions'] == ['D', 'W'] and master['automatic_monitoring'] == 'PAUSED', 'Only authorized D/W, monitoring paused')
    require(master['backbone_seed'] == 20260921 and master['old_backbone_seed'] == 20260917, 'Frozen fresh seed pair required')
    require((master['train_timeout_seconds'], master['eval_timeout_seconds'], master['hard_timeout_seconds'])
            == (21600, 7200, 58200), 'Frozen stage/global limits differ')
    require(len(master['expected_initial_parameter_hashes_sha256']) == 64, 'Initial parameter digest required')
    verify_pins(master['source_sha256'])
    bound = {}
    for field in ('reference_dense_protocol', 'training_protocol_D', 'training_protocol_W', 'eval_template', 'authorization', 'initialization_receipt'):
        path = resolve(master[field])
        require(sha(path) == master[field + '_sha256'], 'Frozen binding changed: ' + field)
        bound[field] = load(path)
    initial = bound['initialization_receipt']
    require(initial['status'] == 'paired_new_initialization_verified' and initial['D_W_exact_equal'] is True
            and initial['old_seed_different'] is True and initial['backbone_seed'] == master['backbone_seed']
            and initial['initial_parameter_hashes_sha256'] == master['expected_initial_parameter_hashes_sha256'], 'Shared initialization receipt differs')
    reference = bound['reference_dense_protocol']
    require(reference['backbone_seed'] == master['old_backbone_seed'] and reference.get('scientific_condition') is None, 'Original D reference required')
    for name, digest in reference['source_sha256'].items():
        require(master['source_sha256'].get(name) == digest, 'Frozen original scientific source changed: ' + name)
    for arm in CONDITIONS:
        validate_training(bound['training_protocol_' + arm], reference, master, arm)
    template = bound['eval_template']
    require(template['scope'] == 'scientific_evaluation' and template['mode'] == 'dense'
            and template['device'] == 'cuda' and template['dtype'] == 'float32'
            and template['checkpoint_format'] == 'model_only', 'Original full FP32 scorer required')
    require(template['model_config'] == reference['model_config'] and template['indexer_seed'] == reference['indexer_seed'], 'Evaluation model config differs')
    require(template['dev_manifest'] == reference['eval_manifest'] and template['dev_manifest_sha256'] == reference['eval_manifest_sha256'], 'Fixed dev changed')
    require(template.get('window_indices') is None and template['position_diagnostics'] is True
            and template.get('enable_routing_diagnostics', False) is False
            and template.get('routing_policy', 'learned') == 'learned'
            and template['torch_num_threads'] == 1, 'No subset/routing/scoring changes allowed')
    require(template['max_wall_seconds'] == 7100 and 'absolute_deadline_utc' not in template, 'Fresh bounded dev allocation required')
    require(template['expected_source_hashes'] == master['source_sha256'], 'Evaluation source pins differ')
    require(template['expected_runtime']['gpu']['uuid'].removeprefix('GPU-') == master['hardware']['physical_uuid'].removeprefix('GPU-'), 'Evaluation GPU differs')
    out = resolve(master['output_dir'])
    require(out.parent == ROOT / 'results', 'Dedicated top-level result directory required')
    return out, bound


@contextmanager
def dense_factory(engine, initial_sha, pins):
    original_factory, original_sources = engine.build_model, engine.source_hashes
    def factory(cfg, mode, backbone_seed, indexer_seed):
        require(mode == 'dense', 'D requires dense factory')
        model = original_factory(cfg, mode, backbone_seed, indexer_seed)
        require('local_attention_contract_v0' not in model.state_dict(), 'D contains W identity')
        require(canonical(common.parameter_hashes(model)) == initial_sha, 'D shared initialization differs')
        return model
    def sources():
        verify_pins(pins)
        actual = original_sources()
        actual.update({str(resolve(n)): d for n, d in pins.items()})
        return actual
    engine.build_model, engine.source_hashes = factory, sources
    try:
        yield
    finally:
        engine.build_model, engine.source_hashes = original_factory, original_sources


def audit_train(run, protocol, master, master_sha, arm):
    # The frozen common audit's checks are backbone-only and valid for both arms.
    # Its W-specific return labels are replaced, never published as D identity.
    result = common.audit_training(run, protocol, master, master_sha)
    from scripts.summarize_babylm_one_epoch_v0 import summarize
    tail = summarize(run, 100)
    result.update(status='complete_seed_epoch_audited', condition=CONDITIONS[arm], arm=arm,
                  actual_policy='dense' if arm == 'D' else 'local',
                  model_factory=('src/babylm_hybrid/model.py:build_model' if arm == 'D'
                                 else common.LOCAL_SOURCE + ':build_local_model'),
                  backbone_seed=master['backbone_seed'], tail100=tail['tail'],
                  tail100_audit=tail, scientific_claim='Single new matched seed; no quality or speed conclusion from completion.')
    summary = load(run / 'summary.json')
    for name, digest in master['source_sha256'].items():
        matches = [v for k, v in summary['source_hashes'].items() if k.replace('\\', '/').endswith(name)]
        require(matches and all(v == digest for v in matches), 'Training provenance missing: ' + name)
    return result


def evaluation_protocol(master, out, bound, arm):
    ap = out / arm / 'train' / 'audit.json'
    audit = load(ap)
    require(audit['status'] == 'complete_seed_epoch_audited' and audit['arm'] == arm
            and audit['condition'] == CONDITIONS[arm], 'Matching training audit required')
    require(all(audit['counts'][k] == v for k, v in common.EXPECTED.items()), 'Incomplete training counts')
    require(audit['initial_parameter_hashes_sha256'] == master['expected_initial_parameter_hashes_sha256'], 'Shared seed identity differs')
    cp = out / arm / 'train' / 'run' / 'snapshots' / common.FINAL_MODEL
    protocol = bound['training_protocol_' + arm]
    require(resolve(audit['final_model_checkpoint_path']) == cp and sha(cp) == audit['final_model_checkpoint_sha256']
            and audit['protocol_sha256'] == canonical(protocol), 'Unique final1413 checkpoint binding failed')
    child = copy.deepcopy(bound['eval_template'])
    child.update(condition=CONDITIONS[arm], actual_policy='dense' if arm == 'D' else 'local', engine_compat_mode='dense',
                 backbone_seed=master['backbone_seed'], checkpoint_path=str(cp.relative_to(ROOT)),
                 checkpoint_sha256=sha(cp), checkpoint_protocol_sha256=canonical(protocol),
                 output_dir=str((out / arm / 'eval').relative_to(ROOT)), training_audit_path=str(ap.relative_to(ROOT)),
                 training_audit_sha256=sha(ap), launch_allowed=True)
    return child


def bind_evaluation(master, out, bound, arm):
    child = evaluation_protocol(master, out, bound, arm)
    ap = resolve(child['training_audit_path'])
    target = out / arm / 'bound-eval.json'
    require(not target.exists(), 'Evaluation binding exists; no retries')
    write(target, child)
    write(out / arm / 'checkpoint-binding.json', {'utc': utc(), 'arm': arm, 'master_protocol_sha256': canonical(master),
          'template_file_sha256': master['eval_template_sha256'], 'checkpoint_path': child['checkpoint_path'],
          'checkpoint_sha256': child['checkpoint_sha256'], 'training_audit_sha256': sha(ap),
          'selection': 'Only prespecified final1413; no score selection', 'model_calls': 0})
    return target


def audit_eval(directory, protocol, arm):
    s = load(directory / 'summary.json')
    require(s['status'] == 'evaluation_complete' and not s['partial_metrics_only'], 'Full dev incomplete')
    require(s['counts']['forward_attempts'] == s['counts']['forward_calls'] == s['counts']['committed_windows'] == 18792
            and s['optimizer_updates'] == s['backward_calls'] == 0, 'Full dev work counts differ')
    require(s['observed_window_indices'] == list(range(18792)) and s['protocol_sha256'] == canonical(protocol), 'Full dev protocol/coverage differs')
    require(s['runtime'] == protocol['expected_runtime'] and s['source_hashes'] == protocol['expected_source_hashes'], 'Full dev runtime/source differs')
    require(s['checkpoint']['sha256'] == protocol['checkpoint_sha256'] and s['checkpoint']['point']['updates'] == 1413, 'Full dev checkpoint differs')
    raw = (directory / 'windows.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'Incomplete dev JSONL')
    rows = [json.loads(line) for line in raw.splitlines()]
    require([r['window_index'] for r in rows] == list(range(18792)), 'Missing/repeated dev rows')
    targets = sum(r['loss_tokens'] for r in rows); inputs = sum(r['input_tokens'] for r in rows)
    require(targets == s['total']['loss_tokens'] == 17418742 and inputs == s['total']['input_tokens'] == 17437534, 'Full dev denominator changed')
    for row in rows:
        require(row['forward_calls'] == 1 and row['grad_enabled'] is False
                and row['model_training'] is False, 'Unexpected dev work')
        if row['loss_tokens'] == 0:
            require(row['nll'] is None and row['ppl'] is None and row['nll_sum'] == 0, 'Zero-target dev row differs')
        else:
            require(math.isfinite(row['nll']) and abs(row['nll'] * row['loss_tokens'] - row['nll_sum']) <= 1e-6
                    and math.isclose(math.exp(row['nll']), row['ppl'], rel_tol=1e-12), 'Invalid dev NLL/PPL row')
    nll = math.fsum(r['nll_sum'] for r in rows) / targets
    require(abs(nll - s['total']['nll']) <= 1e-10 and math.isclose(math.exp(nll), s['total']['ppl'], rel_tol=1e-10), 'Token-weighted dev aggregate differs')
    return {'status': 'complete_seed_full_dev_audited', 'arm': arm, 'condition': CONDITIONS[arm],
            'protocol_sha256': canonical(protocol), 'checkpoint_sha256': protocol['checkpoint_sha256'],
            'counts': s['counts'], 'total': s['total'], 'source_sha256': s['source_hashes'],
            'summary_sha256': sha(directory / 'summary.json'), 'windows_sha256': sha(directory / 'windows.jsonl'),
            'claim': 'Existing development set, one additional seed; independent final comparison still required.'}


def worker(master_path, master_sha, stage):
    require(os.name == 'posix' and os.getppid() == int(os.environ.get('BABYLM_DW_QUEUE_PID', '-1'))
            and os.getpgrp() == int(os.environ.get('BABYLM_DW_QUEUE_PGID', '-1')), 'Worker must belong to the bounded queue')
    require(sha(master_path) == master_sha, 'Worker master changed')
    master = load(master_path); out, bound = validate(master)
    action, arm = stage.split('_'); arm_dir = out / arm
    receipt_path = arm_dir / ('train-worker.json' if action == 'train' else 'eval-worker.json')
    require(not receipt_path.exists(), 'No worker retry/overwrite')
    receipt = {'status': 'running', 'stage': stage, 'arm': arm, 'condition': CONDITIONS[arm], 'started_utc': utc(),
               'master_protocol_sha256': master_sha, 'automatic_retry': False}
    write(receipt_path, receipt)
    def stop(signum, frame):
        raise InterruptedError('Bounded D/W worker terminated')
    signal.signal(signal.SIGTERM, stop)
    try:
        if action == 'train':
            from src.babylm_hybrid import training_timebound_v0 as engine
            common.configure_runtime()
            runtime = engine._runtime_signature('cuda')
            expected = bound['eval_template']['expected_runtime']
            require(all(runtime[k] == expected[k] for k in runtime), 'Training runtime differs from fixed dev runtime')
            protocol = bound['training_protocol_' + arm]
            run = arm_dir / 'train' / 'run'; require(not run.exists(), 'Fresh training only')
            context = (dense_factory(engine, master['expected_initial_parameter_hashes_sha256'], master['source_sha256']) if arm == 'D'
                       else common.injected_factory(engine, master['expected_initial_parameter_hashes_sha256'], master['source_sha256']))
            with context:
                engine.train_run(protocol, 'dense', run)
            audit = audit_train(run, protocol, master, master_sha, arm)
            write(arm_dir / 'train' / 'audit.json', audit)
            receipt.update(status='complete', counts=audit['counts'], eval_counts=audit['eval_counts'],
                           initial_parameter_hashes_sha256=audit['initial_parameter_hashes_sha256'])
        else:
            path = arm_dir / 'bound-eval.json'; protocol = load(path)
            require(protocol == evaluation_protocol(master, out, bound, arm), 'Bound evaluation differs from prescribed template/final checkpoint')
            audit_path = arm_dir / 'train' / 'audit.json'
            require(sha(audit_path) == protocol['training_audit_sha256'], 'Training audit changed before scoring')
            require(protocol['checkpoint_sha256'] == load(audit_path)['final_model_checkpoint_sha256'], 'Wrong eval checkpoint')
            from scripts import run_babylm_checkpoint_eval_v0 as evaluator, run_babylm_local_eval_v0 as local_eval
            context = local_eval.injected_factory(evaluator) if arm == 'W' else nullcontext()
            with context:
                evaluator.run_evaluation(protocol)
            audit = audit_eval(arm_dir / 'eval', protocol, arm)
            write(arm_dir / 'eval-audit.json', audit)
            receipt.update(status='complete', counts=audit['counts'], total=audit['total'])
    except BaseException as exc:
        receipt.update(status='failed_or_incomplete', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        folder = arm_dir / ('train/run' if action == 'train' else 'eval')
        if (folder / 'summary.json').exists():
            actual = load(folder / 'summary.json')
            receipt.update(counts=actual.get('counts'), eval_counts=actual.get('eval_counts'), engine_status=actual.get('status'))
        receipt['finished_utc'] = utc(); write(receipt_path, receipt)
    return 0 if receipt['status'] == 'complete' else 2


def execute(master_path, master_sha):
    import fcntl
    require(os.name == 'posix' and os.getppid() == os.getpgrp(), 'Dedicated timeout parent/process group required')
    require(sha(master_path) == master_sha, 'Master changed')
    master = load(master_path); out, bound = validate(master)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'], text=True, timeout=15).strip()
    hw = master['hardware']; require(gpu == hw['physical_uuid'] + ', ' + hw['driver_version'], 'Frozen GPU/driver differs')
    lock = open('/tmp/babylm-one-epoch-' + hw['physical_uuid'] + '.lock', 'a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not out.exists(), 'No retry/overwrite of confirmation batch')
    out.mkdir(parents=True)
    for arm in CONDITIONS:
        (out / arm / 'train').mkdir(parents=True)
    write(out / 'master.json', master)
    started = time.monotonic(); abnormal = False
    state = {'status': 'running', 'started_utc': utc(), 'pid': os.getpid(), 'pgid': os.getpgrp(),
             'master_protocol_sha256': master_sha, 'stages': [], 'automatic_retry': False,
             'automatic_monitoring': 'PAUSED', 'hourly_rate_usd': None,
             'cost_mode': 'time_bounded_unknown_rate', 'hard_timeout_seconds': 58200}
    def stop(signum, frame):
        raise InterruptedError('Bounded D/W queue terminated')
    signal.signal(signal.SIGTERM, stop)
    try:
        for stage in STAGES:
            validate(master)
            action, arm = stage.split('_')
            if action == 'eval':
                bind_evaluation(master, out, bound, arm)
            limit = master['train_timeout_seconds' if action == 'train' else 'eval_timeout_seconds']
            remaining = min(limit, master['hard_timeout_seconds'] - 60 - (time.monotonic() - started))
            require(remaining >= limit - 1, 'Insufficient full stage allocation; no partial stage start')
            row = {'stage': stage, 'status': 'starting', 'started_utc': utc()}; state['stages'].append(row)
            env = os.environ.copy(); env.update(BABYLM_DW_QUEUE_PID=str(os.getpid()), BABYLM_DW_QUEUE_PGID=str(os.getpgrp()))
            command = [sys.executable, '-u', str(resolve(SELF)), '--protocol', str(master_path),
                       '--protocol-sha256', master_sha, '--execute', '--worker', stage]
            with (out / (stage + '.stdout.log')).open('xb') as stdout, (out / (stage + '.stderr.log')).open('xb') as stderr:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stdout, stderr=stderr, start_new_session=False)
                require(os.getpgid(child.pid) == os.getpgrp(), 'Worker escaped queue group')
                row.update(status='running', pid=child.pid, pgid=os.getpgrp()); state['active_stage'] = stage
                write(out / 'stage.json', state)
                code = child.wait(timeout=remaining)
            row.update(returncode=code, finished_utc=utc())
            require(code == 0, 'Worker failed; stop all later stages: ' + stage)
            receipt = load(out / arm / ('train-worker.json' if action == 'train' else 'eval-worker.json'))
            require(receipt['status'] == 'complete', 'Worker audit incomplete')
            row['status'] = 'complete'; state['active_stage'] = None; write(out / 'stage.json', state)
        da = load(out / 'D' / 'train' / 'audit.json'); wa = load(out / 'W' / 'train' / 'audit.json')
        require(da['initial_parameter_hashes_sha256'] == wa['initial_parameter_hashes_sha256'] == master['expected_initial_parameter_hashes_sha256'], 'D/W initialization differs')
        state.update(status='complete_pending_independent_local_audit', shared_initialization_verified=True,
                     scientific_training_forward_calls=45196, scientific_training_backward_calls=45196,
                     scientific_updates=2826, training_panel_forward_calls=672, full_dev_forward_calls=37584,
                     claim='Additional matched seed only; no automatic expansion, significance, equivalence or speed claim.')
        return 0
    except BaseException as exc:
        abnormal = True
        state.update(status='failed_or_incomplete', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        return 2
    finally:
        state.update(finished_utc=utc(), elapsed_wall_seconds=time.monotonic() - started)
        try:
            state['recorded_worker_receipts'] = {}
            for arm in CONDITIONS:
                for name in ('train-worker.json', 'eval-worker.json'):
                    rp = out / arm / name
                    try:
                        if rp.exists(): state['recorded_worker_receipts'][arm + '/' + name] = load(rp)
                    except (OSError, ValueError) as exc:
                        state['recorded_worker_receipts'][arm + '/' + name] = {'read_error': str(exc)}
            write(out / 'stage.json', state)
        finally:
            if abnormal:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.killpg(os.getpgrp(), signal.SIGTERM)
                time.sleep(5)
                os.killpg(os.getpgrp(), signal.SIGKILL)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True); parser.add_argument('--protocol-sha256')
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--worker', choices=STAGES)
    args = parser.parse_args(); path = resolve(args.protocol)
    if args.protocol_sha256:
        require(sha(path) == args.protocol_sha256, 'Frozen master SHA differs')
    if not args.execute:
        require(args.worker is None, 'Worker mode requires execute')
        validate(load(path)); print(json.dumps({'status': 'validated_no_model_calls', 'model_calls': 0})); return 0
    require(bool(args.protocol_sha256), 'Execute requires frozen master digest')
    return worker(path, args.protocol_sha256, args.worker) if args.worker else execute(path, args.protocol_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
