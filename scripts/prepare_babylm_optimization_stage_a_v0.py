"""Freeze the explicitly authorized, finite Stage A from existing evidence."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    tag = 'babylm-optimization-stage-a-20260920-v0'
    directory = ROOT / 'configs' / tag
    directory.mkdir(exist_ok=False)
    def write(name, value):
        target = directory / name
        with target.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        return target.relative_to(ROOT).as_posix(), sha(target)
    old = json.loads((ROOT / 'configs/babylm-one-epoch-blackwell-mig-20260919-v3.json').read_text())
    receipt = json.loads((ROOT / 'logs/manual-oneepoch-status-20260920T025811Z/final-model-backup-receipt.json').read_text())
    raw = ROOT / 'logs/manual-oneepoch-status-20260920T025811Z/raw'
    expected = {}
    for mode in ('dense', 'sparse'):
        lines = (raw / f'results/blackwell-mig-one-epoch-pair-20260919-v3/{mode}/run/events.jsonl').read_text().splitlines()
        events = [json.loads(line) for line in lines]
        final = [v for v in events if v['type'] == 'evaluation' and v['counts']['updates'] == 1413]
        assert len(final) == 1
        expected[mode] = write(mode + '-replay-reference.json', final[0]['metrics'])
    sources = dict(old['source_sha256'])
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, 'Frozen scientific source changed: ' + name
    for name in ['scripts/run_babylm_checkpoint_eval_v0.py', 'src/babylm_hybrid/routing_diagnostics.py',
                 'scripts/run_babylm_optimization_stage_a_v0.py', 'scripts/prepare_babylm_optimization_stage_a_v0.py']:
        sources[name] = sha(ROOT / name)
    jobs = []
    def add(name, role, mode, step, policy='learned', instrument=False):
        item = next(i for i in receipt['items'] if f'/{mode}/run/snapshots/model-u{step:08d}-' in i['path'])
        meta = json.loads((raw / Path(item['path']).with_suffix('.json')).read_text())
        p = {'schema_version': 1, 'scope': 'scientific_evaluation', 'mode': mode,
             'model_config': old['model_config'], 'backbone_seed': old['backbone_seed'],
             'indexer_seed': old['indexer_seed'], 'checkpoint_path': item['path'],
             'checkpoint_sha256': item['sha256'], 'checkpoint_format': 'model_only',
             'checkpoint_protocol_sha256': meta['protocol_sha256'],
             'dev_manifest': old['eval_manifest'], 'dev_manifest_sha256': old['eval_manifest_sha256'],
             'output_dir': f'results/{tag}/{name}', 'device': 'cuda', 'dtype': 'float32',
             'torch_num_threads': 1, 'max_wall_seconds': 5400 if role == 'full_dev' else 600,
             'window_indices': None if role == 'full_dev' else old['eval_window_indices'],
             'position_diagnostics': True, 'enable_routing_diagnostics': instrument,
             'routing_policy': policy, 'routing_seed': 20260920, 'expected_source_hashes': sources,
             'transformers_gdn_source_sha256': 'ce8dd330895e7b589a773e3aa1574f3f6b1749ad052d0b55efc94a0dc8502dd4',
             'expected_runtime': {'torch': '2.10.0+cu128', 'numpy': '2.4.2', 'cuda_version': '12.8',
                 'torch_num_threads': 1, 'dtype': 'float32', 'cuda_matmul_allow_tf32': False,
                 'cudnn_allow_tf32': False, 'deterministic_algorithms': True},
             'notes': ['No training, no checkpoint mutation, no sparse-speed inference.',
                       'Known panel is diagnostic; full dev is development evidence, not unseen confirmation.']}
        path, digest = write(name + '.json', p)
        job = {'name': name, 'role': role, 'protocol': path, 'protocol_sha256': digest}
        if role == 'replay' or (instrument and policy == 'learned' and step == 1413):
            job.update(replay_reference=expected[mode][0], replay_reference_sha256=expected[mode][1])
        jobs.append(job)
    for mode in ('dense', 'sparse'):
        add('replay-' + mode, 'replay', mode, 1413)
    # Fixed diagnostics precede the long full-dev pass, but no tuning decision
    # is taken until the complete Stage A evidence has been reviewed.
    for step in (137, 693, 1413):
        for policy in ('learned', 'prefix', 'local', 'random'):
            add(f'routing-u{step:04d}-{policy}', 'diagnostic', 'sparse', step, policy, True)
    for mode in ('dense', 'sparse'):
        add('full-dev-' + mode, 'full_dev', mode, 1413)
    p = {'schema_version': 1, 'launch_allowed': True, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
         'scope': 'finite_existing_checkpoint_evaluation_and_routing_intervention_only',
         'authorization': 'logs/babylm-optimization-authorization-current.json', 'cycle_cap_usd': 20,
         'output_dir': f'results/{tag}', 'jobs': jobs, 'execution_hardware': old['execution_hardware'],
         'source_sha256': sources, 'max_wall_seconds': 14280, 'hard_timeout_seconds': 14400,
         'hourly_rate_usd': 0.81, 'actual_gpu_quote_usd_per_hour': 0.59, 'stage_cost_cap_usd': 3.24,
         'replay_tolerance': {'aggregate_nll_abs': 1e-6, 'window_nll_abs': 1e-5},
         'expected_lm_forward_calls_if_all_complete': 38256, 'expected_backward_calls': 0,
         'expected_optimizer_updates': 0, 'recurring_automation_resumed': False,
         'no_training_dispatch': True, 'no_automatic_retry_or_resume': True,
         'diagnostic_attention_calls_are_separately_counted': True,
         'notes': ['Two exact checkpoint panel replays must pass before any new evaluation.',
                   'Four-hour hard process-group timeout includes termination reserve.',
                   'MIG isolation reuses existing hardware gate; no repeated tiny numerics training.',
                   'All 16 jobs are frozen before observing any result; no adaptive selection within this queue.']}
    master, digest = write('master.json', p)
    transfer_paths = [name for name in sources if name in (
        'scripts/run_babylm_checkpoint_eval_v0.py', 'src/babylm_hybrid/routing_diagnostics.py',
        'scripts/run_babylm_optimization_stage_a_v0.py', 'scripts/prepare_babylm_optimization_stage_a_v0.py')]
    transfer_paths += [path.relative_to(ROOT).as_posix() for path in directory.glob('*.json')]
    manifest = {'master_protocol': master, 'master_sha256': digest,
                'files': {name: sha(ROOT / name) for name in transfer_paths}}
    dest = ROOT / 'logs' / (tag + '-deployment.json')
    with dest.open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    print(json.dumps({'master': master, 'sha256': digest, 'jobs': len(jobs), 'files': len(transfer_paths)}))


if __name__ == '__main__':
    main()
