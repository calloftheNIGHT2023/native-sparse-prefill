"""Local metadata-only audit of the completed second-seed D/W execution.

No model imports, remote operations, or mutations of captured evidence.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = 'logs/babylm-dw-final-text-backup-20260921/20260921T211814.452452Z/manifest.json'
PROBE = 'logs/babylm-dw-seed-small-probes-20260921/20260921T211631.079289Z.json'
LAUNCH = 'logs/babylm-dw-seed-launch-20260921-v1.json'
MASTER = 'configs/babylm-dw-seed-confirmation-20260921-v1/master.json'
OLD_MANIFEST = 'logs/babylm-dw-v0-interruption-evidence-20260921/manifest.json'
OLD_TERMINAL = 'logs/babylm-dw-seed-manual-snapshots-20260921/20260921T073059.863835Z/receipt.json'
PREFIX = 'results/babylm-dw-seed-confirmation-20260921-v1/'
checks = []


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with (ROOT / path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check(name, ok):
    checks.append({'check': name, 'passed': bool(ok)})
    if not ok:
        raise ValueError('Audit failed: ' + name)


def dt(value):
    return datetime.fromisoformat(value)


def main():
    manifest, probe, launch, master = map(read, [MANIFEST, PROBE, LAUNCH, MASTER])
    check('immutable terminal manifest hash', sha(MANIFEST) == 'ce20b32bbf460cba299e12f67fe16dd04e58ed0986b54c07803401decd08d103')
    check('terminal text collection complete', manifest['status'] == 'complete_all_terminal_text_sha_verified')
    receipts = {x['remote_relative_path']: x for x in manifest['receipts']}
    check('unique terminal artifact paths', len(receipts) == len(manifest['receipts']))
    for item in manifest['receipts']:
        check('terminal SHA/size: ' + item['remote_relative_path'],
              sha(item['local_path']) == item['sha256'] and (ROOT / item['local_path']).stat().st_size == item['size_bytes'])

    def artifact(rel):
        return read(receipts[PREFIX + rel]['local_path'])

    def artifact_sha(rel):
        return receipts[PREFIX + rel]['sha256']

    stage = artifact('stage.json')
    check('terminal probe exactly matches immutable final stage', stage == probe['stage'])
    check('no registered project processes observed', probe['registered_processes'] == [])
    check('project flock observed free', probe['project_lock_free'] is True)
    check('read-only probe made no model calls', probe['new_model_calls'] == 0)
    master_sha = sha(MASTER)
    check('launch, stage and master file SHA binding', launch['protocol_sha256'] == stage['master_protocol_sha256'] == master_sha)
    remote_launch = read(receipts['logs/dw-seed-confirmation-launch-20260921-v1/launch.json']['local_path'])
    check('immutable remote launch equals local launch', remote_launch == launch)
    check('controller source pin', launch['source_sha256'] == master['source_sha256']['scripts/run_babylm_dw_seed_confirmation_v1.py'])
    for path, expected in master['source_sha256'].items():
        check('source SHA: ' + path, sha(path) == expected)
    for field in ['reference_dense_protocol', 'initialization_receipt', 'authorization', 'training_protocol_D', 'training_protocol_W', 'eval_template']:
        check('master binding: ' + field, sha(master[field]) == master[field + '_sha256'])
    check('bounded unknown-price authorization', master['actual_compute_hourly_rate_usd'] is None and master['rate_unknown'] is True and stage['hourly_rate_usd'] is None)
    check('no retry or automatic follow-up', stage['automatic_retry'] is False and stage['automatic_monitoring'] == 'PAUSED' and master['no_automatic_retry'] is True)
    check('exact queue/stage caps', master['hard_timeout_seconds'] == launch['queue_hard_seconds'] == stage['hard_timeout_seconds'] == 58200 and master['train_timeout_seconds'] == 21600 and master['eval_timeout_seconds'] == 7200)
    check('launch hard deadline bound', (dt(launch['hard_deadline_utc']) - dt(launch['started_utc'])).total_seconds() == 58200)
    check('complete terminal state, no next stage', stage['status'] == 'complete_pending_independent_local_audit' and stage['active_stage'] is None)
    check('exact four-stage sequence', [x['stage'] for x in stage['stages']] == ['train_D', 'train_W', 'eval_D', 'eval_W'])
    timeline = []
    previous = dt(stage['started_utc'])
    for item in stage['stages']:
        start, end = dt(item['started_utc']), dt(item['finished_utc'])
        seconds = (end - start).total_seconds()
        cap = master['train_timeout_seconds'] if item['stage'].startswith('train') else master['eval_timeout_seconds']
        check(item['stage'] + ' return0/identity/order/bound', item['status'] == 'complete' and item['returncode'] == 0 and item['pgid'] == launch['pgid'] == stage['pgid'] and previous <= start < end and seconds < cap)
        previous = end
        timeline.append({**item, 'wall_seconds': seconds, 'hard_limit_seconds': cap})
    wall = stage['elapsed_wall_seconds']
    check('UTC and monotonic elapsed agree', abs(wall - (dt(stage['finished_utc']) - dt(stage['started_utc'])).total_seconds()) < 0.01)
    check('queue ended before deadline', previous <= dt(stage['finished_utc']) < dt(launch['hard_deadline_utc']) and wall < 58200)
    arms = {}
    for arm in ['D', 'W']:
        train = artifact(arm + '/train/audit.json')
        summary = artifact(arm + '/train/run/summary.json')
        evaluation = artifact(arm + '/eval/summary.json')
        eaudit = artifact(arm + '/eval-audit.json')
        bound = artifact(arm + '/bound-eval.json')
        binding = artifact(arm + '/checkpoint-binding.json')
        check(arm + ' completed training audit and epoch', train['status'] == 'complete_seed_epoch_audited' and summary['status'] == 'epoch_complete' and summary['cursor'] == {'epoch': 1, 'position': 0})
        check(arm + ' train audit metadata hashes', train['events_sha256'] == artifact_sha(arm + '/train/run/events.jsonl') and train['summary_sha256'] == artifact_sha(arm + '/train/run/summary.json'))
        check(arm + ' common initialization and seed', train['initial_parameter_hashes_sha256'] == master['expected_initial_parameter_hashes_sha256'] == canonical(summary['initial_parameter_hashes']) and train['backbone_seed'] == master['backbone_seed'] == 20260921)
        check(arm + ' train protocol/source bindings', train['master_protocol_sha256'] == master_sha and train['source_sha256'] == master['source_sha256'] and train['protocol_sha256'] == summary['protocol_sha256'] == canonical(artifact(arm + '/train/run/protocol.json')))
        check(arm + ' train counts agreement', train['counts'] == summary['counts'] and train['eval_counts'] == summary['eval_counts'])
        count = train['counts']
        check(arm + ' exact training workload', all(count[k] == v for k, v in {'forward_attempts': 22598, 'forward_calls': 22598, 'backward_attempts': 22598, 'backward_calls': 22598, 'scientific_updates': 1413, 'updates': 1413, 'windows': 22598, 'input_tokens': 16325414, 'loss_tokens': 16302816, 'word_exposures': 10001709, 'engineering_updates': 0}.items()))
        check(arm + ' exact panel workload', train['eval_counts']['evaluations'] == 7 and train['eval_counts']['forward_calls'] == 336)
        check(arm + ' final checkpoint binding', binding['master_protocol_sha256'] == canonical(master) and binding['template_file_sha256'] == master['eval_template_sha256'] and binding['training_audit_sha256'] == artifact_sha(arm + '/train/audit.json') and binding['checkpoint_sha256'] == bound['checkpoint_sha256'] == train['final_model_checkpoint_sha256'] == eaudit['checkpoint_sha256'] and bound['checkpoint_protocol_sha256'] == train['protocol_sha256'] and 'u00001413' in bound['checkpoint_path'])
        check(arm + ' evaluation completion and file binding', eaudit['status'] == 'complete_seed_full_dev_audited' and evaluation['status'] == 'evaluation_complete' and eaudit['summary_sha256'] == artifact_sha(arm + '/eval/summary.json') and eaudit['windows_sha256'] == artifact_sha(arm + '/eval/windows.jsonl') and evaluation['protocol_sha256'] == eaudit['protocol_sha256'] == canonical(bound))
        check(arm + ' evaluation exact workload', eaudit['counts'] == evaluation['counts'] and all(evaluation['counts'][k] == v for k, v in {'forward_attempts': 18792, 'forward_calls': 18792, 'committed_windows': 18792, 'submitted_input_tokens': 17437534, 'submitted_word_exposures': 10420962, 'backward_calls': 0, 'optimizer_updates': 0}.items()) and evaluation['total']['loss_tokens'] == 17418742)
        check(arm + ' actual runtime and hardware identity', evaluation['runtime'] == bound['expected_runtime'] and bound['execution_hardware'] == master['hardware'] and evaluation['runtime']['gpu']['uuid'] == master['hardware']['physical_uuid'].removeprefix('GPU-'))
        check(arm + ' no fictitious priced training estimate', summary['estimated_gpu_usd'] is None and summary['cost_mode'] == 'time_bounded_unknown_rate')
        for kind, source in [('train', train), ('eval', eaudit)]:
            rel = arm + '/' + kind + '-worker.json'
            worker = artifact(rel)
            interval = next(x for x in stage['stages'] if x['stage'] == kind + '_' + arm)
            check(arm + ' ' + kind + ' immutable worker receipt', worker == stage['recorded_worker_receipts'][rel] and worker['status'] == 'complete' and worker['master_protocol_sha256'] == master_sha and worker['counts'] == source['counts'])
            check(arm + ' ' + kind + ' worker within controller stage', dt(interval['started_utc']) <= dt(worker['started_utc']) < dt(worker['finished_utc']) <= dt(interval['finished_utc']))
        arms[arm] = {'condition': train['condition'], 'training_counts': count, 'panel_counts': train['eval_counts'], 'full_dev_counts': evaluation['counts'], 'final_model_checkpoint_sha256': train['final_model_checkpoint_sha256'], 'full_checkpoint_sha256': summary['checkpoint_sha256'], 'initial_parameter_hashes_sha256': train['initial_parameter_hashes_sha256'], 'training_engine_elapsed_seconds': summary['elapsed_wall_seconds'], 'full_dev_scorer_elapsed_seconds': evaluation['elapsed_wall_seconds'], 'runtime': evaluation['runtime'], 'estimated_gpu_usd': None}
    counts = {'training_forward_calls': 45196, 'training_backward_calls': 45196, 'training_updates': 2826, 'training_panel_forward_calls': 672, 'full_dev_forward_calls': 37584, 'total_model_forward_calls': 83452, 'total_model_backward_calls': 45196}
    check('summed successful scientific workload', sum(x['training_counts']['forward_calls'] for x in arms.values()) == counts['training_forward_calls'] == stage['scientific_training_forward_calls'] and sum(x['training_counts']['backward_calls'] for x in arms.values()) == counts['training_backward_calls'] == stage['scientific_training_backward_calls'] and sum(x['training_counts']['updates'] for x in arms.values()) == counts['training_updates'] == stage['scientific_updates'] and sum(x['panel_counts']['forward_calls'] for x in arms.values()) == counts['training_panel_forward_calls'] == stage['training_panel_forward_calls'] and sum(x['full_dev_counts']['forward_calls'] for x in arms.values()) == counts['full_dev_forward_calls'] == stage['full_dev_forward_calls'] and 45196 + 672 + 37584 == counts['total_model_forward_calls'])
    check('prespecified successful total matched', master['scientific_expected'] == launch['expected_scientific_counts'] == {'forward_calls': 83452, 'backward_calls': 45196, 'updates': 2826})

    old_manifest, old_terminal = read(OLD_MANIFEST), read(OLD_TERMINAL)
    for item in old_manifest['files']:
        if item.get('local_path'):
            check('old attempt retained artifact: ' + item['path'], sha(item['local_path']) == item['sha256'] and (ROOT / item['local_path']).stat().st_size == item['bytes'])
    old_events_path = next(x['local_path'] for x in old_manifest['files'] if x['path'].endswith('events.jsonl'))
    old_failure_path = next(x['local_path'] for x in old_manifest['files'] if '/failures/' in x['path'] and x['path'].endswith('.json'))
    old_events = [json.loads(line) for line in (ROOT / old_events_path).read_text().splitlines()]
    failure = read(old_failure_path)
    old_count = failure['counts']
    check('old physical partial work preserved', old_events[-1]['type'] == 'failure' and old_events[-1]['counts'] == old_count and all(old_count[k] == v for k, v in {'forward_calls': 317, 'forward_attempts': 317, 'backward_calls': 316, 'backward_attempts': 317, 'updates': 19, 'scientific_updates': 19}.items()))
    check('old committed19 plus partial batch', len([x for x in old_events if x['type'] == 'update']) == 19 and sum(x.get('forward_completed', False) for x in failure['pending_batch']) == 13 and sum(x.get('backward_completed', False) for x in failure['pending_batch']) == 12)
    old_panels = [x for x in old_events if x['type'] == 'evaluation']
    check('old initial panel48 preserved', len(old_panels) == 1 and old_panels[0]['eval_counts']['forward_calls'] == 48)
    check('old failure not a resume point', failure['resume_supported'] is False and master['resume_or_checkpoint_reuse'] is False)
    old_stage = old_terminal['stage']
    check('old interrupted attempt terminated before fresh v1', old_stage['status'] == 'failed_or_incomplete' and old_stage['stages'][0]['returncode'] == 2 and old_terminal['processes'] == [] and old_terminal['project_lock_free'] is True and dt(old_stage['finished_utc']) < dt(launch['started_utc']))
    old = {'status': old_stage['status'], 'interruption': 'Administrative duration correction; preserve partial work, no checkpoint reuse.', 'counts': old_count, 'panel_forward_calls': 48, 'total_model_forward_calls': 365, 'controller_elapsed_wall_seconds': old_stage['elapsed_wall_seconds'], 'engine_elapsed_wall_seconds': failure['elapsed_wall_seconds'], 'overlapping_times_must_not_be_added': True, 'failed_backward_attempts': 1, 'failure_snapshot_resume_supported': False, 'local_tensor_backup_complete_not_asserted': True}
    report = {'status': 'passed_terminal_execution_cost_audit', 'audited_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Local immutable metadata, execution timing, cost limits, source identity, physical-work accounting; no model construction or checkpoint tensor loading; quality audit separate.', 'inputs': {p: sha(p) for p in [MANIFEST, PROBE, LAUNCH, MASTER, OLD_MANIFEST, OLD_TERMINAL]}, 'checks_passed': len(checks), 'checks_failed': 0, 'checks': checks, 'terminal_artifacts_independently_sha_verified': len(receipts), 'master_file_sha256': master_sha, 'master_canonical_sha256': canonical(master), 'canonical_hash_note': 'checkpoint-binding records canonical master SHA; launch and workers record raw file SHA. Both independently matched.', 'timeline': timeline, 'successful_v1': {'started_utc': stage['started_utc'], 'finished_utc': stage['finished_utc'], 'controller_elapsed_wall_seconds': wall, 'controller_elapsed_hours': wall / 3600, 'hard_limit_seconds': 58200, 'counts': counts, 'arms': arms}, 'old_v0_separate_waste': old, 'combined_physical_work_v0_plus_v1_only': {'model_forward_calls': 83817, 'backward_calls': 45512, 'backward_attempts': 45513, 'committed_updates': 2845, 'not_successful_pair_scientific_counts': True, 'prior_engineering_and_previous_seed_excluded': True}, 'cost': {'actual_hourly_rate_usd': None, 'estimated_total_usd': None, 'controller_time_sum_v0_plus_v1_seconds': wall + old_stage['elapsed_wall_seconds'], 'controller_time_sum_v0_plus_v1_hours': (wall + old_stage['elapsed_wall_seconds']) / 3600, 'billing_basis_unknown': True, 'idle_setup_transfer_storage_not_in_controller_time': True, 'no_sparse_speed_or_savings_claim': True, 'shared_gpu_no_whole_host_idle_claim': True}, 'terminal_observation': {'observed_utc': probe['utc'], 'registered_research_processes': [], 'project_lock_free': True, 'seconds_since_controller_completion': (dt(probe['utc']) - dt(stage['finished_utc'])).total_seconds(), 'boundary': 'Endpoint observation of registered project work only; no claim of continuous inactivity, whole GPU availability, power-off, or billing termination.'}, 'evidence_boundary': {'all_terminal_text_local': True, 'all_checkpoint_weights_local': False, 'statement': 'This audit verifies checkpoint identity from SHA-bound terminal receipts. Complete tensor backup and tensor-state audit are not claimed.'}, 'audit_work': {'model_forward_calls': 0, 'model_backward_calls': 0, 'optimizer_updates': 0, 'cuda_calls': 0, 'remote_calls': 0}}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'execution-audit.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    rows = '\n'.join(f"| {x['stage']} | {x['started_utc'][11:19]}–{x['finished_utc'][11:19]} | {x['wall_seconds'] / 3600:.4f} | {x['hard_limit_seconds'] / 3600:g} | 0 |" for x in timeline)
    note = f'''# D/W 第二种子执行与费用边界\n\n四个阶段均正常退出，正式队列于 2026-09-21 18:59:08 UTC 完成，共 {wall:.3f} 秒（{wall / 3600:.4f} 小时），低于 16 小时 10 分钟的固定上限。\n\n| 阶段 | UTC 起止 | 实际小时 | 上限小时 | 退出码 |\n|---|---|---:|---:|---:|\n{rows}\n\n正式批次：45,196 次训练前向、45,196 次反向、2,826 次更新；另有 672 次过程面板前向、37,584 次完整开发集前向，共 83,452 次模型前向。\n\n旧 v0 因时限估计偏短而中断，另耗 317 次训练前向、316 次完成反向、19 次完成更新、48 次面板前向；还有 1 次开始但未完成的反向。这些不纳入成功配对的科学计数，但保留在实际消耗中。旧控制器耗时 {old_stage['elapsed_wall_seconds']:.3f} 秒；不能再叠加其内部引擎时间。两次控制器时间合计 {(wall + old_stage['elapsed_wall_seconds']) / 3600:.4f} 小时，未计部署、传输、间隔和资源保留时间。\n\n租价和计费口径未知，美元费用保持 null，不能据此宣称成本下降或稀疏加速。21:16:30 UTC 的只读快照确认登记的研究进程均已退出、项目锁可用；这不等于共享 GPU 整体空闲，也不意味着计费已停止。\n\n45 份终态文本逐 SHA 复核通过；本报告只验证检查点来源绑定，不声称全部权重已备份到本机。独立质量统计见同目录其他审计；本次审计 0 模型调用、0 远端调用。\n'''
    with (OUT / 'execution-cost-note.md').open('x', encoding='utf-8') as f:
        f.write(note)
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(OUT / 'execution-audit.json'), 'stage_seconds': {x['stage']: x['wall_seconds'] for x in timeline}, 'controller_hours': wall / 3600}, ensure_ascii=False))


if __name__ == '__main__':
    main()
