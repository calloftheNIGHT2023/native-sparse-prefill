"""Publish local evidence only after complete independent audit and terminal probe."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()

def fresh(path, data):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def stamp():
    return datetime.now(timezone.utc).isoformat()

audit = read(HERE/'metrics-audit.json')
assert audit['status'] == 'passed_complete_two_segment_W_union_audit'
assert audit['complete_quality_evidence'] and not audit['failures']
assert audit['physical_evaluation_counts']['union_forward_calls'] == 18792
probe_path = ROOT/'logs/babylm-w-remainder-status-current.json'
probe = read(probe_path)
assert probe['remaining_windows'] == 0 and probe['new_committed_windows'] == 6390
assert probe['cumulative_committed_windows'] == 18792 and probe['window_order_exact']
assert probe['finite_loss'] and probe['lock_free'] and not probe['processes']
assert probe['remainder-receipt.json']['status'] == 'complete_local_remainder_and_union_audited'
terminal_immutable = HERE/'final-terminal-probe.json'
assert not terminal_immutable.exists()
terminal_immutable.write_bytes(probe_path.read_bytes())

backup = ROOT/'logs/babylm-w-arm-remainder-backup-20260921-v0'
pointer = read(backup/'manifest.json')
manifest_path = backup/pointer['latest_manifest']
assert sha(manifest_path) == pointer['latest_manifest_sha256'] == audit['requested_manifest_sha256']
manifest = read(manifest_path)
assert manifest['status'] == 'collection_complete_all_files_sha_verified'
files = {}
for receipt in manifest['receipts']:
    path = backup/receipt['local_relative_path']
    assert path.stat().st_size == receipt['size_bytes'] and sha(path) == receipt['sha256']
    files[receipt['relative_path']] = dict(path=relative(path), sha256=sha(path), bytes=path.stat().st_size)

prefix_execution_path = ROOT/'results/babylm-stage-w-partial-20260921/execution-audit.json'
prefix_execution = read(prefix_execution_path)
for key, entry in prefix_execution['files'].items():
    path = ROOT/entry['path']
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    assert key not in files
    files[key] = entry
receipt = audit['remainder_receipt']
launch = read(ROOT/'logs/babylm-w-remainder-launch-20260921.json')
completed = datetime.fromisoformat(receipt['completed_utc'])
started = datetime.fromisoformat(launch['started_utc'])
assert completed < datetime.fromisoformat(launch['hard_deadline_utc'])
prefix_completed = datetime.fromisoformat(prefix_execution['failure']['completed_utc'])
execution = dict(
    status='complete_W_two_segment_evaluation_evidence_verified', utc=stamp(),
    previous_timeout_record_preserved=dict(path=relative(prefix_execution_path),sha256=sha(prefix_execution_path)),
    remainder_manifest=dict(path=relative(manifest_path),sha256=sha(manifest_path)),
    unique_files=len(files), unique_bytes=sum(v['bytes'] for v in files.values()), files=files,
    registered_project_processes_exited=True, project_gpu_lock_free=True,
    other_shared_gpu_jobs_not_inspected_or_modified=True, pod_power_action=False,
    automatic_monitoring='PAUSED', new_training_updates=0, new_backward_calls=0,
    evaluation_counts=audit['physical_evaluation_counts'], full_dev_complete=True,
    no_duplicate_or_missing_windows=True, terminal_status_receipt=relative(terminal_immutable),
    terminal_status_receipt_sha256=sha(terminal_immutable),
    timing_segments=audit['timing_segments'], remainder_wrapper_seconds=(completed-started).total_seconds(),
    original_prefix_wrapper_seconds=prefix_execution['failure']['wrapper_elapsed_seconds'],
    gap_between_original_completion_and_authorized_remainder_seconds=(started-prefix_completed).total_seconds(),
    cost=dict(hourly_rate=None,currency=None,invoice_or_total_spend_verified=False,
              old_GPU_rate_not_substituted=True,original_timeout_work_included=True,
              summed_scorer_active_seconds=audit['timing_segments']['summed_scorer_active_wall_seconds'],
              setup_migration_replay_and_between_segment_time_not_in_scorer_sum=True,
              not_a_whole_GPU_idle_or_exclusive_timing_claim=True),
    independent_metrics_audit=dict(path=relative(HERE/'metrics-audit.json'),sha256=sha(HERE/'metrics-audit.json')),
    no_model_calls_in_local_audits=True, no_new_experiment_queued=True)
fresh(HERE/'execution-audit.json', execution)
subprocess.run([sys.executable,str(HERE/'write_report.py')],check=True,cwd=ROOT)
index_paths=[HERE/name for name in ['REPORT.md','metrics-audit.json','execution-audit.json','audit_union.py','write_report.py','finalize_evidence.py','final-terminal-probe.json']]
index_paths += [ROOT/'configs/babylm-w-arm-remainder-20260921-v0.json',
                ROOT/'scripts/run_babylm_local_eval_remainder_v0.py',
                ROOT/'logs/babylm-w-remainder-authorization-20260921.json',
                ROOT/'docs/babylm-w-remainder-protocol-2026-09-21.md']
fresh(HERE/'evidence-index.json',dict(status='complete_evidence_index',utc=stamp(),
    files=[dict(path=relative(p),sha256=sha(p),bytes=p.stat().st_size) for p in index_paths],
    raw_evidence_manifest='execution-audit.json', no_scientific_source_modified=True))

metrics={k:v['total'] for k,v in audit['full_dev'].items()}
screen=metrics['W']['nll']-metrics['F']['nll'] >= math.log(1.01)
now=stamp()
completion=dict(status='complete_independently_audited', completed_utc=receipt['completed_utc'],
    audit_completed_utc=audit['completed_utc'], full_dev_completed=True,
    old_forward_calls_preserved=12402,new_forward_calls=6390,union_forward_calls=18792,
    input_tokens=17437534,loss_tokens=17418742,backward_calls=0,new_training_updates=0,
    immutable_manifest=relative(manifest_path),immutable_manifest_sha256=sha(manifest_path),
    complete_metrics=metrics, report=relative(HERE/'REPORT.md'),
    metrics_audit=relative(HERE/'metrics-audit.json'),execution_audit=relative(HERE/'execution-audit.json'),
    evidence_index=relative(HERE/'evidence-index.json'), checks=audit['checks'],failures=0,
    learned_router_expansion_point_screen_passed=screen,
    point_screen_not_statistical_test=True, original_timeout_preserved=True,
    no_speedup_or_equivalence_claim=True,new_GPU_hourly_rate=None)
state_paths=[ROOT/'logs'/name for name in ['babylm-stage-w-current.json','babylm-w-migration-current-20260921.json','control-state.json','research-scope-review-current.json']]
archive=ROOT/'logs/w-union-completion-state-before-20260921'
archive.mkdir(exist_ok=False)
for path in state_paths:
    (archive/path.name).write_bytes(path.read_bytes())
    data=read(path)
    data.update(updated_utc=now,status='W_complete_dev_union18792_independently_verified',
        current_scientific_state='One-epoch W complete dev, two disjoint evaluation segments; comparison is one paired-seed development result.',
        full_dev_completed=True,full_dev_partial_only=False,remaining_full_dev_windows=0,
        remaining_full_dev_loss_tokens=0,active_training_jobs=0,active_evaluation_jobs=0,
        new_evaluation_running=False,registered_project_processes_exited=True,project_gpu_lock_free=True,
        automatic_monitoring='PAUSED',complete_dev_union=completion,
        current_full_dev_processes=[],latest_verified_full_dev_counts=dict(forward_calls=18792,loss_tokens=17418742,input_tokens=17437534),
        next_action='Report complete audited D/E/F/W result; no new experiment or automatic monitoring launched.')
    data.setdefault('current_remainder',{}).update(completion)
    data['current_remainder'].update(new_committed_windows=6390,cumulative_committed_windows=18792,remaining_windows=0,live_processes=[])
    data.update(latest_verified_full_dev_live=dict(scope='complete_two_segment_union',terminal_receipt=relative(terminal_immutable),forward_calls=18792,remaining_windows=0),
                latest_full_dev_backup=relative(HERE/'execution-audit.json'))
    # current_full_dev records the immutable original timed-out segment; do not rewrite its status.
    for name in ['outer_pid','controller_pid','training_pid','pgid']:
        if name in data: data[name]=None
    if path.name=='control-state.json':
        data.update(automatic_monitoring_status='PAUSED',automatic_monitoring_paused_by_user=True,
                    current_authoritative_state='logs/babylm-stage-w-current.json',
                    active_job_scope='No active W training/evaluation jobs',
                    ready_for_user_to_stop_pod=False,
                    pod_stop_note='Project jobs finished; other shared GPU jobs may exist. Do not stop shared host automatically.')
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

values='; '.join(f"{tag} PPL={metrics[tag]['ppl']:.6f}" for tag in ['D','E','F','W'])
entry=(f"\n## {now} — W完整开发集补齐并独立核验\n\n"
       f"- 原12,402窗超时前缀完整保留；新补6,390窗，合计18,792窗、17,418,742监督token，无重复/遗漏。新增0训练步、0反向。\n"
       f"- 完整token加权结果：{values}。单种子开发集点估计，不作统计优势/等价、完整Qwen或加速结论。\n"
       f"- 独立{audit['checks']:,}项核验通过，所有原始日志已备份并逐SHA检查；报告 `results/babylm-stage-w-complete-20260921/REPORT.md`。\n"
       f"- 两段评分活动时间合计{audit['timing_segments']['summed_scorer_active_wall_seconds']:.3f}秒；新主机租价未知，未虚构费用或抹去超时成本。\n"
       f"- 本项目进程退出、锁释放；共享GPU其他任务未改动，未停Pod。自动监控仍PAUSED，未追加实验。\n")
state=ROOT/'STATE.md'
state.write_text(entry+'\n'+state.read_text(encoding='utf-8'),encoding='utf-8')
with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as handle: handle.write(entry)
print(json.dumps(dict(status='evidence_and_current_state_finalized',full_dev=metrics,
    report=relative(HERE/'REPORT.md'),new_model_calls=0),ensure_ascii=True))
