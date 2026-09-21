"""Launch the frozen 128-token CPU check and retain lifecycle evidence."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import configuration

def now():return datetime.now(timezone.utc).isoformat()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    run=ROOT/'results/zoology-length128-cpu-v0'
    log=ROOT/'logs/zoology-length128-cpu-v0.log'
    before=ROOT/'provenance/zoology-length128-start-2026-09-14'
    assert not run.exists() and not log.exists()
    before.mkdir(exist_ok=False)
    for rel in ['STATE.md','README.md','TIMELINE.md','logs/control-state.json','docs/zoology-length128-plan-2026-09-14.md']:
        target=before/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    save(ROOT/'configs/zoology-length128-v0.json',configuration(128).model_dump(serialize_as_any=True))
    started=now();statepath=ROOT/'logs/control-state.json'
    state=json.loads(statepath.read_text(encoding='utf-8'))
    state.update(updated_utc=started,status='zoology_length128_running_cpu',active_training_jobs=1,
        current_turn_gpu_jobs_started=0,correctness_tests_passed=38,
        current_decision_report='docs/zoology-length128-plan-2026-09-14.md',
        next_action='Finish frozen length128 CPU run, independent fresh evaluation and query intervention; no GPU launch.')
    save(statepath,state)
    old=(ROOT/'STATE.md').read_text(encoding='utf-8')
    (ROOT/'STATE.md').write_text(f'# 本轮正在运行：Zoology长度128的CPU检查\n\n启动：{started}。38项检查通过，序列与位置表64改128，其他作者配置保留。结果待产生；独立目录results/zoology-length128-cpu-v0，日志logs/zoology-length128-cpu-v0.log。新增GPU作业0。\n\n以下为上一轮已完成状态：\n\n'+old,encoding='utf-8')
    command=[sys.executable,str(ROOT/'src/run_zoology_baseline.py'),'--output',str(run),'--sequence-length','128','--max-seconds','3600']
    save(ROOT/'logs/zoology-length128-launch.json',dict(started_utc=started,command=command,status='running',device='cpu'))
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write(f'\n- {started}：冻结长度128单因素计划，38项CPU检查通过，开始zoology-length128-cpu-v0；同一作者优化流程从头训练，GPU作业0。\n')
    code=-1
    try:
        with log.open('x',encoding='utf-8') as output:
            process=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        code=process.returncode
    finally:
        finished=now();state=json.loads(statepath.read_text(encoding='utf-8'))
        state.update(updated_utc=finished,active_training_jobs=0,status='zoology_length128_training_finished_pending_audit' if code==0 else 'zoology_length128_failed_pending_audit')
        save(statepath,state)
        save(ROOT/'logs/zoology-length128-launch.json',dict(started_utc=started,finished_utc=finished,command=command,exit_code=code,status='finished' if code==0 else 'failed',device='cpu'))
        with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:f.write(f'\n- {finished}：zoology-length128-cpu-v0进程结束，退出码{code}，等待结果审计与查询干预。\n')
    print(json.dumps(dict(exit_code=code,run=str(run),log=str(log))),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
