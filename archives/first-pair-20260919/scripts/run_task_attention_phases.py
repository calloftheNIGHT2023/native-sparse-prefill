"""Bounded head-selective screen with frozen protocol and complete evidence."""
import hashlib, io, json, subprocess, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def main():
 protocol_path=R/'provenance/task-attention-phases-protocol.json';protocol=read(protocol_path)
 for name,h in protocol['sources'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
 out=R/'results/task-attention-phases-stage-v0';out.mkdir(exist_ok=False)
 start=utc();tick=time.monotonic();runs=[];gate=None;error=None
 jobs=[('preflight',[sys.executable,'-u',str(R/'scripts/check_task_attention_phases.py')],120)]
 for seed,parent in [(2026091560,R/'parents/k16-seed0'),(2026091561,R/'results/amp-recovery-stage-v0/repeat-k16')]:
  name=f'seed{seed}';jobs.append((name,[sys.executable,'-u',str(R/'scripts/eval_task_attention_phases.py'),'--run-dir',str(parent),'--output',str(out/name)],protocol['per_job_max_seconds']))
 try:
  for name,cmd,cap in jobs:
   remaining=protocol['max_seconds']-(time.monotonic()-tick)
   if remaining<15:raise TimeoutError('frozen total runtime cap')
   row=dict(name=name,started_utc=utc(),command=cmd,timeout_seconds=min(cap,remaining))
   with (out/f'{name}.log').open('w') as f:
    p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
    try:code=p.wait(timeout=row['timeout_seconds'])
    except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
   row.update(returncode=code,finished_utc=utc());runs.append(row)
   (out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
   print(json.dumps(row),flush=True)
   if code:raise RuntimeError(name+' failed')
  scores=[]
  for seed in [2026091560,2026091561]:
   d=read(out/f'seed{seed}'/'result.json')
   scores.append({c:sum(x['correct'] for x in d['predictions'] if x['condition']==c) for c in protocol['conditions']})
  delta={candidate:{control:[z[candidate]-z[control] for z in scores] for control in controls} for candidate,controls in protocol['gate'].items()}
  passed={candidate:all(min(delta[candidate][control])>=g['each_seed'] and sum(delta[candidate][control])/2>=g['mean'] for control,g in controls.items()) for candidate,controls in protocol['gate'].items()}
  gate=dict(passed=passed,scores=scores,delta_correct=delta)
  (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n');status='complete'
 except Exception as exc:status='failed';error=repr(exc)
 result=dict(status=status,error=error,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-tick,
             runs=runs,gate=gate,optimizer_updates=0,protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest())
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 files=[p for sub in [out,R/'results/task-attention-phases-preflight-v0'] for p in sub.rglob('*') if p.is_file()]
 files += [R/name for name in protocol['sources']] + [protocol_path]
 files += [R/'provenance/task-quality-route-chain-protocol.json',R/'docs/task-attention-phases-protocol-2026-09-15.md']
 archive=R/'exports/task-attention-phases-evidence-v0.tar.gz';assert not archive.exists();entries=[]
 with tarfile.open(archive,'w:gz') as tar:
  for path in sorted(set(files)):
   raw=path.read_bytes();name=path.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
   item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries,completed_job_directories=[f'results/task-attention-phases-stage-v0/{r["name"]}' for r in runs if r['name']!='preflight' and r['returncode']==0]),indent=2).encode()
  item=tarfile.TarInfo('amp-recovery-results-manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 proof=dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(entries))
 archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps(dict(result=result,archive=proof)),flush=True)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
