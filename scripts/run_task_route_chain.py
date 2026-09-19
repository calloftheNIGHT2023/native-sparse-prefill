"""E1a then conditionally E1b, frozen gate, bounded total runtime, full evidence."""
import json,subprocess,sys,time,tarfile,io,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def main():
 protocol=read(R/'provenance/task-quality-route-chain-protocol.json');out=R/'results/task-route-chain-stage-v0';out.mkdir(exist_ok=False)
 start=utc();tick=time.monotonic();runs=[];gate=None;error=None
 try:
  for part in ['screen','remaining']:
   if part=='remaining' and not gate['passed']:break
   scores=[]
   for seed,parent in [(2026091560,R/'parents/k16-seed0'),(2026091561,R/'results/amp-recovery-stage-v0/repeat-k16')]:
    remain=protocol['max_seconds']-(time.monotonic()-tick)
    if remain<30:raise TimeoutError('Chain time cap')
    name=f'{part}-seed{seed}';cmd=[sys.executable,'-u',str(R/'scripts/eval_task_route_chain.py'),'--run-dir',str(parent),'--output',str(out/name),'--part',part]
    row=dict(name=name,started_utc=utc(),command=cmd)
    with (out/f'{name}.log').open('w') as f:
     p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
     try:code=p.wait(timeout=min(750,remain))
     except subprocess.TimeoutExpired:p.kill();p.wait();code=-9
    row.update(returncode=code,finished_utc=utc());runs.append(row);(out/'timeline.json').write_text(json.dumps(runs,indent=2)+'\n')
    if code:raise RuntimeError(name+' failed')
    d=read(out/name/'result.json');scores.append({c:sum(x['correct'] for x in d['predictions'] if x['condition']==c) for c in protocol['conditions']})
   if part=='screen':
    b=[s['target']-s['baseline'] for s in scores];c=[s['target']-s['sham'] for s in scores];g=protocol['gate']
    gate=dict(utc=utc(),scores=scores,target_minus_baseline=b,target_minus_sham=c,passed=min(b)>=g['each_seed_target_minus_baseline_min_correct'] and sum(b)/2>=g['mean_target_minus_baseline_min_correct'] and min(c)>=g['each_seed_target_minus_sham_min_correct'] and sum(c)/2>=g['mean_target_minus_sham_min_correct'])
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate),flush=True)
  status='complete'
 except Exception as e:status='failed';error=str(e)
 result=dict(status=status,error=error,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-tick,runs=runs,gate=gate,optimizer_updates=0)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 files=[p for sub in [out,R/'results/task-route-preflight-v0',R/'results/task-route-preflight-v1'] for p in sub.rglob('*') if p.is_file()]
 files += [R/p for p in ['scripts/task_route_intervention.py','scripts/check_task_route_intervention.py','scripts/eval_task_route_chain.py','scripts/run_task_route_chain.py','provenance/task-quality-route-chain-protocol.json','docs/routing-experiment-chain-2026-09-15.md']]
 a=R/'exports/task-route-chain-evidence-v0.tar.gz';assert not a.exists();rows=[]
 with tarfile.open(a,'w:gz') as t:
  for p in sorted(files):
   raw=p.read_bytes();name=p.relative_to(R).as_posix();rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=rows,completed_job_directories=[f"results/task-route-chain-stage-v0/{x['name']}" for x in runs if x['returncode']==0]),indent=2).encode();m=tarfile.TarInfo('amp-recovery-results-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest(),bytes=a.stat().st_size,files=len(rows));a.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
 if status=='failed':raise SystemExit(1)
if __name__=='__main__':main()
