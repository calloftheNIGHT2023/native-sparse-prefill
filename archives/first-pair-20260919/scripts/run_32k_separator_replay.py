"""Bounded evaluation-only preprocessing correction; preserve original stage."""
import json,hashlib,subprocess,sys,time,tarfile,io
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pp=R/'provenance/32k-separator-replay-protocol.json';protocol=json.loads(pp.read_text());assert sha(Path(__file__))==protocol['controller_sha256']
 original=R/'results/32k-adaptation-stage-v0';assert json.loads((original/'result.json').read_text())['status']=='complete'
 cfg=json.loads((R/'data/32k-adaptation-v0/config.json').read_text());chosen=json.loads((original/'selection-lock.json').read_text())['selected_lrs']
 out=R/'results/32k-separator-replay-stage-v0';out.mkdir(exist_ok=False);tick=time.monotonic();start=utc();runs=[];error=None
 conditions=[(k,s,64) for k in [0,32] for s in cfg['seeds']]+[(k,cfg['seeds'][0],0) for k in [0,32]]
 try:
  for k,seed,step in conditions:
   lr=chosen[str(k)];parent=original/(f'cal-k{k}-lr{lr:g}' if seed==cfg['seeds'][0] else f'repeat-k{k}')
   name=f'k{k}-seed{seed}-step{step}';remaining=protocol['max_seconds']-(time.monotonic()-tick);assert remaining>20
   cmd=[sys.executable,'-u',str(R/'scripts/eval_32k_separator_replay.py'),'--phase','report','--output',str(out/name),'--k',str(k),'--seed',str(seed),'--lr',str(lr),'--preflight',str(original/f'preflight-k{k}'),'--selection',str(original/'selection-lock.json'),'--resume',str(parent/f'checkpoint-{step}.pt')]
   row=dict(name=name,started_utc=utc(),command=cmd)
   with (out/f'{name}.log').open('w') as f:
    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=R)
    try:code=proc.wait(timeout=min(75,remaining))
    except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
   row.update(returncode=code,finished_utc=utc());runs.append(row);(out/'timeline.json').write_text(json.dumps(runs,indent=2));print(json.dumps(row),flush=True)
   if code:raise RuntimeError(name+' failed')
  status='complete'
 except Exception as e:status='failed';error=repr(e)
 result=dict(status=status,error=error,started_utc=start,finished_utc=utc(),seconds=time.monotonic()-tick,optimizer_updates=0,runs=runs,protocol_sha256=sha(pp));(out/'result.json').write_text(json.dumps(result,indent=2))
 files=[p for p in out.rglob('*') if p.is_file()]+[pp,Path(__file__),R/'scripts/eval_32k_separator_replay.py',R/'data/32k-separator-replay-v0/report.npz']
 archive=R/'exports/32k-separator-replay-evidence-v0.tar.gz';assert not archive.exists();entries=[]
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(set(files)):
   raw=p.read_bytes();name=p.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=sha(p)));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('32k-separator-replay-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(sha256=sha(archive),bytes=archive.stat().st_size,files=len(entries));archive.with_suffix('.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
 if status!='complete':raise SystemExit(1)
if __name__=='__main__':main()
