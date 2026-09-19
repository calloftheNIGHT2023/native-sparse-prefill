"""Serialize a bounded cost screen after audited extension; retain failed candidates."""
import hashlib,io,json,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 pp=R/'provenance/m64-density-cost-protocol.json';p=json.loads(pp.read_text());statefile=R/'logs/m64-density-cost-queue-v0.json';start=time.monotonic()
 for name,h in p['source_sha256'].items():assert sha(R/name)==h
 assert json.loads((R/'results/topk-m64-math-v0/result.json').read_text())['status']=='passed'
 assert json.loads((R/'results/fresh256-stage-v0/result.json').read_text())['status']=='complete'
 save(statefile,dict(status='waiting_for_extension_audit',utc=utc(),protocol_sha256=sha(pp)))
 while True:
  audit_file=R/'results/32k-extension-audit-v0/result.json'
  try:
   parent=json.loads(audit_file.read_text())
   if parent['status']=='complete':break
  except (FileNotFoundError,json.JSONDecodeError):pass
  assert time.monotonic()-start<7200
  f=R/'logs/32k-extension-chain-v0.json'
  if f.exists():assert json.loads(f.read_text())['status'] not in ['failed','extension_failed']
  time.sleep(15)
 parent=json.loads((R/'results/32k-extension-audit-v0/result.json').read_text());assert parent['status']=='complete'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; no concurrent launch'
 out=R/'results/m64-density-cost-stage-v0';out.mkdir(exist_ok=False);tick=time.monotonic();started=utc();jobs=[];failed=set()
 save(statefile,dict(status='running',utc=utc(),protocol_sha256=sha(pp),output=str(out)))
 try:
  for ri,order in enumerate(p['orders']):
   for mode in order:
    name=f'round{ri}-k{mode}'
    if mode in failed:
     jobs.append(dict(name=name,round=ri,k=mode,status='skipped_after_candidate_failure',utc=utc()));save(out/'timeline.json',jobs);continue
    remain=p['maximum_seconds']-(time.monotonic()-tick);assert remain>30
    dest=out/name;cmd=[sys.executable,'-u',str(R/'scripts/benchmark_m64_density_cost.py'),'--phase','train','--output',str(dest),'--k','0','--seed','2026091660','--lr','0.001','--mode',str(mode),'--round',str(ri),'--preflight',str(R/'results/32k-adaptation-stage-v0/preflight-k0'),'--resume',str(R/p['parent_checkpoint'])]
    row=dict(name=name,round=ri,k=mode,status='running',started_utc=utc(),command=cmd);jobs.append(row);save(out/'timeline.json',jobs)
    with (out/f'{name}.log').open('w') as f:
     proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
     try:code=proc.wait(timeout=min(150,remain))
     except subprocess.TimeoutExpired:proc.kill();proc.wait();code=-9
    row.update(status='complete' if code==0 else 'failed',finished_utc=utc(),returncode=code);save(out/'timeline.json',jobs)
    if code:
     failed.add(mode)
     if mode==0:raise RuntimeError('Dense reference failed; speed conclusions unavailable')
  status='complete' if not failed else 'complete_with_candidate_failures'
 except Exception:
  status='failed';(out/'error.txt').write_text(traceback.format_exc())
 diagnostic_updates=0;measurements=[];incomplete_update_accounting=[]
 for j in jobs:
  if j.get('status')=='failed' and not (out/j['name']/'result.json').exists():incomplete_update_accounting.append(j['name'])
 for f in out.glob('round*/result.json'):diagnostic_updates+=json.loads(f.read_text()).get('optimizer_updates_this_process',0)
 for f in out.glob('round*/measurement.json'):measurements.append(json.loads(f.read_text()))
 candidates=[]
 for mode in [32,48,64]:
  rows=sorted([x for x in measurements if x['mode']==mode],key=lambda x:x['round']);ratios=[]
  for x in rows:
   dense=next((y for y in measurements if y['mode']==0 and y['round']==x['round']),None)
   if dense:ratios.append(x['median_seconds']/dense['median_seconds'])
  candidates.append(dict(k=mode,complete_rounds=len(ratios),ratios=ratios,eligible_for_calibration_training=len(ratios)==3 and all(x<1 for x in ratios),scope='Timing screen only, no trained-quality or formal statistical superiority conclusion'))
 result=dict(status=status,started_utc=started,finished_utc=utc(),seconds=time.monotonic()-tick,jobs=jobs,failed_candidates=sorted(failed),diagnostic_optimizer_updates=diagnostic_updates,update_accounting_complete=not incomplete_update_accounting,incomplete_update_accounting=incomplete_update_accounting,scientific_optimizer_updates=0,task_predictions=0,candidates=candidates,protocol_sha256=sha(pp))
 save(out/'result.json',result)
 a=R/'exports/m64-density-cost-evidence-v0.tar.gz';assert not a.exists();entries=[]
 files=list(out.rglob('*'))+[R/n for n in p['source_sha256']]+[pp]
 with tarfile.open(a,'w:gz',compresslevel=1) as t:
  for f in sorted(set(files)):
   if not f.is_file():continue
   raw=f.read_bytes();name=f.relative_to(R).as_posix();entries.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries),indent=2).encode();m=tarfile.TarInfo('m64-density-cost-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 save(a.with_suffix('.json'),dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries)))
 save(statefile,dict(status=status,utc=utc(),output=str(out),candidates=candidates,diagnostic_optimizer_updates=diagnostic_updates,archive_sha256=sha(a)))
if __name__=='__main__':
 try:main()
 except Exception:
  save(R/'logs/m64-density-cost-queue-v0.json',dict(status='failed',utc=utc(),error=traceback.format_exc()))
  raise
