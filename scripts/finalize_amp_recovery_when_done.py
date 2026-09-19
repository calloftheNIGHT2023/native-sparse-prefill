"""One-shot finalization of the already-running bounded stage; launches no training."""
import argparse,json,time,subprocess,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def main(a):
 deadline=time.monotonic()+7800
 while not (a.stage/'result.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('No stage result before finalizer deadline; inspect manually.')
  time.sleep(15)
 result=json.loads((a.stage/'result.json').read_text());analysis_code=None
 if result['status']=='complete' and not result['preflight_only']:
  with (a.stage/'analysis.log').open('w') as f:
   analysis_code=subprocess.run([sys.executable,str(R/'scripts/analyze_amp_recovery.py'),'--stage',str(a.stage)],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
 for p in Path('/workspace').glob('amp-recovery-*.log'):shutil.copyfile(p,R/'provenance'/p.name)
 summary=dict(utc=datetime.now(timezone.utc).isoformat(),stage_status=result['status'],analysis_returncode=analysis_code,training_launched_by_finalizer=False)
 (R/'provenance/amp-recovery-finalizer-status.json').write_text(json.dumps(summary,indent=2)+'\n')
 subprocess.run([sys.executable,str(R/'scripts/package_amp_recovery_results.py'),'--archive',str(a.archive)],cwd=R,check=True,timeout=600)
 a.archive.with_suffix('.ready.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);main(p.parse_args())
