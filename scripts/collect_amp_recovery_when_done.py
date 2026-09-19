"""One-shot artifact transfer for this running batch, with a bounded polling lifetime."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def main(a):
 flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes','-i',a.key,'-p',str(a.port),f'root@{a.host}']
 deadline=time.monotonic()+8400
 ready='/workspace/amp-recovery-final-evidence-v0.tar.ready.json'
 while time.monotonic()<deadline:
  try:
   c=subprocess.run(ssh+[f'test -f {ready}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20,**flags)
   if c.returncode==0:break
  except subprocess.TimeoutExpired:pass
  time.sleep(30)
 else:raise TimeoutError('Final archive not ready; cloud training was not modified.')
 scp=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes','-i',a.key,'-P',str(a.port)]
 for name in ['amp-recovery-final-evidence-v0.tar.gz','amp-recovery-final-evidence-v0.tar.json','amp-recovery-final-evidence-v0.tar.ready.json']:
  subprocess.run(scp+[f'root@{a.host}:/workspace/{name}',str(R/'exports'/name)],check=True,timeout=900,**flags)
 subprocess.run([sys.executable,str(R/'scripts/verify_amp_recovery_results.py'),'--archive',str(R/'exports/amp-recovery-final-evidence-v0.tar.gz'),'--output',str(R/'results/cloud-amp-recovery-final-evidence-v0')],check=True,**flags)
 p=dict(status='downloaded_and_verified',utc=datetime.now(timezone.utc).isoformat(),cloud_stop_performed=False)
 (R/'logs/amp-recovery-automatic-collection.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--key',required=True);main(p.parse_args())
