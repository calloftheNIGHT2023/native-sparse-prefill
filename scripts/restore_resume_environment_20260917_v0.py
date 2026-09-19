"""Restore exactly pinned dependencies on the migrated Pod; retain old empty package tree."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,json,hashlib,time,traceback
R=Path(__file__).resolve().parents[1];out=R/'results/resume-environment-20260917-v0';env=Path('/workspace/nsp-resume-env-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out.mkdir(exist_ok=False);started=datetime.now(timezone.utc).isoformat();tick=time.monotonic();state=R/'logs/resume-environment-20260917-v0.json'
 def save(v):state.write_text(json.dumps(v,indent=2)+'\n')
 save(dict(status='restoring',started_utc=started,optimizer_updates=0));(out/'source.py').write_bytes(Path(__file__).read_bytes())
 try:
  assert not env.exists();subprocess.run([sys.executable,'-m','venv',str(env)],check=True,timeout=60);py=str(env/'bin/python');pins=R/'provenance/day-end-20260916-v0/pinned-packages.txt'
  with (out/'install.log').open('w') as f:subprocess.run([py,'-m','pip','install','--no-deps','--progress-bar','off','--extra-index-url','https://download.pytorch.org/whl/cu128','-r',str(pins)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1500)
  wheel=R/'exports/flashmoba-barrier-wheels-v0/flash_moba-2.0.0-cp311-cp311-linux_x86_64.whl'
  with (out/'extension.log').open('w') as f:
   subprocess.run([py,'-m','pip','install','--no-deps',str(wheel)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
   subprocess.run([py,'-m','pip','check'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=30)
  cmd='import torch,transformers,flash_moba_cuda,json,hashlib;print(json.dumps(dict(torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),extension_sha256=hashlib.sha256(open(flash_moba_cuda.__file__,"rb").read()).hexdigest())))'
  v=json.loads(subprocess.check_output([py,'-c',cmd],text=True));assert v['torch']=='2.8.0+cu128' and v['transformers']=='4.57.6' and v['gpu']=='NVIDIA GeForce RTX 4090' and v['capability']==[8,9] and v['extension_sha256']=='72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c'
  m=json.loads((R/'data/flashmoba-qwen-precision-v0/manifest.json').read_text())
  for x in m['files']:
   if x['path'].startswith('model/'):assert sha(R/'data/flashmoba-qwen-precision-v0'/x['path'])==x['sha256']
  result=dict(status='complete',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick,python=py,environment=v,model_hashes_verified=True,optimizer_updates=0,scope='Environment only. Fresh checkpoint calibration and saved-row replay required before continuing.')
 except Exception:result=dict(status='failed',started_utc=started,error=traceback.format_exc(),optimizer_updates=0)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');save(result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
