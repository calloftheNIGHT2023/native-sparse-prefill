"""Restore pinned runtime and public model without changing recorded experiments."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,sys,time,json,hashlib,urllib.request,traceback,concurrent.futures,os
R=Path(__file__).resolve().parents[1];PY='/workspace/nsp-env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'results/restore-midpoint4090-v0';out.mkdir(exist_ok=False);state=R/'logs/restore-midpoint4090-v0.json';started=datetime.now(timezone.utc).isoformat()
    def save(v):state.write_text(json.dumps(v,indent=2)+'\n')
    def env():
        # The separately started torch installation must finish before pip runs again.
        limit=time.monotonic()+600
        while Path('/proc/1716/cmdline').exists() and b'pip' in Path('/proc/1716/cmdline').read_bytes():
            assert time.monotonic()<limit,'Torch installation timeout';time.sleep(5)
        log=(out/'install-pins.log').open('w')
        subprocess.run([PY,'-m','pip','install','--no-deps','--progress-bar','off','-r',str(R/'provenance/day-end-20260916-v0/pinned-packages.txt')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=480)
        subprocess.run([PY,'-m','pip','install','--no-deps',str(R/'exports/flashmoba-barrier-wheels-v0/flash_moba-2.0.0-cp311-cp311-linux_x86_64.whl')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        subprocess.run([PY,'-m','pip','check'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
    def model():
        base=R/'data/flashmoba-qwen-precision-v0';m=json.loads((base/'manifest.json').read_text())
        for x in m['files']:
            if not x['path'].startswith('model/'):continue
            f=base/x['path']
            if not f.exists():
                assert f.suffix=='.safetensors'
                url='https://huggingface.co/'+m['model_id']+'/resolve/'+m['revision']+'/'+f.name
                req=urllib.request.Request(url,headers={'User-Agent':'native-sparse-research-recovery'})
                tmp=f.with_suffix('.download')
                with urllib.request.urlopen(req,timeout=60) as src,tmp.open('wb') as dst:
                    while chunk:=src.read(1024*1024):dst.write(chunk)
                assert sha(tmp)==x['sha256'];tmp.rename(f)
            assert sha(f)==x['sha256'] and f.stat().st_size==x['bytes'],x['path']
    save(dict(status='restoring',started_utc=started,optimizer_updates=0))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(env),pool.submit(model)]
            for j in jobs:j.result()
        code="import torch,transformers,flash_moba_cuda,json,hashlib;print(json.dumps(dict(torch=str(torch.__version__),cuda=torch.version.cuda,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),extension_sha256=hashlib.sha256(open(flash_moba_cuda.__file__,'rb').read()).hexdigest())))"
        v=json.loads(subprocess.check_output([PY,'-c',code],text=True))
        assert v['torch']=='2.8.0+cu128' and v['transformers']=='4.57.6' and v['capability']==[8,9]
        assert v['extension_sha256']=='72c3fda9e7d4da701bfc76b0e32f4d80fdbeec70d462c83b87b97f51d887ed1c'
        result=dict(status='complete',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),environment=v,optimizer_updates=0,scope='Environment and public model restored; task/calibration numerical replay not yet executed.')
    except Exception:result=dict(status='failed',started_utc=started,error=traceback.format_exc(),optimizer_updates=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');save(result);print(json.dumps(result),flush=True)
    if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
