"""Restore pinned baseline in a new venv and compile one isolated barrier candidate."""
import os,json,hashlib,subprocess,sys,tarfile,time,traceback,urllib.request,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-backward-environment-v5';OUT.mkdir(parents=True,exist_ok=False)
ENV=Path('/opt/native-sparse-flashmoba-env-v2');TK=ROOT/'toolchains/cuda-12.9.1'
REPO=ROOT/'third_party/flash-moba-official-20260915';CAND=ROOT/'third_party/flash-moba-barrier-candidate-v0'
COMMIT='39d9ac043b271d046a2181a9991e99a26b67bca1';CUTLASS='a2439551c765c5393aebe557ee75d3a0412d2211'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
start=utc();tick=time.perf_counter();(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
def event(kind,**kw):
    obj=dict(utc=utc(),seconds=time.perf_counter()-tick,event=kind,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
    print(json.dumps(obj),flush=True)
def run(label,cmd,cwd=ROOT,env=None,timeout=1500):
    event('command_start',label=label,command=[str(x) for x in cmd]);st=time.perf_counter()
    with (OUT/f'{label}.log').open('w') as f:r=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    event('command_end',label=label,returncode=r.returncode,elapsed=time.perf_counter()-st)
    if r.returncode:raise RuntimeError(f'{label} failed: '+(OUT/f'{label}.log').read_text()[-6000:])
try:
    py=ENV/'bin/python'
    wheel=Path('/workspace/flash_moba-2.0.0-cp311-cp311-linux_x86_64.whl')
    event('resume_compile',max_jobs=12,nvcc_threads=1,retain_completed_objects=True)
    (OUT/'candidate.diff').write_text(subprocess.check_output(['git','diff'],cwd=CAND,text=True))
    inc=sorted((ENV/'lib/python3.11/site-packages/nvidia').glob('*/include'));assert any((x/'cusparse.h').exists() for x in inc)
    env=os.environ.copy();env.update(CUDA_HOME=str(TK),MAX_JOBS='12',NVCC_THREADS='1',FLASH_MOBA_CUDA_ARCHS='80',FLASH_MOBA_FORCE_BUILD='TRUE',CPATH=':'.join(map(str,inc)))
    env['PATH']=str(TK/'bin')+':'+str(ENV/'bin')+':'+env['PATH']
    save(OUT/'build-env.json',{k:env[k] for k in ['CUDA_HOME','MAX_JOBS','NVCC_THREADS','FLASH_MOBA_CUDA_ARCHS','FLASH_MOBA_FORCE_BUILD','CPATH']})
    run('candidate-build',[py,'setup.py','build_ext','--inplace'],cwd=CAND,env=env,timeout=2100)
    so=list(CAND.glob('flash_moba_cuda*.so'));assert len(so)==1
    run('candidate-import',[py,'-c','import torch,flash_moba_cuda; print(torch.__version__,flash_moba_cuda.__file__)'],cwd=CAND,env=env)
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    save(OUT/'environment.json',dict(status='complete',started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        environment=str(ENV),candidate=str(CAND),candidate_extension_sha256=sha(so[0]),original_wheel_sha256=sha(wheel),toolkit=str(TK),
        original_repo_unchanged=True,original_environment_preserved=True,scientific_optimizer_updates=0))
    (OUT/'pip-freeze.txt').write_text(subprocess.check_output([py,'-m','pip','freeze'],text=True));event('complete')
except Exception:
    save(OUT/'environment.json',dict(status='incomplete',started_utc=start,finished_utc=utc(),error=traceback.format_exc(),scientific_optimizer_updates=0));event('failed',error=traceback.format_exc());raise
