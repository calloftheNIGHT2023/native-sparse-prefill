"""Restore pinned baseline in a new venv and compile one isolated barrier candidate."""
import os,json,hashlib,subprocess,sys,tarfile,time,traceback,urllib.request,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-backward-environment-v3';OUT.mkdir(parents=True,exist_ok=False)
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
    run('baseline-import',[py,'-c','import torch,flash_moba_cuda;print(torch.__version__,torch.cuda.get_device_name(),flash_moba_cuda.__file__)'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==COMMIT
    # Migration left tracked source files absent. Restore only missing tracked files;
    # retain all existing files and record the exact before/after status.
    for label,repo in [('main',REPO),('cutlass',REPO/'csrc/cutlass')]:
        before=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
        (OUT/f'{label}-migration-before.txt').write_text(before)
        deleted=subprocess.check_output(['git','diff','--name-only','--diff-filter=D','-z'],cwd=repo)
        if deleted:
            p=OUT/f'{label}-missing-paths.bin';p.write_bytes(deleted)
            run(f'{label}-restore-missing',['git','restore','--source=HEAD','--worktree',f'--pathspec-from-file={p}','--pathspec-file-nul'],cwd=repo)
    assert CUTLASS in subprocess.check_output(['git','submodule','status'],cwd=REPO,text=True)
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    event('migrated_source_restored',commit=COMMIT,cutlass=CUTLASS)
    assert not TK.exists();TK.mkdir(parents=True);cache=ROOT/'toolchains/downloads';cache.mkdir(exist_ok=True)
    specs=[('cuda_nvcc','12.9.86','7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'),
      ('cuda_cudart','12.9.79','1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99'),
      ('cuda_cccl','12.9.27','8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa')]
    packages=[]
    for component,version,digest in specs:
        name=f'{component}-linux-x86_64-{version}-archive.tar.xz';url=f'https://developer.download.nvidia.com/compute/cuda/redist/{component}/linux-x86_64/{name}'
        p=cache/name;urllib.request.urlretrieve(url,p);assert sha(p)==digest
        with tarfile.open(p) as t:t.extractall(cache,filter='data')
        folder=cache/name[:-7]
        for sub in ['bin','include','lib','lib64','nvvm']:
            if (folder/sub).exists():shutil.copytree(folder/sub,TK/sub,dirs_exist_ok=True)
        packages.append(dict(url=url,sha256=digest));event('toolkit_component_ready',component=component)
    if not (TK/'lib64').exists():(TK/'lib64').symlink_to('lib',target_is_directory=True)
    save(OUT/'toolkit.json',packages)
    assert not CAND.exists();shutil.copytree(REPO,CAND,symlinks=True)
    patch=ROOT/'patches/flashmoba-shared-index-barrier-candidate-v0.patch'
    run('apply-check',['git','apply','--check',patch],cwd=CAND);run('apply',['git','apply',patch],cwd=CAND)
    (OUT/'candidate.diff').write_text(subprocess.check_output(['git','diff'],cwd=CAND,text=True))
    inc=sorted((ENV/'lib/python3.11/site-packages/nvidia').glob('*/include'));assert any((x/'cusparse.h').exists() for x in inc)
    env=os.environ.copy();env.update(CUDA_HOME=str(TK),MAX_JOBS='4',NVCC_THREADS='2',FLASH_MOBA_CUDA_ARCHS='80',FLASH_MOBA_FORCE_BUILD='TRUE',CPATH=':'.join(map(str,inc)))
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
