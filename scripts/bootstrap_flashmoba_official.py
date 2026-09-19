"""Install pinned official baseline in a separate cloud env; never replace driver/system CUDA."""
import hashlib,json,os,shutil,subprocess,sys,tarfile,time,traceback,urllib.request
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/flashmoba-environment-v0'
ENV=ROOT/'.venv-flashmoba-v0'
TOOLKIT=ROOT/'toolchains/cuda-12.9.1'
REPO=ROOT/'third_party/flash-moba-official-20260915'


def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter();events=[]
    def event(kind,**data):
        row=dict(utc=utc(),elapsed_seconds=time.perf_counter()-tick,event=kind,**data);events.append(row)
        with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def run(args,env=None,cwd=ROOT,timeout=1500):
        event('command_start',command=[str(x) for x in args]);t=time.perf_counter()
        subprocess.run([str(x) for x in args],env=env,cwd=cwd,check=True,timeout=timeout)
        event('command_complete',command=[str(x) for x in args],seconds=time.perf_counter()-t)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        assert commit=='39d9ac043b271d046a2181a9991e99a26b67bca1'
        sub=subprocess.check_output(['git','submodule','status'],cwd=REPO,text=True).strip()
        assert 'a2439551c765c5393aebe557ee75d3a0412d2211' in sub
        assert not ENV.exists() and not TOOLKIT.exists()
        event('start',upstream_commit=commit,cutlass=sub,existing_environment_preserved=True)
        cache=ROOT/'toolchains/downloads';cache.mkdir(parents=True,exist_ok=True)
        specs=[('cuda_nvcc','12.9.86','7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'),
            ('cuda_cudart','12.9.79','1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99'),
            ('cuda_cccl','12.9.27','8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa')]
        TOOLKIT.mkdir(parents=True)
        packages=[]
        for component,version,digest in specs:
            name=f'{component}-linux-x86_64-{version}-archive.tar.xz'
            url=f'https://developer.download.nvidia.com/compute/cuda/redist/{component}/linux-x86_64/{name}'
            path=cache/name
            if not path.exists():urllib.request.urlretrieve(url,path)
            assert sha(path)==digest
            extract=cache/(name[:-7]);assert not extract.exists()
            with tarfile.open(path) as tar:
                tar.extractall(cache,filter='data')
            assert extract.exists()
            for directory in ['bin','include','lib','lib64','nvvm']:
                if (extract/directory).exists():shutil.copytree(extract/directory,TOOLKIT/directory,dirs_exist_ok=True)
            packages.append(dict(url=url,sha256=digest,bytes=path.stat().st_size))
            event('toolkit_component',component=component,version=version)
        if not (TOOLKIT/'lib64').exists():(TOOLKIT/'lib64').symlink_to('lib',target_is_directory=True)
        save(OUT/'toolkit-manifest.json',packages)
        run([TOOLKIT/'bin/nvcc','--version'])
        run([sys.executable,'-m','venv',ENV])
        py=ENV/'bin/python'
        run([py,'-m','pip','install','--upgrade','pip','setuptools','wheel'],timeout=300)
        run([py,'-m','pip','install','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cu128'],timeout=900)
        run([py,'-m','pip','install','numpy==1.26.3','einops==0.8.2','packaging','ninja','psutil','pytest',
             'pydantic==2.13.5','pandas==2.2.3','wandb==0.30.0','transformers==4.57.6'],timeout=600)
        env=os.environ.copy();env['CUDA_HOME']=str(TOOLKIT);env['PATH']=str(TOOLKIT/'bin')+':'+str(ENV/'bin')+':'+env['PATH']
        env['MAX_JOBS']='4';env['NVCC_THREADS']='2';env['FLASH_MOBA_CUDA_ARCHS']='80';env['FLASH_MOBA_FORCE_BUILD']='TRUE'
        # Official supported build flag emits sm80 cubins, executable on Ada sm89 without newer-driver PTX JIT.
        save(OUT/'build-options.json',{k:env[k] for k in ['CUDA_HOME','MAX_JOBS','NVCC_THREADS','FLASH_MOBA_CUDA_ARCHS','FLASH_MOBA_FORCE_BUILD']})
        run([py,'-c','import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); x=torch.randn(32,32,device="cuda"); print(float((x@x).sum()))'],env=env)
        run([py,'-m','pip','install','--no-build-isolation','--no-deps','.'],env=env,cwd=REPO,timeout=1800)
        run([py,'-c','import torch, flash_moba, flash_moba_cuda; print(torch.__version__, flash_moba.__file__, flash_moba_cuda.__file__)'],env=env)
        freeze=subprocess.check_output([py,'-m','pip','freeze'],text=True);(OUT/'pip-freeze.txt').write_text(freeze)
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip();assert not dirty
        status='complete';error=None
    except Exception as exc:
        status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
    save(OUT/'environment.json',dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        environment=str(ENV),toolkit=str(TOOLKIT),scientific_optimizer_updates=0,system_driver_modified=False,old_environment_modified=False))
    if status!='complete':raise SystemExit(1)


if __name__=='__main__':main()
