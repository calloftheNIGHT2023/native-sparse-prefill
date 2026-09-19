"""Controlled continuation of this project's compiler, retaining v0 and cached objects."""
import argparse,hashlib,json,os,signal,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/flashmoba-environment-resume-v1'
ENV=ROOT/'.venv-flashmoba-v0';TK=ROOT/'toolchains/cuda-12.9.1';REPO=ROOT/'third_party/flash-moba-official-20260915'
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main(a):
    OUT.mkdir(parents=True,exist_ok=False);start=utc();tick=time.perf_counter()
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    def event(kind,**kw):
        row=dict(event=kind,utc=utc(),elapsed_seconds=time.perf_counter()-tick,**kw)
        with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        if a.ninja_pid:
            p=psutil.Process(a.ninja_pid);assert p.name()=='ninja' and str(REPO/'build') in p.cwd()
            children=[dict(pid=x.pid,created=x.create_time(),name=x.name()) for x in p.children(recursive=True)]
            event('controlled_compile_interruption',ninja_pid=p.pid,cwd=p.cwd(),children=children,
                reason='Preserve built objects and continue with six jobs after verified 62GB memory / 13.6 CPU quota')
            p.send_signal(signal.SIGINT)
            # Ninja handles child process groups; wait for exactly this build to exit before any resume.
            p.wait(timeout=90)
        old=ROOT/'results/flashmoba-environment-v0/environment.json'
        for _ in range(60):
            if old.exists():break
            time.sleep(2)
        assert old.exists()
        for p in psutil.process_iter(['name','cwd']):
            if p.info['name']=='ninja' and p.info['cwd'] and str(REPO/'build') in p.info['cwd']:
                raise RuntimeError('Old project Ninja process still active; will not run a concurrent build')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='39d9ac043b271d046a2181a9991e99a26b67bca1'
        env=os.environ.copy();env.update(CUDA_HOME=str(TK),MAX_JOBS='6',NVCC_THREADS='2',FLASH_MOBA_CUDA_ARCHS='80',FLASH_MOBA_FORCE_BUILD='TRUE')
        env['PATH']=str(TK/'bin')+':'+str(ENV/'bin')+':'+env['PATH']
        options={k:env[k] for k in ['CUDA_HOME','MAX_JOBS','NVCC_THREADS','FLASH_MOBA_CUDA_ARCHS','FLASH_MOBA_FORCE_BUILD']}
        save(OUT/'build-options.json',options);event('resume_build',options=options)
        cmd=[str(ENV/'bin/python'),'-m','pip','install','-v','--no-build-isolation','--no-deps','.']
        subprocess.run(cmd,cwd=REPO,env=env,check=True,timeout=2400)
        subprocess.run([str(ENV/'bin/python'),'-c','import torch,flash_moba,flash_moba_cuda; print(torch.__version__,flash_moba.__file__,flash_moba_cuda.__file__)'],check=True,env=env)
        (OUT/'pip-freeze.txt').write_text(subprocess.check_output([str(ENV/'bin/python'),'-m','pip','freeze'],text=True))
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
        status='complete';error=None
    except Exception as exc:
        status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
    save(OUT/'environment.json',dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
        original_environment_record='results/flashmoba-environment-v0/environment.json',prior_attempt_preserved=True,
        environment=str(ENV),old_environment_modified=False,system_driver_modified=False,scientific_optimizer_updates=0))
    if status!='complete':raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ninja-pid',type=int);main(p.parse_args())
