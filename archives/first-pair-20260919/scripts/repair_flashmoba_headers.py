"""Expose torch-wheel CUDA library headers; compile only the previously missing sort object."""
import hashlib,json,os,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/flashmoba-environment-headers-v2';OUT.mkdir(parents=True,exist_ok=False)
ENV=ROOT/'.venv-flashmoba-v0';TK=ROOT/'toolchains/cuda-12.9.1';REPO=ROOT/'third_party/flash-moba-official-20260915'
start=datetime.now(timezone.utc).isoformat();tick=time.perf_counter();(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
def utc():return datetime.now(timezone.utc).isoformat()
def event(kind,**kw):
    row=dict(event=kind,utc=utc(),elapsed_seconds=time.perf_counter()-tick,**kw)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
try:
    assert json.loads((ROOT/'results/flashmoba-environment-resume-v1/environment.json').read_text())['status']=='incomplete'
    dirs=sorted((ENV/'lib/python3.11/site-packages/nvidia').glob('*/include'))
    assert any((p/'cusparse.h').exists() for p in dirs)
    manifest=[dict(path=str(p),headers=[dict(name=h.name,sha256=hashlib.sha256(h.read_bytes()).hexdigest()) for h in p.glob('*.h')]) for p in dirs]
    (OUT/'header-manifest.json').write_text(json.dumps(manifest,indent=2))
    env=os.environ.copy();env.update(CUDA_HOME=str(TK),MAX_JOBS='6',NVCC_THREADS='2',FLASH_MOBA_CUDA_ARCHS='80',FLASH_MOBA_FORCE_BUILD='TRUE')
    env['PATH']=str(TK/'bin')+':'+str(ENV/'bin')+':'+env['PATH'];env['CPATH']=':'.join(map(str,dirs))
    event('header_search_path',cpath=env['CPATH'],cuda_library_headers_from_installed_torch_wheels=True)
    cmd=[str(ENV/'bin/python'),'-m','pip','install','-v','--no-build-isolation','--no-deps','.']
    event('build_start',command=cmd)
    subprocess.run(cmd,cwd=REPO,env=env,check=True,timeout=900)
    subprocess.run([str(ENV/'bin/python'),'-c','import torch,flash_moba,flash_moba_cuda; print(torch.__version__,flash_moba.__file__,flash_moba_cuda.__file__)'],check=True,env=env)
    (OUT/'pip-freeze.txt').write_text(subprocess.check_output([str(ENV/'bin/python'),'-m','pip','freeze'],text=True))
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    status='complete';error=None
except Exception as exc:
    status='incomplete';error=dict(message=str(exc),traceback=traceback.format_exc());event('failed',error=error)
(OUT/'environment.json').write_text(json.dumps(dict(status=status,error=error,started_utc=start,finished_utc=utc(),wall_seconds=time.perf_counter()-tick,
    environment=str(ENV),prior_attempts_preserved=True,official_source_modified=False,old_environment_modified=False,
    system_driver_modified=False,scientific_optimizer_updates=0),indent=2))
if status!='complete':raise SystemExit(1)
