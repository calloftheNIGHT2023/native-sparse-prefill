"""Reuse compiled official objects to produce a transportable wheel with a digest."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'exports/flashmoba-wheels-v0';out.mkdir(parents=True,exist_ok=False)
envroot=ROOT/'.venv-flashmoba-v0';toolkit=ROOT/'toolchains/cuda-12.9.1';repo=ROOT/'third_party/flash-moba-official-20260915'
p=argparse.ArgumentParser();p.add_argument('--environment-record',type=Path,default=ROOT/'results/flashmoba-environment-v0/environment.json');a=p.parse_args()
assert json.loads(a.environment_record.read_text())['status']=='complete'
env=os.environ.copy();env.update(CUDA_HOME=str(toolkit),MAX_JOBS='4',NVCC_THREADS='2',FLASH_MOBA_CUDA_ARCHS='80',FLASH_MOBA_FORCE_BUILD='TRUE')
env['PATH']=str(toolkit/'bin')+':'+str(envroot/'bin')+':'+env['PATH']
env['CPATH']=':'.join(str(p) for p in sorted((envroot/'lib/python3.11/site-packages/nvidia').glob('*/include')))
cmd=[str(envroot/'bin/python'),'-m','pip','wheel','--no-build-isolation','--no-deps','--wheel-dir',str(out),'.']
start=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
subprocess.run(cmd,cwd=repo,env=env,check=True,timeout=600)
files=list(out.glob('*.whl'));assert len(files)==1
rows=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
subprocess.run(['git','archive','--format=tar.gz','--output='+str(out/'official-source.tar.gz'),'HEAD'],cwd=repo,check=True)
manifest=dict(started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-t,command=cmd,files=rows,
    environment_record=str(a.environment_record),
    torch_abi_target='torch2.8.0+cu128 Python3.11 Linux x86_64 SM80 binary',
    upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    cutlass_commit='a2439551c765c5393aebe557ee75d3a0412d2211',
    source_tar_sha256=hashlib.sha256((out/'official-source.tar.gz').read_bytes()).hexdigest(),
    source_tar_excludes_cutlass_submodule=True,system_driver_included=False)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
