"""Build only an isolated candidate top-k module, without reinstalling FlashMoBA."""
import os
os.environ['CUDA_HOME']='/usr/local/cuda'
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
os.environ['MAX_JOBS']='2'
import json,hashlib,tarfile,urllib.request,shutil,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 started=datetime.now(timezone.utc).isoformat();tick=time.monotonic();src=R/'experiments/topk-m64-v0';m=json.loads((src/'manifest.json').read_text())
 for n,h in m['files'].items():assert sha(src/n)==h,n
 deps=src/'deps';deps.mkdir(exist_ok=True);commit=m['cutlass_commit'];archive=deps/f'cutlass-{commit}.tar.gz';include=deps/f'cutlass-{commit}/include'
 if not archive.exists():
  with urllib.request.urlopen(f'https://codeload.github.com/NVIDIA/cutlass/tar.gz/{commit}',timeout=120) as response,archive.open('wb') as f:shutil.copyfileobj(response,f)
 if not include.exists():
  with tarfile.open(archive) as t:
   selected=[x for x in t.getmembers() if x.name.startswith(f'cutlass-{commit}/include/') or x.name==f'cutlass-{commit}/LICENSE.txt']
   assert selected and all(not Path(x.name).is_absolute() and '..' not in Path(x.name).parts and (x.isfile() or x.isdir()) for x in selected)
   t.extractall(deps,members=selected,filter='data')
 assert (include/'cutlass/cutlass.h').exists()
 from torch.utils.cpp_extension import load
 build=src/'build';build.mkdir(exist_ok=True)
 ns='-DFLASH_TOPK_NAMESPACE=nsp_topk_m64_v0'
 module=load(name='nsp_topk_m64_v0',sources=[str(src/'api.cpp'),str(src/'kernel.cu')],extra_include_paths=[str(src),str(include)],extra_cflags=['-O2',ns],extra_cuda_cflags=['-O2',ns,'--expt-relaxed-constexpr','--expt-extended-lambda','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__'],build_directory=str(build),verbose=True)
 result=dict(status='built',started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick,module_path=module.__file__,module_sha256=sha(Path(module.__file__)),manifest_sha256=sha(src/'manifest.json'),cutlass_archive_sha256=sha(archive),nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),scope='Build only; math/runtime not yet validated. Does not replace original wheel.')
 (src/'build-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
