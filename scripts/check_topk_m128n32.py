"""Candidate genuine-sparsity forward/backward checks; zero optimizer updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,time,hashlib,functools,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch
R=Path(__file__).resolve().parents[1]
def main():
 out=R/'results/topk-m128n32-math-v0';out.mkdir(exist_ok=False);started=datetime.now(timezone.utc).isoformat();tick=time.monotonic();gates=[]
 try:
  import run_flashmoba_realtext_precision as base
  from topk_m128n32_adapter import enable
  from density_tradeoff_math_gate import check
  original=Path(base.flash_moba_cuda.__file__);original_sha=hashlib.sha256(original.read_bytes()).hexdigest();torch.use_deterministic_algorithms(True);base.LOCKED_POOL_CONFIG=(32,4,3)
  base.flash_moba_attn_varlen_func=functools.partial(base.flash_moba_attn_varlen_func,deterministic=True)
  implementation=enable(base)
  for mode in [48,64]:
   gate=check(mode,base);gates.append(gate);print(json.dumps(gate),flush=True)
  assert hashlib.sha256(original.read_bytes()).hexdigest()==original_sha
  result=dict(status='passed',gates=gates,implementation=implementation,original_extension_sha256=original_sha,scientific_updates=0,diagnostic_optimizer_updates=0,task_predictions=0,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick)
 except Exception:result=dict(status='failed',gates=gates,error=traceback.format_exc(),started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-tick)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
