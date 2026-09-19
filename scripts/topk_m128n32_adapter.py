"""Process-local top-k override; original attention and K<=32 remain original."""
import importlib.util,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def enable(base):
 p=R/'experiments/topk-m128n32-v0/build-result.json';v=json.loads(p.read_text());assert v['status']=='built'
 binary=Path(v['module_path']);assert hashlib.sha256(binary.read_bytes()).hexdigest()==v['module_sha256']
 spec=importlib.util.spec_from_file_location('nsp_topk_m128n32_v0',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 original=base.flash_moba_cuda.moba_fused_topk
 def dispatch(q,km,cq,ck,cm,nq,nk,topk,block,causal):
  fn=module.moba_fused_topk if topk>32 else original
  return fn(q,km,cq,ck,cm,nq,nk,topk,block,causal)
 base.flash_moba_cuda.moba_fused_topk=dispatch
 return dict(module_path=str(binary),sha256=v['module_sha256'],manifest_sha256=v['manifest_sha256'],scope='Only this Python process uses candidate K>32 selection; original wheel file is unchanged.')
