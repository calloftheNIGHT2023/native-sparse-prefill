"""Prepare an isolated upstream-derived smaller-query-tile top-k module."""
from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=R/'experiments/topk-m64-v0';out.mkdir(parents=True,exist_ok=False)
 src=R/'third_party/flash-moba-official-20260915/csrc/flash_topk'
 for p in (src/'src').glob('*.h'):shutil.copyfile(p,out/p.name)
 s=(src/'flash_topk_api.cpp').read_text(encoding='utf-8')
 start=s.index('void run_fused_topk(Fused_topk_params &params, cudaStream_t stream) {');end=s.index('\n\nstd::vector<at::Tensor> moba_fused_topk',start)
 s=s[:start]+'''void run_m64(Fused_topk_params &params, cudaStream_t stream);
void run_fused_topk(Fused_topk_params &params, cudaStream_t stream) {
    TORCH_CHECK(params.is_bf16 && params.d == 64 && params.is_causal,
                "Isolated candidate supports BF16 head64 causal only");
    TORCH_CHECK(params.moba_topk > 32 && params.moba_topk <= 64,
                "Isolated candidate supports K33..64 only");
    run_m64(params, stream);
}
'''+s[end:]
 s=s[:s.index('// Expose init function')]+'''PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moba_fused_topk", &FLASH_TOPK_NAMESPACE::moba_fused_topk);
}
'''
 (out/'api.cpp').write_text(s,encoding='utf-8')
 (out/'kernel.cu').write_text('''// Derived from FlashMoBA commit39d9ac0; upstream headers and licenses retained.
#include "flash_topk_launch_template.h"
namespace FLASH_TOPK_NAMESPACE {
void run_m64(Fused_topk_params &params, cudaStream_t stream) {
    using Traits = Fused_topk_kernel_traits<64,64,64,4,64,true,cutlass::bfloat16_t>;
    static_assert(Traits::kSmemSize == 57344, "Verify candidate resource calculation");
    run_fused_topk<Traits,true>(params,stream);
    run_topk_epilogue<Topk_epilogue_kernel_traits<128,4,64,cutlass::bfloat16_t>>(params,stream);
}
}
''',encoding='utf-8')
 manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),upstream_commit='39d9ac043b271d046a2181a9991e99a26b67bca1',cutlass_commit='a2439551c765c5393aebe557ee75d3a0412d2211',upstream_api_sha256=sha(src/'flash_topk_api.cpp'),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},namespace='nsp_topk_m64_v0',change='Fused selection query tile128->64 forBF16,head64,K33..64,causal. Same scoring/selection/epilogue math; validate before use. Unmodified upstream headers. Separate module, original wheel untouched.',upstream_shared_bytes=106496,candidate_shared_bytes=57344)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='prepared',files=len(manifest['files']))))
if __name__=='__main__':main()
