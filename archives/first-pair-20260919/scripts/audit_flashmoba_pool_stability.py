"""Preserve failed replay and quantify controlled, full-model config changes."""
import json,itertools,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
out=R/'flashmoba-pool-stability-audit-v0';out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
old=np.load(R/'flashmoba-qwen-long-precision-v0/per-token-results.npz')
replay=np.load(R/'flashmoba-qwen-long-replay-v0/per-token-results.npz')
checks=[dict(array=k,bitwise_equal=bool(np.array_equal(old[k],replay[k])),values=int(replay[k].size),
             changed_values=int((old[k]!=replay[k]).sum()),max_abs=float(np.max(np.abs(old[k]-replay[k])))) for k in replay.files]
save(R/'flashmoba-qwen-long-replay-v0/replay-verification.json',dict(status='passed' if all(x['bitwise_equal'] for x in checks) else 'failed',checks=checks,
       earlier_process_autotune_choices_not_recorded=True,comparison_changed_both_process_and_sparse_execution_order=True))
tables={bn:np.load(R/f'flashmoba-fixed-pool-bn{bn}-v0/per-token-results.npz') for bn in [32,64,128]}
fixed=[];summaries=[];hashes={}
for bn in tables:
    d=R/f'flashmoba-fixed-pool-bn{bn}-v0';record=json.loads((d/'evaluation.json').read_text())
    assert record['status']=='complete' and len(record['conditions'])==80
    manifest=json.loads((d/'manifest.json').read_text());assert sha(d/'source.py')==manifest['source_sha256']
    hashes[str(d.relative_to(ROOT))]=sha(d/'evaluation.json')
    summaries.append(dict(kBlockN=bn,summaries=record['summaries']))
for left,right in itertools.combinations(tables,2):
    for mode in ['dense','official_k2','fp32_k2','official_k4','fp32_k4']:
        a=np.concatenate([tables[left][f'n8192_c{c}_{mode}_argmax'] for c in range(16)])
        b=np.concatenate([tables[right][f'n8192_c{c}_{mode}_argmax'] for c in range(16)])
        ln=np.concatenate([tables[left][f'n8192_c{c}_{mode}_nll'] for c in range(16)])
        rn=np.concatenate([tables[right][f'n8192_c{c}_{mode}_nll'] for c in range(16)])
        fixed.append(dict(left=left,right=right,mode=mode,tokens=len(a),argmax_changed=int((a!=b).sum()),
            argmax_changed_fraction=float(np.mean(a!=b)),nll_bitwise_equal=bool(np.array_equal(ln,rn)),
            left_mean_nll=float(ln.mean()),right_mean_nll=float(rn.mean()),max_abs_token_nll_delta=float(np.max(np.abs(ln-rn)))))
raw=json.loads((R/'flashmoba-pool-autotune-v0/diagnostic.json').read_text());assert raw['status']=='complete'
diag=[]
for dtype in ['torch.bfloat16','torch.float32']:
    rows=[x for x in raw['conditions'] if x['dtype']==dtype]
    diag.append(dict(dtype=dtype,conditions=len(rows),max_changed_routes=max(x['route_rows_changed_from_first'] for x in rows),
                     route_rows=rows[0]['total_route_rows'],max_relative_output_delta=max(x['output_relative_difference'] for x in rows)))
save(out/'audit.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
    replay_status='failed',replay=checks,fixed_config_comparisons=fixed,fixed_config_summaries=summaries,
    single_real_qkv=diag,verified_sources=hashes,scientific_optimizer_updates=0,
    main_claim='Fixed real inputs and weights can change sparse routes/predictions when BF16 reduction configuration changes. Do not claim a new general floating-point mechanism or universal FP32 bitwise invariance.',
    earlier_average_precision_quality_claim_superseded=True))
(out/'source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(single_real_qkv=diag,comparisons=fixed)))
