"""Recompute the frozen two-round cost screen from every timing sample."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'results/cloud-task-density-cost-evidence-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified'
d=read(E/'results/task-density-cost-stage-v1/benchmark/result.json');c=read(E/'results/task-density-cost-stage-v1/result.json')
assert c['status']==d['status']=='complete' and d['weights_unchanged'] and d['optimizer_updates']==0 and d['calibration_replay_max_abs_error']<=1e-6
protocol=E/'provenance/task-quality-density-cost-v1-protocol.json';assert hashlib.sha256(protocol.read_bytes()).hexdigest()==d['protocol_sha256']
assert len(d['rows'])==6;rows={(x['round'],x['k']):x for x in d['rows']};assert set(rows)=={(i,k) for i in [0,1] for k in [0,16,32]}
for x in rows.values():
 assert len(x['times'])==len(x['losses'])==len(x['gradient_norms'])==5
 assert np.isfinite(x['times']).all() and min(x['times'])>0 and np.isfinite(x['losses']).all() and np.isfinite(x['gradient_norms']).all()
 assert abs(np.median(x['times'])-x['median_seconds'])<1e-12
results=[]
for k in [0,16,32]:
 ratios=[rows[i,k]['median_seconds']/rows[i,0]['median_seconds'] for i in [0,1]]
 results.append(dict(k=k,median_seconds=[rows[i,k]['median_seconds'] for i in [0,1]],ratios=ratios,passes_frozen_cost_gate=all(v<=.95 for v in ratios)))
 for x in d['candidates']:
  if x['k']==k:assert x['ratios']==ratios and x['both_rounds_at_least_5pct_faster']==all(v<=.95 for v in ratios)
lines=['# 多保留注意力块：成本筛查','','同一已有密集适配断点，两段16K训练输入，K0/16/32按正反顺序测试，每格2次热身、5次计时。计入输入搬运、完整模型前向、分块语言模型损失反向及梯度裁剪；没有optimizer.step，0优化更新。权重前后摘要一致。','','|保留块数（K0=密集）|第一轮秒|第二轮秒|相对密集耗时比（两轮）|两轮均节时≥5%|','|---|---:|---:|---|---|']
for x in results:lines.append(f"|{x['k']}|{x['median_seconds'][0]:.4f}|{x['median_seconds'][1]:.4f}|{x['ratios'][0]:.3f}, {x['ratios'][1]:.3f}|{'是' if x['passes_frozen_cost_gate'] else '否'}|")
lines+=['','v0在K64第一次热身时于moba_fused_topk报CUDA invalid argument，完整失败日志另存。v1冻结为K0/16/32双轮复核；K64没有可用成本结论。','','K32通过此门槛才考虑下一步训练。门槛是成本筛查，不代表准确率恢复、完整训练节时或方法有效。只有两个窗口、当前0.5B LoRA和48GB RTX6000 Ada实现；不能外推到大模型或其他内核。K表示最多选中的块数，不是统一百分比稀疏度，因果前缀会改变实际覆盖比例。',f"\nUTC {c['started_utc']} — {c['finished_utc']}；控制器{c['seconds']:.2f}秒；30计时样本、12热身，另4校准窗口前向。"]
out=R/'results/task-density-cost-audit-v1';out.mkdir(exist_ok=False)
result=dict(status='complete',optimizer_updates=0,results=results,eligible_new_candidates=[x['k'] for x in results if x['k'] in [32] and x['passes_frozen_cost_gate']],timed_samples=30,warmups=12,seconds=c['seconds'])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());(R/'docs/task-density-cost-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result))
