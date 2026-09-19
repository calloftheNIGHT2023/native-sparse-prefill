"""Audit privileged route diagnostic, gate and all examples without selection."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'results/cloud-task-route-chain-evidence-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified'
 protocol=read(E/'provenance/task-quality-route-chain-protocol.json');stage=E/'results/task-route-chain-stage-v0';controller=read(stage/'result.json')
 assert controller['status']=='complete' and controller['optimizer_updates']==0 and all(x['returncode']==0 for x in controller['runs'])
 assert read(E/'results/task-route-preflight-v1/result.json')['status']=='passed'
 assert sha(E/'scripts/task_route_intervention.py')==protocol['route_source_sha256']
 parts=['screen','remaining'] if controller['gate']['passed'] else ['screen'];assert len(controller['runs'])==len(parts)*2
 placements={x['item_id']:x for x in protocol['rows']};summary=[];score_screen=[];count=0
 lines=['# 固定K16预算：问题阶段原文路由诊断','','目标组使用事先知道的目标段落位置；无关对照强制等量无关块。仅在问题/选项/Answer后缀query上干预所有24层，不增加每query/head的选块数，不使用正确答案来选块。结果为旧题上的特权诊断，不是可部署方法或独立确认。','','|阶段|种子|原路由正确|目标段正确|无关块正确|题数|','|---|---|---:|---:|---:|---:|']
 for part in parts:
  ids=protocol['screen_ids' if part=='screen' else 'remaining_ids'];models=[]
  for seedindex in [0,1]:
   d=read(stage/f'{part}-seed{2026091560+seedindex}'/'result.json')
   assert d['status']=='complete' and d['optimizer_updates']==0 and d['k']==16 and d['lr']==.001 and d['seed']==2026091560+seedindex
   assert d['calibration_replay_max_abs_error']<=1e-6 and d['control_max_logit_error']<=1e-6
   assert d['route_protocol_sha256']==sha(E/'provenance/task-quality-route-chain-protocol.json')
   assert d['task_protocol_sha256']==sha(R/'data/task-quality-v0/protocol.json')
   old=read(R/f'results/cloud-task-quality-evidence-v0/results/task-quality-stage-v0/sparse{seedindex}/result.json')
   assert d['checkpoint_sha256']==old['checkpoint_sha256'] and d['training_result_sha256']==old['training_result_sha256']
   old={x['item_id']:x for x in old['predictions'] if x['variant']=='long16384'}
   assert len(d['predictions'])==len(ids)*3;count+=len(d['predictions']);mapping={}
   for c in protocol['conditions']:
    xs={x['item_id']:x for x in d['predictions'] if x['condition']==c};assert set(xs)==set(ids)
    for i,x in xs.items():
     assert x['gold']==old[i]['gold'] and x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']) and len(x['layer_telemetry'])==24
     p=placements[i];assert x['suffix_start']==p['suffix_start'] and x['target_blocks']==p['target_blocks']
     assert x['forced_blocks']==([] if c=='baseline' else p['target_blocks' if c=='target' else 'sham_blocks'])
     if c=='baseline':assert x['prediction']==old[i]['prediction'] and np.max(np.abs(np.array(x['choice_logits'])-old[i]['choice_logits']))<=1e-6
    mapping[c]=xs
   scores={c:sum(x['correct'] for x in mapping[c].values()) for c in protocol['conditions']}
   lines.append(f"|{part}|{seedindex}|{scores['baseline']}|{scores['target']}|{scores['sham']}|{len(ids)}|")
   if part=='screen':score_screen.append(scores)
   models.append(mapping)
  arrays={c:np.array([[models[s][c][i]['correct'] for i in ids] for s in [0,1]],dtype=float) for c in protocol['conditions']}
  contrasts=[];rng=np.random.default_rng(2026091611);samples=rng.integers(0,len(ids),(10000,len(ids)))
  for a,b in [('target','baseline'),('target','sham'),('sham','baseline')]:
   diff=(arrays[a]-arrays[b]).mean(0);contrasts.append(dict(comparison=a+' minus '+b,gap_pp=100*float(diff.mean()),descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist()))
  coverage={c:float(np.mean([v['target_block_recall_at_final_query'] for m in models for x in m[c].values() for v in x['layer_telemetry']])) for c in protocol['conditions']}
  summary.append(dict(part=part,questions=len(ids),accuracy={c:float(ar.mean()) for c,ar in arrays.items()},contrasts=contrasts,mean_pre_intervention_target_recall=coverage))
 g=protocol['gate'];b=[s['target']-s['baseline'] for s in score_screen];c=[s['target']-s['sham'] for s in score_screen]
 passed=min(b)>=g['each_seed_target_minus_baseline_min_correct'] and np.mean(b)>=g['mean_target_minus_baseline_min_correct'] and min(c)>=g['each_seed_target_minus_sham_min_correct'] and np.mean(c)>=g['mean_target_minus_sham_min_correct']
 assert bool(passed)==controller['gate']['passed'] and score_screen==controller['gate']['scores']
 lines+=['',f"E1a冻结门槛：{'通过，已继续E1b剩余72题' if passed else '未通过，没有继续E1b或选择器训练'}。目标对基线两种子差值{b}；目标对无关对照差值{c}。",'','## 配对描述','','|阶段|比较|差值pp|描述性95%区间pp|','|---|---|---:|---|']
 for row in summary:
  for x in row['contrasts']:
   lo,hi=x['descriptive_95pct_interval_pp'];lines.append(f"|{row['part']}|{x['comparison']}|{x['gap_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
 lines+=['','原目标块覆盖率仅是按层/head等权平均的标记段落块召回，不能把它等同于因果有效信息。目标组的覆盖统计是在每一层实施替换前记录，且该层表示已经受之前各层干预影响，不是冻结Q/K上的纯路由指标。无关组的块数与目标相同，但新替换块数可能因原覆盖不同而不同。','','E0重建输出在512/2048/4096长度完全一致；强制路由独立FP32参照通过。正式基线路由在16K逐题与旧logit重放核验，强制路由每次检查无重复、预算相同、当前块保留和无未来块；所有条件保持参数不变。目标特权干预失败不能排除更早表示传播受损；成功也不能直接推断低成本代理能学会或带来训练节时。','','所有区间事后描述、旧题、两种子共享数据顺序，不构成等价、泛化或原创贡献证明。E1b本身仍是旧题开发复核。后续可部署方法与论文贡献必须独立设计并查重。','','## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=count,screen_gate_passed=bool(passed),summary=summary,seconds=controller['seconds'],scope='Privileged question-suffix route diagnostic, not deployable method or independent confirmation')
 out=R/'results/task-route-chain-audit-v0';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 (R/'docs/task-route-chain-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
