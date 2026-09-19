"""Audit every fixed layer condition; exploratory paired question bootstrap."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
E=R/'results/cloud-task-layers-evidence-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified'
 stage=E/'results/task-layers-stage-v0';controller=read(stage/'result.json')
 assert controller['status']=='complete' and controller['optimizer_updates']==0
 assert len(controller['runs'])==2 and all(x['returncode']==0 for x in controller['runs'])
 protocol=read(E/'provenance/task-quality-layer-protocol.json');conditions=list(protocol['conditions']);models=[];dense=[]
 for seed in [0,1]:
  d=read(stage/f'seed{2026091560+seed}/result.json');old=read(E/f'results/task-quality-stage-v0/sparse{seed}/result.json');sd=read(E/f'results/task-runtime-stage-v0/sparse_weights_dense_eval{seed}/result.json')
  assert d['status']=='complete' and d['optimizer_updates']==0 and d['seed']==2026091560+seed
  assert d['calibration_replay_max_abs_error']<=1e-6 and d['control_max_logit_abs_error']<=1e-6
  assert d['checkpoint_sha256']==old['checkpoint_sha256']==sd['checkpoint_sha256']
  assert d['training_result_sha256']==old['training_result_sha256']==sd['training_result_sha256']
  assert d['layer_protocol_sha256']==sha(E/'provenance/task-quality-layer-protocol.json')
  assert d['task_protocol_sha256']==sha(E/'data/task-quality-v0/protocol.json')
  assert len(d['predictions'])==576
  mapping={}
  for c in conditions:
   rows=[x for x in d['predictions'] if x['condition']==c];assert len(rows)==96
   mapping[c]={x['item_id']:x for x in rows};assert len(mapping[c])==96
   for x in rows:
    assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
    assert np.isfinite(x['choice_logits']).all() and x['prefill_seconds']>0
  for c,control in [('all_sparse',old),('all_dense',sd)]:
   control={x['item_id']:x for x in control['predictions'] if x['variant']=='long16384'}
   assert set(control)==set(mapping[c])
   for i,x in mapping[c].items():
    y=control[i];assert x['gold']==y['gold'] and x['prediction']==y['prediction']
    assert np.max(np.abs(np.array(x['choice_logits'])-y['choice_logits']))<=1e-6
  models.append(mapping)
  dd=read(E/f'results/task-quality-stage-v0/dense{seed}/result.json')
  dense.append({x['item_id']:x for x in dd['predictions'] if x['variant']=='long16384'})
 ids=sorted(models[0]['all_sparse']);arrays={};times={}
 for c in conditions:
  for seed in [0,1]:
   assert sorted(models[seed][c])==ids==sorted(dense[seed])
   assert all(models[seed][c][i]['gold']==dense[seed][i]['gold'] for i in ids)
  arrays[c]=np.array([[models[s][c][i]['correct'] for i in ids] for s in [0,1]],dtype=float)
  times[c]=np.array([[models[s][c][i]['prefill_seconds'] for i in ids] for s in [0,1]])
 dd=np.array([[dense[s][i]['correct'] for i in ids] for s in [0,1]],dtype=float)
 rng=np.random.default_rng(2026091602);samples=rng.integers(0,96,(10000,96))
 rows=[]
 for c in conditions:
  diff=(arrays[c]-arrays['all_sparse']).mean(0);dddiff=(arrays[c]-dd).mean(0)
  rows.append(dict(corrected_per_seed=((arrays[c]==1)&(arrays['all_sparse']==0)).sum(1).astype(int).tolist(),new_errors_per_seed=((arrays[c]==0)&(arrays['all_sparse']==1)).sum(1).astype(int).tolist(),condition=c,dense_layers=protocol['conditions'][c],correct_per_seed=arrays[c].sum(1).astype(int).tolist(),accuracy=float(arrays[c].mean()),gain_over_all_sparse_pp=100*float(diff.mean()),descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist(),gap_to_dense_trained_pp=100*float(dddiff.mean()),gap_to_dense_trained_interval_pp=(100*np.quantile(dddiff[samples].mean(1),[.025,.975])).tolist(),median_paired_prefill_ratio_to_all_sparse=float(np.median(times[c]/times['all_sparse']))))
 lines=['# 分层切回密集注意力：固定稀疏权重的16K诊断','','同一批已观察的96道阅读理解题，两个固定稀疏训练断点；不更新参数。24层按连续6层分四组，逐组切回密集注意力，其余保持K16。全稀疏、全密集为端点重放对照。所有组均完整报告。','','|密集层（编号从0开始）|种子0正确/96|种子1正确/96|平均准确率|比全稀疏变化pp|描述性95%区间pp|','|---|---:|---:|---:|---:|---|']
 for x in rows:
  lo,hi=x['descriptive_95pct_interval_pp'];a,b=x['correct_per_seed'];lines.append(f"|{x['condition']} {x['dense_layers']}|{a}|{b}|{100*x['accuracy']:.2f}%|{x['gain_over_all_sparse_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
 lines += ['', '各条件相对全稀疏的改对题数 / 新错题数（按两个种子分别列出）：']
 for x in rows:lines.append(f"- {x['condition']}: 改对{x['corrected_per_seed']}，新错{x['new_errors_per_seed']}。配对prefill时间比（相对全稀疏）{x['median_paired_prefill_ratio_to_all_sparse']:.3f}；只作本轮描述。")
 lines += ['',f'真正密集训练且密集使用的参照平均为 {100*dd.mean():.2f}%，与本轮权重不同。全密集使用的稀疏训练模型不能代替这个参照。', '', '## 审计与解释范围','','两个模型断点、训练结果和协议哈希一致；原校准NLL重放误差≤1e-6。两个端点的全部题目选项logit和预测与原记录逐一核对，最大误差≤1e-6。共1152次计分前向，另有24次热身、8次校准窗口前向，0个优化更新。','','按题配对重采样并共同保留两个种子。区间为事后探索性区间，未做多组选择校正，不能作为独立确认或等价证明。同题两种子不是192道独立题。位置切片各用不同题，难度与位置混杂，不能推断位置因果效应。','','本轮只改变推理路径，没有训练混合注意力模型。旧全稀疏训练约9.2%的节时不能转移给混合训练，也不能等同于prefill加速。分组干预不能排除跨组非线性互作；某组切换有效也不能单独证明该组是唯一原因。','','## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=1152,results=rows,dense_trained_accuracy=float(dd.mean()),seconds=controller['seconds'],scope='Post-hoc layer intervention; not independent confirmation or mixed-attention training',controls_verified=True)
 out=R/'results/task-layers-audit-v0';out.mkdir(exist_ok=False)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 (R/'docs/task-layers-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()
