"""Audit all six task cells, preserving original best-calibration comparison."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'results/cloud-task-factorial-evidence-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified'
 stage=E/'results/task-factorial-stage-v0';controller=read(stage/'result.json');protocol=read(E/'provenance/task-quality-factorial-protocol.json')
 assert controller['status']=='complete' and controller['optimizer_updates']==0
 assert len(controller['runs'])==6 and all(x['returncode']==0 for x in controller['runs'])
 models={};inputs=[]
 for name,c in protocol['conditions'].items():
  p=stage/name/'result.json';d=read(p)
  assert d['status']=='complete' and d['optimizer_updates']==0 and d['condition']==name and d['seed']==2026091560
  assert d['source_training_identity']==c['identity'] and d['weights_mode']==c['weights_mode'] and d['k']==c['task_attention_k']
  assert d['checkpoint_sha256']==c['checkpoint_sha256'] and d['training_result_sha256']==c['training_result_sha256']
  assert d['diagnostic_protocol_sha256']==sha(E/'provenance/task-quality-factorial-protocol.json') and d['protocol_sha256']==sha(E/'data/task-quality-v0/protocol.json')
  assert d['calibration_replay_max_abs_error']<=1e-6 and len(d['predictions'])==384 and d['selected_tasks']==['race_mc']
  rows={(x['item_id'],x['variant']):x for x in d['predictions']};assert len(rows)==384
  assert all(x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']) and np.isfinite(x['choice_logits']).all() for x in rows.values())
  if c['control_result']:
   old={(x['item_id'],x['variant']):x for x in read(E/c['control_result'])['predictions']};assert set(old)==set(rows) and d['control_max_logit_error']<=1e-6
   for i,x in rows.items():assert x['prediction']==old[i]['prediction'] and np.max(np.abs(np.array(x['choice_logits'])-old[i]['choice_logits']))<=1e-6
  models[name]=rows;inputs.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p)))
 names=['base_dense','base_sparse','dense_low','sparse_low','dense_high','sparse_high']
 comparisons=[('sparse_low','dense_low'),('sparse_high','dense_high'),('sparse_high','dense_low'),('dense_low','base_dense'),('dense_high','base_dense'),('sparse_low','base_sparse'),('sparse_high','base_sparse'),('base_sparse','base_dense')]
 labels={'base_dense':'原始底座/D','base_sparse':'原始底座/S','dense_low':'D训 LR0.0003','sparse_low':'S训 LR0.0003','dense_high':'D训 LR0.001','sparse_high':'S训 LR0.001'}
 lines=['# 同学习率与原始底座：任务表现诊断','','这是对旧96道题的事后探索，只有一个种子2026091560。D为密集注意力，S为K16；训练后各使用自己的训练注意力。所有训练断点均256步，底座参照完全移除LoRA适配器，不做优化更新。','','|输入|'+'|'.join(labels[n] for n in names)+'|','|---|'+'---:|'*6]
 results=[];contrasts=[];rng=np.random.default_rng(2026091603);samples=rng.integers(0,96,(10000,96))
 for variant in protocol['variants']:
  ids=sorted(i for i in models[names[0]] if i[1]==variant);assert len(ids)==96
  for name in names:
   assert sorted(i for i in models[name] if i[1]==variant)==ids
   assert all(models[name][i]['gold']==models[names[0]][i]['gold'] for i in ids)
  arrays={n:np.array([models[n][i]['correct'] for i in ids],dtype=float) for n in names}
  row=dict(variant=variant,correct={n:int(arrays[n].sum()) for n in names},accuracy={n:float(arrays[n].mean()) for n in names});results.append(row)
  lines.append('|'+variant+'|'+'|'.join(f"{100*row['accuracy'][n]:.2f}% ({row['correct'][n]}/96)" for n in names)+'|')
  for a,b in comparisons:
   diff=arrays[a]-arrays[b];contrasts.append(dict(variant=variant,comparison=a+' minus '+b,gap_pp=100*float(diff.mean()),descriptive_95pct_interval_pp=(100*np.quantile(diff[samples].mean(1),[.025,.975])).tolist(),corrected=int(((arrays[a]==1)&(arrays[b]==0)).sum()),new_errors=int(((arrays[a]==0)&(arrays[b]==1)).sum())))
 lines+=['','## 完整配对比较','','|输入|A−B|准确率差pp|描述性95%区间pp|改对/新错|','|---|---|---:|---|---:|']
 for c in contrasts:
  lo,hi=c['descriptive_95pct_interval_pp'];lines.append(f"|{c['variant']}|{c['comparison']}|{c['gap_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{c['corrected']}/{c['new_errors']}|")
 lines+=['','## 解释边界','','原最佳校准配置比较仍是sparse_high对dense_low；不能因为dense_high较差就替换密集基线来宣称质量相同。相同学习率对照帮助检查配置混杂，但不是内部机制的唯一解释。原始底座参照能观察本轮适配前后的任务变化；不代表原生稀疏预训练。','','区间按96题配对重采样，事后探索且未经多重比较校正，未显著不等于等价。没有新留出确认，不把最高得分直接当最终方案。长文本使用明确标记TARGET的自建RACE形式。底座移除了LoRA模块，因此其前向时间不能直接解释为训练方法速度差。旧约9.2%训练节时来自原训练轨迹，不能转移给未训练的新方案。','','全部6条件通过原校准NLL重放、数据/模型/源代码/断点哈希核验；dense_low和sparse_high共768条预测与旧记录逐项logit核对。总2304条计分预测，18次热身、24个校准窗口，0优化更新。','','## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in controller['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,predictions=2304,seconds=controller['seconds'],results=results,contrasts=contrasts,inputs=inputs,scope='Post-hoc one-seed task diagnostic, no independent confirmation or original method claim')
 out=R/'results/task-factorial-audit-v0';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
 (R/'docs/task-factorial-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='complete',results=results)))
if __name__=='__main__':main()
