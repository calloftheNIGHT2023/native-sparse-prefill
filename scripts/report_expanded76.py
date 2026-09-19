"""Verify raw archive, full training states and predictions before reporting."""
import json,hashlib,tarfile,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 a=R/'exports/expanded76-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
 dest=R/'results/cloud-expanded76-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  entries=json.loads(t.extractfile('expanded76-manifest.json').read())['files'];names=[m.name for m in t.getmembers()]
  assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={x['path'] for x in entries}|{'expanded76-manifest.json'}
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   raw=t.extractfile(m).read();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 p=load(dest/'provenance/expanded76-protocol.json');assert sha(dest/'provenance/expanded76-protocol.json')==sha(R/'provenance/expanded76-protocol.json')
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h
 stage=dest/'results/expanded76-stage-v0';control=load(stage/'result.json');cfg=load(dest/'data/32k-expanded-training-v0/config.json')
 if control['status']!='complete':
  save(R/'results/expanded76-audit-v0/result.json',dict(status='failed_evidence_verified',archive=proof,control=control));raise RuntimeError('Stage failed; raw failure evidence verified. No quality conclusion.')
 assert not control['incomplete_jobs'] and control['counts']==dict(scientific_updates=512,diagnostic_updates=12,task_predictions=1152)
 assert len(control['jobs'])==12 and all(j['status']=='complete' and j['returncode']==0 for j in control['jobs'])
 import torch
 torch.set_num_threads(4);order=np.random.default_rng(2026091662).permutation(76).tolist();ncheck=0;inits={};groups={};train_s={}
 from amp_recovery_state import lr_factor
 for j in control['jobs']:
  d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['identity']['config_sha256']==sha(dest/'data/32k-expanded-training-v0/config.json')
  assert v['identity']['sources']==cfg['sources_sha256'] and v['identity']['k']==j['k'] and v['identity']['seed']==j['seed']
  assert sha(d/'source.py')==cfg['sources_sha256']['run_expanded76.py']
  if j['phase']=='preflight':assert v['optimizer_updates_this_process']==6 and all(g['passed'] for g in v['gates']) and len(v['gates'])==6
  if j['phase']=='train':
   assert len(v['trace'])==128 and v['optimizer_updates_this_process']==128
   assert [x['step'] for x in v['trace']]==list(range(1,129))
   for x in v['trace']:
    assert x['window_index']==order[(x['step']-1)%76] and math.isfinite(x['train_nll']) and x['seconds']>0
    assert abs(x['lr']-.001*lr_factor(x['step']-1,4,64))<1e-12
   assert abs(sum(x['seconds'] for x in v['trace'])-v['training_seconds'])<1e-8
   train_s[(j['k'],j['seed'])]=v['training_seconds']
   assert [e['step'] for e in v['evaluations']]==[0,32,64,96,128]
   for e in v['evaluations']:assert e['split']=='calibration' and len(e['values'])==4 and abs(e['mean_nll']-np.mean(e['values']))<1e-10
  for f in d.glob('checkpoint-*.pt'):
   c=torch.load(f,map_location='cpu',weights_only=False);s=c['step'];assert s==c['data_cursor']==c['scheduler']['last_epoch'] and c['identity']==v['identity']
   assert sum(t.numel() for t in c['params'].values())==1081344 and all(torch.isfinite(t).all() for t in c['params'].values())
   assert set(c['rng'])=={'python','numpy','torch','cuda'} and len(c['rng']['cuda'])==1
   assert len(c['extra']['rows'])==s and c['extra']['rows']==v['trace'][:s]
   assert abs(c['optimizer']['param_groups'][0]['lr']-.001*lr_factor(s,4,64))<1e-12
   if s:
    assert len(c['optimizer']['state'])==192
    for z in c['optimizer']['state'].values():assert float(z['step'])==s and torch.isfinite(z['exp_avg']).all() and torch.isfinite(z['exp_avg_sq']).all()
   else:
    assert not c['optimizer']['state']
    digest=hashlib.sha256(b''.join(x.numpy().tobytes() for x in c['params'].values())).hexdigest();assert digest==v['initial_sha256']
    inits[(j['k'],j['seed'])]=digest
   ncheck+=1
  if j['phase']=='report':
   assert v['optimizer_updates_this_process']==0
   step=v['step'];parent=stage/f"k{j['k']}-seed{j['seed']}"/f'checkpoint-{step}.pt';rep=load(d/'replay-verification.json');assert rep['checkpoint_sha256']==sha(parent) and rep['max_abs_error']<=1e-6
   pred=load(d/'task-results.json');meta=load(dest/'data/32k-expanded-training-v0/tasks.json');assert len(pred['predictions'])==len(meta)==192
   assert pred['checkpoint_sha256']==sha(parent) and pred['task_metadata_sha256']==cfg['task_metadata_sha256'] and pred['task_tokens_sha256']==cfg['task_tokens_sha256']
   assert pred['predictions']==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
   acc={}
   for x,y in zip(pred['predictions'],meta):
    assert all(x[k]==y[k] for k in ['item_id','variant','gold']) and np.isfinite(x['choice_logits']).all()
    assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
    acc.setdefault(x['variant'],[]).append(x['correct'])
   ev=next(x for x in v['evaluations'] if x['split']=='report');assert len(ev['values'])==9 and abs(ev['mean_nll']-np.mean(ev['values']))<1e-10
   context=load(d/'context-diagnostic.json');assert np.allclose(np.array(context['short_context_tail_nll'])-context['full_context_tail_nll'],context['benefit_nats'],rtol=0,atol=1e-12)
   groups.setdefault((j['k'],step),[]).append(dict(seed=j['seed'],nll=ev['mean_nll'],values=ev['values'],accuracies={k:float(np.mean(z)) for k,z in acc.items()},seconds=train_s[(j['k'],j['seed'])] if step else 0,context_benefit=float(np.mean(context['benefit_nats']))))
 for seed in p['seeds']:assert inits[(0,seed)]==inits[(32,seed)]
 assert inits[(0,p['seeds'][0])]!=inits[(0,p['seeds'][1])]
 summary=[]
 for (k,step),rows in sorted(groups.items()):
  assert len(rows)==(2 if step==128 else 1)
  summary.append(dict(k=k,step=step,seeds=len(rows),train_seconds=float(np.mean([x['seconds'] for x in rows])),ppl=float(np.exp(np.mean([x['nll'] for x in rows]))),accuracy={t:float(np.mean([x['accuracies'][t] for x in rows])) for t in rows[0]['accuracies']},context_benefit_nats=float(np.mean([x['context_benefit'] for x in rows])),rows=rows))
 dense=next(x for x in summary if x['k']==0 and x['step']==128);sparse=next(x for x in summary if x['k']==32 and x['step']==128)
 trade=dict(training_time_saving_percent=100*(1-sparse['train_seconds']/dense['train_seconds']),ppl_cost_percent=100*(sparse['ppl']/dense['ppl']-1),task_changes_pp={k:100*(sparse['accuracy'][k]-dense['accuracy'][k]) for k in dense['accuracy']})
 result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,checkpoints=ncheck,control=control,summary=summary,tradeoff=trade,scope=p['scope'])
 save(R/'results/expanded76-audit-v0/result.json',result)
 lines=['# 76窗口32K训练：固定128步成本与质量','','四条全新优化轨迹从同一预训练模型开始；沿用原两颗初始化种子，每颗种子下密集/稀疏初始LoRA参数逐字节一致。每种方法128更新，学习率原64步计划后保持0.0001。新增44窗口，同一语料；两个阶段的样本覆盖与顺序都改变，历史对比不是只改变单一因素的随机试验。','', '|方法|步数|平均训练秒|PPL|短题|无原文|32K|','|---|---:|---:|---:|---:|---:|---:|']
 for x in summary:
  ac=x['accuracy'];lines.append(f"|{'dense' if x['k']==0 else 'K32'}|{x['step']}|{x['train_seconds']:.2f}|{x['ppl']:.4f}|{ac['short']:.2%}|{ac['no_context']:.2%}|{ac['long32768']:.2%}|")
 lines+=['',f"同128步节时{trade['training_time_saving_percent']:.2f}%，PPL代价{trade['ppl_cost_percent']:.2f}%，长题变化{trade['task_changes_pp']['long32768']:.2f}个百分点。",'','时间是训练循环，包含前反向和更新；不包含加载/校准/测试/租卡空闲。总控制器时长另见机器结果。本轮不要求稀疏反超，但不把未显著下降当作已证明等价。RACE64和WikiText测试都已被前阶段使用，属于探索性复用；更强的论文结论需要另行冻结独立确认集。','',f"原始包{proof['files']}文件，完整检查点{ncheck}个；512科学更新、12诊断更新、1152任务计分前向。全部断点状态、学习率、数据游标与预测已核验。包SHA {proof['sha256']}。",'', '## 时间轴','','时间UTC。','','|作业|开始|结束|','|---|---|---|']
 for j in control['jobs']:lines.append(f"|{j['name']}|{j['started_utc']}|{j['finished_utc']}|")
 (R/'docs/expanded76-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='verified',tradeoff=trade,checkpoints=ncheck)))
if __name__=='__main__':main()
