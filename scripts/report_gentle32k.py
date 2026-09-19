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
 a=R/'exports/gentle32k-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
 dest=R/'results/cloud-gentle32k-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  entries=json.loads(t.extractfile('gentle32k-manifest.json').read())['files'];names=[m.name for m in t.getmembers()]
  assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={x['path'] for x in entries}|{'gentle32k-manifest.json'}
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   raw=t.extractfile(m).read();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 p=load(dest/'provenance/gentle32k-protocol.json');assert sha(dest/'provenance/gentle32k-protocol.json')==sha(R/'provenance/gentle32k-protocol.json')
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h
 stage=dest/'results/gentle32k-stage-v0';control=load(stage/'result.json');cfg=load(dest/'data/gentle32k-v0/config.json')
 if control['status']!='complete':
  save(R/'results/gentle32k-audit-v0/result.json',dict(status='failed_evidence_verified',archive=proof,control=control));raise RuntimeError('Stage failed; raw failure evidence verified. No quality conclusion.')
 assert not control['incomplete_jobs'] and control['counts']==dict(scientific_updates=256,diagnostic_updates=6,task_predictions=576)
 assert len(control['jobs'])==6 and all(j['status']=='complete' and j['returncode']==0 for j in control['jobs'])
 import torch
 torch.set_num_threads(4);order=np.random.default_rng(2026091662).permutation(76).tolist();ncheck=0;inits={};groups={};train_s={}
 from amp_recovery_state import lr_factor
 for j in control['jobs']:
  d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['identity']['config_sha256']==sha(dest/'data/gentle32k-v0/config.json')
  assert v['identity']['sources']==cfg['sources_sha256'] and v['identity']['k']==j['k'] and v['identity']['seed']==j['seed']
  assert sha(d/'source.py')==cfg['sources_sha256']['run_gentle32k.py']
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
   step=v['step'];parent=stage/f"seed{j['seed']}"/f'checkpoint-{step}.pt';rep=load(d/'replay-verification.json');assert rep['checkpoint_sha256']==sha(parent) and rep['max_abs_error']<=1e-6
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
 parent_audit=load(R/'results/expanded76-audit-v0/result.json');assert parent_audit['status']=='verified' and sha(R/'results/expanded76-audit-v0/result.json')==p['parent_audit_sha256']
 parent_stage=R/'results/cloud-expanded76-evidence-v0/results/expanded76-stage-v0'
 assert sha(dest/'data/32k-expanded-training-v0/config.json')==p['parent_config_sha256']
 oldcfg=load(dest/'data/32k-expanded-training-v0/config.json')
 for name in ['data_sha256','report_sha256','task_metadata_sha256','task_tokens_sha256','seeds','steps','learning_rates','warmup_steps','lr_schedule_steps','betas','eps','weight_decay','gradient_clip','chunk_size','extension_sha256','model_path']:assert cfg[name]==oldcfg[name],name
 dense_replays=[]
 for seed in p['seeds']:
  old=load(parent_stage/f'k0-seed{seed}/result.json');cp0=torch.load(parent_stage/f'k0-seed{seed}/checkpoint-0.pt',map_location='cpu',weights_only=False)
  oldinit=hashlib.sha256(b''.join(x.numpy().tobytes() for x in cp0['params'].values())).hexdigest()
  assert inits[(p['k'],seed)]==oldinit==old['initial_sha256']
  replay=load(stage/f'dense-replay-{seed}/result.json');proof_replay=load(stage/f'dense-replay-{seed}/replay-verification.json')
  assert replay['status']=='complete' and replay['optimizer_updates_this_process']==0 and replay['identity']==old['identity']
  assert replay['environment']==old['environment']==load(stage/f'seed{seed}/result.json')['environment']
  assert proof_replay['checkpoint_sha256']==sha(parent_stage/f'k0-seed{seed}/checkpoint-128.pt') and proof_replay['max_abs_error']<=1e-6
  cal=next(e for e in old['evaluations'] if e['step']==128)
  assert max(abs(x-y) for x,y in zip(cal['values'],proof_replay['calibration_values']))<=1e-6
  dense_replays.append(proof_replay)
 assert inits[(p['k'],p['seeds'][0])]!=inits[(p['k'],p['seeds'][1])]
 for g in parent_audit['summary']:
  if g['k']==0:groups[(0,g['step'])]=g['rows']
 summary=[]
 for (k,step),rows in sorted(groups.items()):
  assert len(rows)==(2 if step==128 else 1)
  summary.append(dict(k=k,step=step,seeds=len(rows),train_seconds=float(np.mean([x['seconds'] for x in rows])),ppl=float(np.exp(np.mean([x['nll'] for x in rows]))),accuracy={t:float(np.mean([x['accuracies'][t] for x in rows])) for t in rows[0]['accuracies']},context_benefit_nats=float(np.mean([x['context_benefit'] for x in rows])),rows=rows))
 dense=next(x for x in summary if x['k']==0 and x['step']==128);sparse=next(x for x in summary if x['k']==p['k'] and x['step']==128)
 trade=dict(training_time_saving_percent=100*(1-sparse['train_seconds']/dense['train_seconds']),ppl_cost_percent=100*(sparse['ppl']/dense['ppl']-1),task_changes_pp={k:100*(sparse['accuracy'][k]-dense['accuracy'][k]) for k in dense['accuracy']})
 from paired_article_uncertainty import summarize
 sparse_runs=[load(stage/f'report-seed{seed}-step128/task-results.json')['predictions'] for seed in p['seeds']]
 dense_runs=[load(parent_stage/f'report-k0-seed{seed}-step128/task-results.json')['predictions'] for seed in p['seeds']]
 uncertainty=summarize(meta,sparse_runs,dense_runs)
 for t,v in uncertainty['sparse_minus_dense'].items():assert abs(v['mean_pp']-trade['task_changes_pp'][t])<1e-10
 result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,checkpoints=ncheck,control=control,summary=summary,tradeoff=trade,scope=p['scope'],dense_replays=dense_replays,dense_controls_reused=True,counts=control['counts'],paired_article_uncertainty=uncertainty)
 save(R/'results/gentle32k-audit-v0/result.json',result)
 lines=['# K48温和稀疏：两种子固定128步质量试验','','配置由完整更新速度筛选确定：保留更多块，先检验能否减少长文本质量损失。密集128步对照复用既有已核验实验，本轮重放两颗种子的密集校准值，核验模型、环境、数据、学习率以及逐字节初始化一致。密集时间取历史同卡记录，因此只作配合同轮短测速的成本估计，不代表新的随机交错长训练计时。','','|方法|步数|平均训练秒|PPL|短题|无原文|32K|','|---|---:|---:|---:|---:|---:|---:|']
 for x in summary:
  ac=x['accuracy'];label='dense' if x['k']==0 else f"K{x['k']}"
  lines.append(f"|{label}|{x['step']}|{x['train_seconds']:.2f}|{x['ppl']:.4f}|{ac['short']:.2%}|{ac['no_context']:.2%}|{ac['long32768']:.2%}|")
 lines+=['',f"相对复用密集128步训练时间，节时{trade['training_time_saving_percent']:.2f}%，PPL代价{trade['ppl_cost_percent']:.2f}%，长题变化{trade['task_changes_pp']['long32768']:.2f}个百分点。",'', '这是旧64篇RACE开发文章上的探索结果。两颗种子沿用旧初始化，不是两个新独立确认实验。未显著下降不等于证明等价；没有预设非劣界限，不能报告通过非劣检验。此前K32在256篇独立文章的长题落后13.48个百分点，失败记录保留。本轮不重复那256篇，不自动延长训练。', '', f"新增256科学更新、6诊断更新、576任务前向。核验{ncheck}个完整断点、数据游标、学习率、优化器与预测。密集重放0更新。包SHA {proof['sha256']}。", '', '时间是训练循环，包含前反向和优化器；不含加载、校准、测试及租卡闲置。']
 lines+=['', '文章配对95%区间（仅条件于这两颗种子；不能据此宣称等价）：', '', '|指标|差值pp|95%区间pp|','|---|---:|---|']
 for t,v in uncertainty['sparse_minus_dense'].items():lines.append(f"|稀疏减密集 {t}|{v['mean_pp']:+.2f}|{v['ci95_pp']}|")
 for t,v in uncertainty['long_minus_no_context'].items():lines.append(f"|{t} 长题减无原文|{v['mean_pp']:+.2f}|{v['ci95_pp']}|")
 lines+=['','作业时间轴：','','|作业|开始UTC|结束UTC|','|---|---|---|']
 for j in control['jobs']:lines.append(f"|{j['name']}|{j['started_utc']}|{j['finished_utc']}|")
 (R/'docs/gentle32k-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='verified',tradeoff=trade,checkpoints=ncheck)))
if __name__=='__main__':main()
