"""Independently audit archived training, selection, checkpoints and all task predictions."""
import json,hashlib,tarfile,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
torch.set_num_threads(4)
import pyarrow.parquet as pq
from amp_recovery_selection import select_lrs
from amp_recovery_state import lr_factor
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 archive=R/'exports/32k-adaptation-evidence-v0.tar.gz';proof=read(archive.with_suffix('.json'));assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-32k-adaptation-evidence-v0';dest.mkdir(exist_ok=False)
 with tarfile.open(archive) as t:
  members=t.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
  manifest=json.load(t.extractfile('32k-adaptation-evidence-manifest.json'));expected={x['path']:x for x in manifest['files']}
  assert set(names)==set(expected)|{'32k-adaptation-evidence-manifest.json'}
  for m in members:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   p=(dest/m.name).resolve();assert dest.resolve() in p.parents;raw=t.extractfile(m).read()
   if m.name in expected:
    e=expected[m.name];assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 verification=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),sha256=proof['sha256'],files=len(expected));dump(dest/'LOCAL-VERIFICATION.json',verification)
 pp=dest/'provenance/32k-adaptation-protocol.json';protocol=read(pp);assert sha(pp)==sha(R/'provenance/32k-adaptation-protocol.json')
 for name,h in protocol['source_sha256'].items():assert sha(dest/name)==h
 data=dest/'data/32k-adaptation-v0';cfg=read(data/'config.json');assert sha(data/'config.json')==protocol['config_sha256']==sha(R/'data/32k-adaptation-v0/config.json')
 for name,key in [('train-calibration.npz','data_sha256'),('report.npz','report_sha256'),('tasks.json','task_metadata_sha256'),('tasks.npz','task_tokens_sha256')]:assert sha(data/name)==cfg[key]
 assert sha(data/'test-source.parquet')=='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
 stage=dest/'results/32k-adaptation-stage-v0';controller=read(stage/'result.json');assert controller['status']=='complete' and not controller['preflight_only']
 assert len(controller['runs'])==14 and all(x['returncode']==0 for x in controller['runs'])
 recovery=read(stage/'archive-recovery.json');assert recovery['all_jobs_completed_before_repair'] and recovery['optimizer_updates']==0
 for prev,nxt in zip(controller['runs'],controller['runs'][1:]):assert prev['finished_utc']<=nxt['started_utc']
 selection=read(stage/'selection-lock.json');assert selection['config_sha256']==sha(data/'config.json') and selection['sources']==cfg['sources_sha256']
 order=np.random.default_rng(2026091662).permutation(32).tolist();training=[];diagnostics=[];checkpoint_count=0
 for job in controller['runs']:
  path=stage/job['name'];r=read(path/'result.json');assert r['status']=='complete' and r['identity']['sources']==cfg['sources_sha256']
  assert r['identity']['config_sha256']==sha(data/'config.json')
  assert r['started_utc']>=job['started_utc'] and r['finished_utc']<=job['finished_utc']
  if r['phase']=='train':
   assert r['step']==r['optimizer_updates_this_process']==len(r['trace'])==64
   assert all(x['split']=='calibration' for x in r['evaluations'])
   assert [x['step'] for x in r['evaluations']]==[0,16,32,64]
   for i,x in enumerate(r['trace']):
    assert x['step']==i+1 and x['window_index']==order[i%32] and x['seconds']>0
    assert math.isfinite(x['train_nll']) and math.isfinite(x['gradient_norm'])
    assert abs(x['lr']-r['identity']['lr']*lr_factor(i,4,64))<1e-12
   assert abs(sum(x['seconds'] for x in r['trace'])-r['training_seconds'])<1e-9
   for step in cfg['checkpoint_steps']:
    cp=path/f'checkpoint-{step}.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
    assert state['step']==state['data_cursor']==step and state['identity']==r['identity']
    assert state['extra']['rows']==r['trace'][:step]
    assert state['extra']['initial_sha256']==r['initial_sha256']
    assert set(state['rng'])=={'python','numpy','torch','cuda'} and state['rng']['cuda'] is not None
    assert sum(p.numel() for p in state['params'].values())==1081344 and all(torch.isfinite(p).all() for p in state['params'].values())
    assert state['scheduler']['last_epoch']==step
    if step==0:assert not state['optimizer']['state'] and all(torch.count_nonzero(p)==0 for n,p in state['params'].items() if n.endswith('.B'))
    else:assert state['optimizer']['state']
    del state;checkpoint_count+=1
   index=read(path/'checkpoint-index.json');assert index['step']==64 and index['sha256']==sha(path/'checkpoint-64.pt')
   events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()];assert len([e for e in events if e['event']=='step'])==64
   training.append(dict(name=job['name'],result=r))
  elif r['phase']=='preflight':
   assert r['optimizer_updates_this_process']==6 and r['step']==4 and len(r['gates'])==5 and all(x['passed'] for x in r['gates'])
   events=[json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()];assert len([e for e in events if e['event']=='step'])==6
   diagnostics.append(r)
  else:assert r['phase']=='report' and r['optimizer_updates_this_process']==0
 assert len(training)==6 and len(diagnostics)==2 and checkpoint_count==30
 calruns=[x['result'] for x in training if x['name'].startswith('cal-')];chosen=select_lrs(calruns,cfg);assert chosen==selection['selected_lrs']
 for x in selection['calibration_results']:
  name=Path(x['path']).parent.name;assert sha(stage/name/'result.json')==x['sha256']
 assert max(x['result']['finished_utc'] for x in training if x['name'].startswith('cal-'))<=selection['created_utc']
 for seed in cfg['seeds']:
  xs=[x['result'] for x in training if x['result']['identity']['seed']==seed];assert len({x['initial_sha256'] for x in xs})==1
 for k in cfg['methods']:
  x=next(x for x in training if x['name']==f'repeat-k{k}')['result'];assert x['identity']['lr']==chosen[str(k)]
 # Verify new question identity and original correct-answer preservation.
 meta=read(data/'tasks.json');meta_map={(x['item_id'],x['variant']):x for x in meta};assert len(meta_map)==len(meta)==192
 src=R/'data/task-quality-sources-v0';assert sha(src/'source-manifest.json')==cfg['task_source_manifest_sha256']
 original=pq.read_table(src/'test.parquet').to_pylist();old=set(cfg['excluded_old_task_article_hashes'])
 for x in meta:
  orig=original[x['source_row']];h=hashlib.sha256(orig['article'].strip().encode()).hexdigest()
  assert h==x['article_hash'] and h not in old and orig['example_id']==x['source_id']
  assert sorted(orig['options'])==sorted(x['options']) and x['options'][x['gold']]==orig['options']['ABCD'.index(orig['answer'])]
 ids=sorted({x['item_id'] for x in meta});assert len(ids)==64
 reports={};predictions=0
 for job in controller['runs']:
  path=stage/job['name'];r=read(path/'result.json')
  if r['phase']!='report':continue
  assert job['started_utc']>selection['created_utc'];k=r['identity']['k'];seed=r['identity']['seed'];lr=chosen[str(k)];step=r['step']
  parent=stage/(f'cal-k{k}-lr{lr:g}' if seed==cfg['seeds'][0] else f'repeat-k{k}')
  replay=read(path/'replay-verification.json');assert replay['step']==step and replay['max_abs_error']<=1e-6 and replay['checkpoint_sha256']==sha(parent/f'checkpoint-{step}.pt')
  task=read(path/'task-results.json');assert task['checkpoint_sha256']==replay['checkpoint_sha256'] and task['step']==step
  assert task['task_metadata_sha256']==cfg['task_metadata_sha256'] and task['task_tokens_sha256']==cfg['task_tokens_sha256']
  preds=task['predictions'];assert len(preds)==192 and preds==[json.loads(s) for s in (path/'task-predictions.jsonl').read_text().splitlines()]
  m={(x['item_id'],x['variant']):x for x in preds};assert m.keys()==meta_map.keys()
  for key,x in m.items():
   assert len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
   assert x['gold']==meta_map[key]['gold'] and x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
  ev=r['evaluations'];assert len(ev)==1 and ev[0]['split']=='report' and ev[0]['step']==step and len(ev[0]['values'])==9
  assert np.isfinite(ev[0]['values']).all() and abs(np.mean(ev[0]['values'])-ev[0]['mean_nll'])<1e-12
  context=read(path/'context-diagnostic.json')
  assert all(len(context[key])==9 and np.isfinite(context[key]).all() for key in ['full_context_tail_nll','short_context_tail_nll','benefit_nats'])
  assert np.max(np.abs(np.array(context['short_context_tail_nll'])-context['full_context_tail_nll']-np.array(context['benefit_nats'])))<1e-10
  reports[k,seed,step]=dict(result=r,predictions=m,values=ev[0]['values'],context=context)
  predictions+=len(preds)
 assert predictions==1152 and len(reports)==6
 return finalize(dest,stage,cfg,protocol,verification,controller,training,diagnostics,reports,ids,chosen,checkpoint_count,predictions)

def finalize(dest,stage,cfg,protocol,verification,controller,training,diagnostics,reports,ids,chosen,checkpoint_count,predictions):
 variants=['short','no_context','long32768'];seeds=cfg['seeds'];arrays={};points=[];run_map={x['name']:x['result'] for x in training}
 for k in cfg['methods']:
  xs=[reports[k,s,64] for s in seeds];arrays[k]={'nll':np.array([x['values'] for x in xs])}
  for v in variants:arrays[k][v]=np.array([[x['predictions'][i,v]['correct'] for i in ids] for x in xs],dtype=float)
  chosen_runs=[run_map[f'cal-k{k}-lr{chosen[str(k)]:g}'],run_map[f'repeat-k{k}']]
  points.append(dict(k=k,lr=chosen[str(k)],nll=float(arrays[k]['nll'].mean()),ppl=float(np.exp(arrays[k]['nll'].mean())),
   mean_training_seconds=float(np.mean([x['training_seconds'] for x in chosen_runs])),mean_process_seconds=float(np.mean([x['wall_seconds'] for x in chosen_runs])),
   per_seed_training_seconds=[x['training_seconds'] for x in chosen_runs],per_seed_ppl=[float(np.exp(np.mean(x['values']))) for x in xs],
   total_search_and_repeat_training_seconds=sum(x['result']['training_seconds'] for x in training if x['result']['identity']['k']==k),
   total_search_and_repeat_process_seconds=sum(x['result']['wall_seconds'] for x in training if x['result']['identity']['k']==k),
   accuracy={v:float(arrays[k][v].mean()*100) for v in variants},correct={v:[int(a.sum()) for a in arrays[k][v]] for v in variants}))
 d,s=points;rng=np.random.default_rng(2026091685);contrasts={}
 for metric,n in [('nll',9)]+[(v,64) for v in variants]:
  diff=(arrays[32][metric]-arrays[0][metric]).mean(0);samples=rng.integers(0,n,(10000,n));q=np.quantile(diff[samples].mean(1),[.025,.975])
  contrasts[metric]=dict(mean_difference=float(diff.mean()),descriptive_95_interval=q.tolist())
 comparison=dict(time_saved_pct=100*(1-s['mean_training_seconds']/d['mean_training_seconds']),ppl_increase_pct=100*(s['ppl']/d['ppl']-1),
  ppl_increase_descriptive_95_interval=(100*np.expm1(contrasts['nll']['descriptive_95_interval'])).tolist(),
  per_seed_ppl_increase_pct=(100*np.expm1((arrays[32]['nll']-arrays[0]['nll']).mean(1))).tolist(),
  per_seed_time_saved_pct=[100*(1-a/b) for a,b in zip(s['per_seed_training_seconds'],d['per_seed_training_seconds'])],
  task_gap_pp={v:100*contrasts[v]['mean_difference'] for v in variants},
  task_gap_descriptive_95_interval_pp={v:(100*np.array(contrasts[v]['descriptive_95_interval'])).tolist() for v in variants},
  search_and_repeat_training_time_saved_pct=100*(1-s['total_search_and_repeat_training_seconds']/d['total_search_and_repeat_training_seconds']))
 baseline=[]
 for k in [0,32]:
  r=reports[k,seeds[0],0];baseline.append(dict(k=k,ppl=float(np.exp(np.mean(r['values']))),accuracy={v:100*np.mean([r['predictions'][i,v]['correct'] for i in ids]) for v in variants}))
 contexts=[];depths=[];meta=read(dest/'data/32k-adaptation-v0/tasks.json')
 for k in [0,32]:
  contexts.append(dict(k=k,full_context_tail_nll=float(np.mean([reports[k,s,64]['context']['full_context_tail_nll'] for s in seeds])),short_context_tail_nll=float(np.mean([reports[k,s,64]['context']['short_context_tail_nll'] for s in seeds])),benefit_nats=float(np.mean([reports[k,s,64]['context']['benefit_nats'] for s in seeds]))))
 for fraction in [.1,.35,.65,.9]:
  part=[x['item_id'] for x in meta if x['variant']=='long32768' and x['evidence_fraction']==fraction];assert len(part)==16
  scores={str(k):[sum(reports[k,s,64]['predictions'][i,'long32768']['correct'] for i in part) for s in seeds] for k in [0,32]}
  depths.append(dict(evidence_fraction=fraction,items=16,correct=scores,accuracy_gap_pp=(np.mean(scores['32'])-np.mean(scores['0']))/16*100))
 calpoints=[]
 for x in training:
  r=x['result']
  for e in r['evaluations']:
   calpoints.append(dict(run=x['name'],k=r['identity']['k'],lr=r['identity']['lr'],seed=r['identity']['seed'],step=e['step'],seconds=sum(t['seconds'] for t in r['trace'][:e['step']]),nll=e['mean_nll'],ppl=float(np.exp(e['mean_nll']))))
 intervals=[]
 for x in controller['runs']:
  r=read(stage/x['name']/'result.json');intervals.append(dict(name=x['name'],phase=r['phase'],started_utc=x['started_utc'],finished_utc=x['finished_utc'],updates=r['optimizer_updates_this_process'],process_seconds=r['wall_seconds'],training_seconds=r['training_seconds']))
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),archive=verification,scientific_runs=6,scientific_updates=384,diagnostic_updates=12,
  scientific_target_exposures=384*32768,distinct_train_target_positions=32*32768,task_predictions=predictions,cumulative_task_predictions=11168+predictions,
  total_amp_scientific_runs=21,total_amp_scientific_updates=4224,total_amp_diagnostic_updates=55,
  checkpoint_files_audited=checkpoint_count,controller_seconds=controller['seconds'],scientific_training_seconds=sum(x['result']['training_seconds'] for x in training),
  scientific_process_seconds=sum(x['result']['wall_seconds'] for x in training),diagnostic_process_seconds=sum(x['wall_seconds'] for x in diagnostics),
  selected_lrs=chosen,selected_points=points,comparison=comparison,baseline=baseline,calibration_curves=calpoints,timeline=intervals,context_diagnostics=contexts,evidence_depth_breakdown=depths,
  scope=cfg['scope'])
 out=R/'results/32k-adaptation-audit-v0';out.mkdir(exist_ok=False);dump(out/'result.json',audit);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.8));colors={0:'#2166ac',32:'#d6604d'}
 for k in [0,32]:
  for seed in seeds:
   ps=[x for x in calpoints if x['k']==k and x['lr']==chosen[str(k)] and x['seed']==seed]
   axes[0].plot([p['seconds'] for p in ps],[p['ppl'] for p in ps],'-o',alpha=.8 if seed==seeds[0] else .45,color=colors[k],label=('Dense' if k==0 else 'Sparse K32') if seed==seeds[0] else None)
 axes[0].set_xlabel('Cumulative training seconds');axes[0].set_ylabel('Calibration perplexity (lower is better)');axes[0].set_title('Selected LR trajectories; development data');axes[0].legend();axes[0].grid(alpha=.25)
 pos=np.arange(3)
 for p in points:
  shift=-.18 if p['k']==0 else .18;vals=[p['accuracy'][v] for v in variants]
  axes[1].bar(pos+shift,vals,width=.34,color=colors[p['k']],label='Dense' if p['k']==0 else 'Sparse K32')
  for j,v in enumerate(variants):
   for seed_i in [0,1]:axes[1].scatter(j+shift,p['correct'][v][seed_i]/64*100,color='black',s=18)
 axes[1].set_xticks(pos,['Short','No context','32K']);axes[1].set_ylabel('Choice accuracy (%)');axes[1].set_ylim(0,100);axes[1].legend();axes[1].set_title('64 new marked-target RACE questions');axes[1].grid(axis='y',alpha=.25)
 fig.suptitle('Qwen2.5-0.5B LoRA: 32K, 64 training steps, two initialization seeds');fig.tight_layout();(R/'docs/figures').mkdir(exist_ok=True);fig.savefig(R/'docs/figures/32k-adaptation-2026-09-16.png',dpi=180);plt.close(fig)
 lines=['# 32K稀疏适配：完整训练成本与质量对照','','本轮目标是更低成本、质量足够接近；不要求稀疏反超密集。这里报告完整的固定训练量折中，并分别展示语料PPL和阅读准确率。','',
 '## 主要结果','','|条件|选中LR|平均训练秒数|测试PPL|短题准确率|无原文准确率|32K准确率|','|---|---:|---:|---:|---:|---:|---:|']
 for p in points:lines.append(f"|{'密集' if p['k']==0 else '稀疏K32'}|{p['lr']:g}|{p['mean_training_seconds']:.2f}|{p['ppl']:.4f}|{p['accuracy']['short']:.2f}%|{p['accuracy']['no_context']:.2f}%|{p['accuracy']['long32768']:.2f}%|")
 lines+=['',f"稀疏完整训练循环节时 **{comparison['time_saved_pct']:.2f}%**；测试PPL增加 **{comparison['ppl_increase_pct']:.2f}%**；32K任务差 **{comparison['task_gap_pp']['long32768']:+.2f}个百分点**。PPL百分比与准确率百分点不是同一个量。",
 f"两个种子各自节时：{comparison['per_seed_time_saved_pct'][0]:.2f}% / {comparison['per_seed_time_saved_pct'][1]:.2f}%；PPL增加：{comparison['per_seed_ppl_increase_pct'][0]:.2f}% / {comparison['per_seed_ppl_increase_pct'][1]:.2f}%。",'',
 '训练时间来自同步计时的输入搬运、完整前向、全部token损失反向、梯度裁剪、AdamW和学习率调度；包含预热阶段更新，但不含加载、校准、存档或最终评测。这次比上一轮只测前反向更接近实际训练循环成本。','',
 '## 语料与任务的不确定性','','配对重采样先平均种子，再按9个语料窗口或64个问题重采样；窗口相邻、样本少，仅为描述性区间，不因差异不显著就宣布等价。','',
 '|指标|稀疏相对密集|描述性95%区间|','|---|---:|---|']
 q=comparison['ppl_increase_descriptive_95_interval'];lines.append(f"|PPL增加|{comparison['ppl_increase_pct']:+.2f}%|[{q[0]:+.2f}, {q[1]:+.2f}]%|")
 for v in variants:
  q=comparison['task_gap_descriptive_95_interval_pp'][v];lines.append(f"|{v}准确率变化|{comparison['task_gap_pp'][v]:+.2f}pp|[{q[0]:+.2f}, {q[1]:+.2f}]pp|")
 lines+=['','## 每个种子的正确题数','','每格分母64；短题、无原文和32K是同题变体，不是三个独立题集。','', '|条件|短题种子0/1|无原文种子0/1|32K种子0/1|','|---|---|---|---|']
 for p in points:lines.append(f"|K{p['k']}|{p['correct']['short']}|{p['correct']['no_context']}|{p['correct']['long32768']}|")
 lines+=['','短题不超过1200 token，K32在该长度覆盖所有可见块，短题接近不能单独证明稀疏长文效果接近。无原文变体用于观察猜题能力，不作为“没学坏”的唯一证据。','','## 固定位置分组（描述性）','','完整64题仍为主要结果；下面不挑有利组，四个位置都保留，每组仅16题。','', '|原文位置|密集两种子正确/16|K32两种子正确/16|差值pp|','|---|---|---|---:|']
 for x in depths:lines.append(f"|{x['evidence_fraction']:.0%}|{x['correct']['0']}|{x['correct']['32']}|{x['accuracy_gap_pp']:+.2f}|")
 lines+=['','## 不同容忍幅度的描述性判断','','只比较点估计，并同时要求实际节省训练时间；这不是统计等价检验或唯一验收线。','', '|允许代价|PPL是否落入范围|32K答题是否落入范围|','|---|---|---|']
 for tol in [3,5,10]:
  ppl_ok=comparison['time_saved_pct']>0 and comparison['ppl_increase_pct']<=tol
  task_ok=comparison['time_saved_pct']>0 and comparison['task_gap_pp']['long32768']>=-tol
  lines.append(f"|PPL {tol}% / 任务 {tol}pp|{'是' if ppl_ok else '否'}|{'是' if task_ok else '否'}|")
 lines+=['','## 相同尾部目标的上下文诊断','','固定最后4096个目标token，比较完整32K上下文与最后4K。短输入位置重新编号，诊断含长度和位置变化，不能单独定位路由损伤。收益为短上下文NLL减完整上下文NLL，越大表示这些目标更依赖额外上下文。','', '|方法|32K尾部NLL|4K尾部NLL|额外上下文收益nats|','|---|---:|---:|---:|']
 for x in contexts:lines.append(f"|K{x['k']}|{x['full_context_tail_nll']:.5f}|{x['short_context_tail_nll']:.5f}|{x['benefit_nats']:+.5f}|")
 lines+=['','## 没有进行适配时的基线','','种子0的0步断点，LoRA增量为零，分别按密集/K32运行；这部分没有追加优化更新。','', '|注意力|测试PPL|短题|无原文|32K|','|---|---:|---:|---:|---:|']
 for b in baseline:lines.append(f"|K{b['k']}|{b['ppl']:.4f}|{b['accuracy']['short']:.2f}%|{b['accuracy']['no_context']:.2f}%|{b['accuracy']['long32768']:.2f}%|")
 lines+=['','## 搜索预算与完整成本','','每边两个LR候选、同样64步，再复核选中LR的第二种子；每边3条科学训练。最终评测是在选择锁定后才读取。','', '|条件|搜索+复核训练循环秒|搜索+复核进程秒|','|---|---:|---:|']
 for p in points:lines.append(f"|K{p['k']}|{p['total_search_and_repeat_training_seconds']:.2f}|{p['total_search_and_repeat_process_seconds']:.2f}|")
 lines += ['',f"整个控制器{controller['seconds']:.2f}秒；科学训练循环合计{audit['scientific_training_seconds']:.2f}秒，科学训练进程{audit['scientific_process_seconds']:.2f}秒，另两项预检进程{audit['diagnostic_process_seconds']:.2f}秒。GPU账单未读取，安装/空闲/下载等不在这些训练循环数字中。",'',
 '## 校准轨迹与早停参照','','下图左侧只有校准PPL，不冒充测试集的中间断点质量曲线；两边全部学习率和步数记录均保留。允许密集也提前停止，不把少训的收益全归于稀疏。','',
 '![32K训练与任务结果](figures/32k-adaptation-2026-09-16.png)','','|训练|LR|步数|累计训练秒|校准PPL|','|---|---:|---:|---:|---:|']
 for p in calpoints:lines.append(f"|{p['run']}|{p['lr']:g}|{p['step']}|{p['seconds']:.2f}|{p['ppl']:.4f}|")
 lines+=['','## 已验证与适用范围','',f"新增6条科学训练、384优化更新，另12个诊断更新；科学训练曝光{384*32768:,}目标token，独特训练位置约{32*32768:,}。累计本AMP分支21条科学训练4224更新、55诊断更新。新增1152次任务计分前向，累计12320次，不是独立题数。",f"证据包{verification['files']}个文件逐项大小/SHA核验；30个科学检查点的参数有限性、身份、数据游标、训练前缀、优化器/scheduler/RNG审计；两项GPU预检通过，所有最终与0步断点先复现校准误差≤1e-6，1152个选项argmax与原始正确标签全部复算。",'全部14项GPU作业成功后，控制器在归档时因相对输出路径与绝对项目根混用而异常退出。保留不完整包，用绝对输出路径调用未改动的原打包器恢复；没有重跑训练或预测。archive-recovery.json记录修复，原打包器源码保留在证据包中，当前工作区另修复路径处理以供后续使用。','模型是密集预训练0.5B的LoRA适配，非Qwen4、非从头原生稀疏全参数训练。训练和测试为WikiText流分块，可跨文章边界。64篇RACE材料排除旧文章，但带TARGET标记并加入无关文段，非官方长文基准。预训练污染未排除；两个种子共享数据顺序。样本数与语料覆盖不足以证明普遍等价或原创论文贡献。',f"证据包SHA256：{verification['sha256']}。原始逐步UTC/损失/学习率/范数在隔离证据目录；简明时间轴见docs/32k-adaptation-timeline-2026-09-16.md。",'']
 (R/'docs/32k-adaptation-results-2026-09-16.md').write_text('\n'.join(lines),encoding='utf-8')
 lines=['# 32K适配作业时间轴（UTC）','','|作业|开始UTC|结束UTC|实际优化更新|进程秒|训练循环秒|','|---|---|---|---:|---:|---:|']
 for x in intervals:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['updates']}|{x['process_seconds']:.2f}|{x['training_seconds']:.2f}|")
 lines+=['','预检会回滚断点后复算两步，实际各6更新；result.trace只保留最终4步路径，所以预检训练循环列不包含被回滚的两个更新耗时。进程耗时包含它们，events.jsonl保留全部6个更新事件。科学训练没有回滚。','']
 (R/'docs/32k-adaptation-timeline-2026-09-16.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(dict(status='complete',selected_lrs=chosen,points=points,comparison=comparison,archive=verification)))

if __name__=='__main__':main()
