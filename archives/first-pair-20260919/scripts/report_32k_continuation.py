"""Verify complete continuation evidence and produce cost-quality report."""
import hashlib,json,math,tarfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 archive=R/'exports/32k-continuation-evidence-v0.tar.gz';proof=load(archive.with_suffix('.json'))
 assert sha(archive)==proof['sha256'] and archive.stat().st_size==proof['bytes']
 dest=R/'results/cloud-32k-continuation-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(archive) as tar:
  entries=load_tar= json.loads(tar.extractfile('32k-continuation-manifest.json').read())['files']
  names=[m.name for m in tar.getmembers()];assert len(set(names))==len(names)
  assert set(names)=={e['path'] for e in entries}|{'32k-continuation-manifest.json'}
  assert len(entries)==proof['files']
  for e in entries:
   m=tar.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   raw=tar.extractfile(m).read();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
   p=dest/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 stage=dest/'results/32k-continuation-stage-v0';protocol=load(dest/'provenance/32k-continuation-protocol.json');psha=sha(dest/'provenance/32k-continuation-protocol.json')
 assert psha==sha(R/'provenance/32k-continuation-protocol.json')
 for n,h in protocol['source_sha256'].items():assert sha(dest/n)==h
 control=load(stage/'result.json');assert control['status']=='complete' and len(control['jobs'])==14 and all(j['returncode']==0 for j in control['jobs'])
 lock=load(stage/'report-lock.json');assert lock['protocol_sha256']==psha
 assert lock['created_utc']<min(j['started_utc'] for j in control['jobs'] if j['phase']=='report')
 # Check new article exclusions and balance independently of preparation code.
 meta=load(dest/protocol['task_data_path']/'tasks.json');items={x['item_id']:x for x in meta if x['variant']=='long32768'}
 assert len(meta)==192 and len(items)==64
 hashes={x['article_hash'] for x in items.values()};assert len(hashes)==64
 previous={x['article_hash'] for n in ['pilot','formal'] for x in load(R/f'data/task-quality-v0/{n}.json') if x['task']=='race_mc'}|{x['article_hash'] for x in load(R/'data/32k-adaptation-v0/tasks.json')}
 assert hashes.isdisjoint(previous)
 for depth in [.1,.35,.65,.9]:
  for label in range(4):assert sum(x['evidence_fraction']==depth and x['gold']==label for x in items.values())==4
 import torch
 torch.set_num_threads(4)
 trains={};checkpoint_count=0
 order=np.random.default_rng(2026091662).permutation(32).tolist()
 mirror=R/'results/cloud-32k-adaptation-evidence-v0'
 for k,seed in protocol['run_order']:
  key=f'k{k}-seed{seed}';d=stage/key;v=load(d/'result.json');j=protocol['parents'][key]
  assert v['status']=='complete' and v['optimizer_updates_this_process']==64 and v['step']==128
  trace=v['trace'];assert len(trace)==128 and [x['step'] for x in trace]==list(range(1,129))
  original=load(mirror/Path(j['checkpoint']).parent/'result.json');assert trace[:64]==original['trace']
  assert all(x['lr']==.0001 and x['window_index']==order[(x['step']-1)%32] and math.isfinite(x['train_nll']) and x['seconds']>0 for x in trace[64:])
  assert abs(sum(x['seconds'] for x in trace)-v['training_seconds'])<1e-8
  verification=load(d/'parent-verification.json');assert verification['replay_error']<=1e-6 and verification['parent_checkpoint_sha256']==j['checkpoint_sha256']
  assert sha(d/'source.py')==protocol['source_sha256']['scripts/run_32k_continuation.py']
  assert v['initial_sha256']==original['initial_sha256']
  for cp in d.glob('checkpoint-*.pt'):
   c=torch.load(cp,map_location='cpu',weights_only=False);s=c['step'];assert c['data_cursor']==s and c['identity']==verification['child_identity']
   assert c['extra']['rows']==trace[:s] and len(c['params'])>0 and c['optimizer']['state']
   assert c['scheduler']['last_epoch']==s and c['optimizer']['param_groups'][0]['lr']==.0001
   assert all(float(x['step'])==s for x in c['optimizer']['state'].values())
   assert all(torch.isfinite(x).all() for x in c['params'].values())
   assert c['rng']['cuda'] is not None and c['rng']['python'] and c['rng']['numpy']
   checkpoint_count+=1
  if k==32:
   cut=load(d/'time-budget-cut.json');s=cut['compliant_step'];budget=j['dense64_training_seconds']
   assert sum(x['seconds'] for x in trace[:s])<=budget<sum(x['seconds'] for x in trace[:s+1])
   assert (d/f'checkpoint-{s}.pt').exists() and (d/f'checkpoint-{s+1}.pt').exists()
  trains[key]=v
 reports={};total_predictions=0
 for t in lock['reports']:
  d=stage/t['name'];v=load(d/'result.json');assert v['status']=='complete' and v['optimizer_updates_this_process']==0
  cp=(dest if t['origin']=='continuation' else mirror)/t['checkpoint'];assert sha(cp)==t['checkpoint_sha256']
  replay=load(d/'replay-verification.json');assert replay['max_abs_error'] is None or replay['max_abs_error']<=1e-6
  task=load(d/'task-results.json');assert task['task_metadata_sha256']==protocol['task_metadata_sha256'] and task['task_tokens_sha256']==protocol['task_tokens_sha256']
  assert len(task['predictions'])==192
  for q,m in zip(task['predictions'],meta):
   assert (q['item_id'],q['variant'],q['gold'])==(m['item_id'],m['variant'],m['gold'])
   assert q['prediction']==int(np.argmax(q['choice_logits'])) and q['correct']==(q['prediction']==q['gold']) and np.isfinite(q['choice_logits']).all()
  ev=[e for e in v['evaluations'] if e['split']=='report'];assert len(ev)==1 and len(ev[0]['values'])==9
  if t['origin']=='parent':
   oldroot=R/'results/cloud-32k-separator-replay-evidence-v0/results/32k-separator-replay-stage-v0'
   matches=[]
   for f in oldroot.rglob('result.json'):
    old=load(f)
    if old.get('identity',{}).get('k')==t['k'] and old.get('identity',{}).get('seed')==t['seed'] and old.get('step')==t['step']:
     matches.append(old)
   assert len(matches)==1
   oldeval=[e for e in matches[0]['evaluations'] if e['split']=='report'][0]
   assert np.max(np.abs(np.array(oldeval['values'])-ev[0]['values']))<=1e-6

  scores={var:np.array([int(q['correct']) for q in task['predictions'] if q['variant']==var]) for var in ['short','no_context','long32768']}
  if t['step']==0:seconds=0
  elif t['origin']=='continuation':seconds=sum(x['seconds'] for x in trains[f"k{t['k']}-seed{t['seed']}"]['trace'][:t['step']])
  else:seconds=protocol['parents'][f"k{t['k']}-seed{t['seed']}"]['parent_training_seconds']
  reports[t['name']]=dict(k=t['k'],seed=t['seed'],step=t['step'],seconds=seconds,nll=ev[0]['values'],ppl=math.exp(np.mean(ev[0]['values'])),correct={v:int(a.sum()) for v,a in scores.items()},scores={v:a.tolist() for v,a in scores.items()})
  total_predictions+=192
 assert total_predictions==1920
 seeds=protocol['seeds'];groups={}
 for tag,k,kind in [('dense64',0,'step64'),('dense128',0,'step128'),('sparse_budget',32,'budget'),('sparse128',32,'step128')]:
  rs=[reports[f'report-k{k}-seed{seed}-{kind}'] for seed in seeds]
  groups[tag]=dict(steps=[v['step'] for v in rs],training_seconds=float(np.mean([v['seconds'] for v in rs])),ppl=float(math.exp(np.mean([v['nll'] for v in rs]))),accuracy={var:float(np.mean([v['scores'][var] for v in rs])) for var in ['short','no_context','long32768']},correct={var:[v['correct'][var] for v in rs] for var in ['short','no_context','long32768']})
 for k in [0,32]:
  v=reports[f'report-k{k}-baseline'];groups[f'baseline{k}']=dict(steps=[0],training_seconds=0,ppl=v['ppl'],accuracy={var:float(np.mean(v['scores'][var])) for var in ['short','no_context','long32768']},correct={var:[v['correct'][var]] for var in ['short','no_context','long32768']})
 rng=np.random.default_rng(2026091690);ci_indices={n:rng.integers(0,n,(10000,n)) for n in [9,64]}
 comparisons={}
 for tag,st,dt in [('near_equal_time','budget','step64'),('equal_tokens128','step128','step128'),('sparse128_vs_dense64','step128','step64')]:
  sr=[reports[f'report-k32-seed{s}-{st}'] for s in seeds];dr=[reports[f'report-k0-seed{s}-{dt}'] for s in seeds]
  diff=np.mean([v['nll'] for v in sr],axis=0)-np.mean([v['nll'] for v in dr],axis=0)
  delta=(np.exp(diff.mean())-1)*100;boot=(np.exp(diff[ci_indices[9]].mean(axis=1))-1)*100
  res=dict(ppl_increase_percent=float(delta),ppl_ci95=np.percentile(boot,[2.5,97.5]).tolist(),training_saving_percent=100*(1-np.mean([v['seconds'] for v in sr])/np.mean([v['seconds'] for v in dr])),tasks={})
  for var in ['short','no_context','long32768']:
   d=np.mean([v['scores'][var] for v in sr],axis=0)-np.mean([v['scores'][var] for v in dr],axis=0)
   res['tasks'][var]=dict(delta_pp=100*float(d.mean()),ci95=(100*np.percentile(d[ci_indices[64]].mean(axis=1),[2.5,97.5])).tolist())
  comparisons[tag]=res
 training_seconds=sum(sum(x['seconds'] for x in v['trace'][64:]) for v in trains.values())
 curve=[]
 for key,v in trains.items():
  for e in v['evaluations']:
   if e['split']=='calibration':curve.append(dict(key=key,step=e['step'],training_seconds=sum(x['seconds'] for x in v['trace'][:e['step']]),nll=e['mean_nll'],ppl=math.exp(e['mean_nll'])))
 epochs={key:[dict(epoch=e+1,start_step=e*32+1,end_step=(e+1)*32,mean_train_nll=float(np.mean([x['train_nll'] for x in v['trace'][e*32:(e+1)*32]]))) for e in range(4)] for key,v in trains.items()}
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),archive=proof,verified_checkpoint_count=checkpoint_count,new_updates=256,new_unique_initializations=0,continued_trajectories=4,new_target_exposures=256*32768,epoch_train_nll=epochs,task_predictions=total_predictions,training_seconds_this_stage=training_seconds,controller_seconds=control['seconds'],training_process_seconds=sum(v['wall_seconds'] for v in trains.values()),groups=groups,comparisons=comparisons,calibration_curve=curve,reports=reports,jobs=control['jobs'],cumulative_amp_updates=4480,cumulative_amp_diagnostic_updates=55,cumulative_task_predictions=14240)
 save(R/'results/32k-continuation-audit-v0/result.json',audit);save(dest/'LOCAL-VERIFICATION.json',dict(status='verified',archive=proof,checkpoints=checkpoint_count,new_updates=256,predictions=1920,utc=audit['utc']))
 lines=['# 32K续训结果：把省下的时间用于更多更新','',f"完成UTC：{control['finished_utc']}。四条已有轨迹各从64续至128步，新增256更新；完整参数、AdamW、调度器、随机状态和数据顺序连续，学习率沿用0.0001尾段。目标是较低成本与足够接近的质量，不要求稀疏质量超过密集。",'', '## 实测对照','', '|条件|步数（两种子）|累计训练秒|测试PPL|短题|无原文|32K|','|---|---|---:|---:|---:|---:|---:|']
 names={'dense64':'密集64','dense128':'密集128','sparse_budget':'稀疏近似等时','sparse128':'稀疏128','baseline0':'密集零更新','baseline32':'稀疏零更新'}
 for name,g in groups.items():lines.append(f"|{names[name]}|{' / '.join(map(str,g['steps']))}|{g['training_seconds']:.2f}|{g['ppl']:.4f}|{100*g['accuracy']['short']:.2f}%|{100*g['accuracy']['no_context']:.2f}%|{100*g['accuracy']['long32768']:.2f}%|")
 lines+=['','PPL按两种子、同9个窗口平均NLL取指数。任务是64篇本项目此前未评测的文章，每篇短/无原文/32K三个变体；单种子每格分母64，不能把重复种子当成新题。零更新仅种子0，LoRA增量为零。','', '## 成本与质量折中','', '|比较（稀疏相对密集）|训练节时|PPL增加|短题差|无原文差|32K差|','|---|---:|---:|---:|---:|---:|']
 labels={'near_equal_time':'稀疏等时点 vs 密集64','equal_tokens128':'稀疏128 vs 密集128','sparse128_vs_dense64':'稀疏128 vs 密集64'}
 for n,c in comparisons.items():lines.append(f"|{labels[n]}|{c['training_saving_percent']:.2f}%|{c['ppl_increase_percent']:.2f}%|{c['tasks']['short']['delta_pp']:+.2f}pp|{c['tasks']['no_context']['delta_pp']:+.2f}pp|{c['tasks']['long32768']['delta_pp']:+.2f}pp|")
 lines+=['','节时负数表示花费更多。等时点取最后一个累计训练秒数不超过对应密集64步预算的完整更新；保留少于一步的余量，不宣称绝对相同墙钟时间。累计采用历史父阶段实测秒数加本轮同步训练循环秒数，包含输入搬运/前反向/AdamW，加载、校准、存盘、测试和空闲另计。','', '## 描述性不确定区间','']
 for n,c in comparisons.items():
  q=c['tasks']['long32768'];lines.append(f"- {labels[n]}：PPL增加 {c['ppl_increase_percent']:.2f}%，窗口配对95%区间[{c['ppl_ci95'][0]:.2f}, {c['ppl_ci95'][1]:.2f}]%；32K差 {q['delta_pp']:+.2f}pp，问题配对95%区间[{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}]pp。")
 lines+=['','上述先平均种子，再重采样9个相邻窗口或64篇问题，仅描述性；区间包含0不等于证明等价。WikiText是已观察测试集的探索复用，新RACE文章只保证项目内未用过，不排除预训练污染。无原文较高、短题K32覆盖全部可见块时，不能把这些结果泛化成完整长上下文能力保持。','', '## 两种子正确题数','', '|条件|短题 /64|无原文 /64|32K /64|','|---|---|---|---|']
 for name,g in groups.items():lines.append(f"|{names[name]}|{g['correct']['short']}|{g['correct']['no_context']}|{g['correct']['long32768']}|")
 lines+=['','## 校准曲线与学习率限制','','![续训校准与成本](figures/32k-continuation-2026-09-16.png)','','|轨迹|步数|累计训练秒|校准PPL|','|---|---:|---:|---:|']
 for c in curve:lines.append(f"|{c['key']}|{c['step']}|{c['training_seconds']:.2f}|{c['ppl']:.4f}|")
 lines+=['','这是延长原64步训练的低学习率尾段，不是重新优化128步学习率曲线。若收益小，只能说明这一续训配方边际收益小，不能据此否定所有稀疏训练。训练数据仍是原32个窗口循环，没有新增独特训练文本。','', '## 日志、核验和总成本','', f"新增256更新，{256*32768:,}目标token曝光；4条已有轨迹延长，不新计为独立初始化。新增1920次任务计分前向。AMP累计4480科学更新、55诊断更新，任务累计14240次。",'',f"本轮新增科学训练循环 {training_seconds:.2f}秒；训练进程 {audit['training_process_seconds']:.2f}秒；控制器含评测 {control['seconds']:.2f}秒。实际租卡账单未知，未折算美元。",'',f"原始包SHA `{proof['sha256']}`，{proof['files']}文件、{checkpoint_count}续训完整检查点逐一核验。4条父断点校准重放≤1e-6；计时切点边界、全部optimizer步骤/LR/数据游标与1920个预测argmax复算通过。",'', '原始包：exports/32k-continuation-evidence-v0.tar.gz；本地镜像：results/cloud-32k-continuation-evidence-v0；机器结果：results/32k-continuation-audit-v0/result.json。协议和旧阶段保持不变。']
 lines+=['','## 不同容忍范围（点估计）','','只要求节省训练时间并把质量代价控制在范围内；不要求质量优于密集。下面列出多个范围，不在看到结果后挑一个宣称正式等价。','','|比较|PPL代价≤3%且32K下降≤3pp|≤5%且≤5pp|≤10%且≤10pp|','|---|---|---|---|']
 for n in ['near_equal_time','equal_tokens128']:
  c=comparisons[n];verdicts=['是' if c['training_saving_percent']>0 and c['ppl_increase_percent']<=tol and c['tasks']['long32768']['delta_pp']>=-tol else '否' for tol in [3,5,10]]
  lines.append('|'+labels[n]+'|'+'|'.join(verdicts)+'|')
 lines+=['','新题上的32K相对无原文增益如下。正数表示有上下文时答对更多；这个诊断仍含位置、干扰文本和提示等因素，不直接等同因果证明。','']
 for name,g in groups.items():lines.append(f"- {names[name]}：{100*(g['accuracy']['long32768']-g['accuracy']['no_context']):+.2f}pp。")
 lines+=['','## 重复曝光与训练损失','','每遍都是固定的32个训练窗口，下面按完整一遍平均，避免把不同窗口的难度波动误判成收敛速度。64步为两遍，128步为四遍；每条轨迹累计4,194,304目标曝光、1,048,576独特目标位置。','','|轨迹|第1遍训练NLL|第2遍|第3遍|第4遍|','|---|---:|---:|---:|---:|']
 for key,es in epochs.items():lines.append('|'+key+'|'+'|'.join(f"{e['mean_train_nll']:.5f}" for e in es)+'|')
 lines+=['','训练NLL是更新过程中的损失，校准NLL是固定断点评价；二者测量方式不同。若训练下降而校准停滞，只提示额外重复的收益受限，不单独证明一个稀疏机制失效。']
 (R/'docs/32k-continuation-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 timeline=['# 32K续训时间轴','','全部UTC；父训练不计入本轮新增更新。','','|作业|开始|结束|状态|','|---|---|---|---|']
 for j in control['jobs']:timeline.append(f"|{j['name']}|{j['started_utc']}|{j['finished_utc']}|exit {j['returncode']}|")
 (R/'docs/32k-continuation-timeline-2026-09-16.md').write_text('\n'.join(timeline)+'\n',encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.3))
 for key,v in trains.items():
  cs=[c for c in curve if c['key']==key and c['step']>=64];sparse=key.startswith('k32');col='#bd5628' if sparse else '#2368a0'
  axes[0].plot([c['step'] for c in cs],[c['ppl'] for c in cs],marker='o',color=col,linestyle='-' if key.endswith('660') else '--',label=key)
 axes[0].set(xlabel='Cumulative steps',ylabel='Calibration PPL (lower is better)',title='Continued at LR 0.0001');axes[0].legend(fontsize=7)
 for n,g in groups.items():
  axes[1].scatter(g['training_seconds'],g['ppl'],s=40);axes[1].annotate(n,(g['training_seconds'],g['ppl']),xytext=(4,5),textcoords='offset points',fontsize=7)
 axes[1].set(xlabel='Cumulative measured training seconds',ylabel='Reused WikiText test PPL',title='Cost-quality points (not billing)')
 for ax in axes:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(R/'docs/figures/32k-continuation-2026-09-16.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(status='verified',groups=groups,comparisons=comparisons,checkpoints=checkpoint_count,seconds=control['seconds']),ensure_ascii=False))
if __name__=='__main__':main()
