"""Verify every checkpoint evaluation and report the full empirical cost-quality curve."""
import hashlib,json,tarfile,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 archive=R/'exports/checkpoint-quality-cost-evidence-v0.tar.gz';proof=read(archive.with_suffix('.json'))
 assert sha(archive)==proof['sha256']
 dest=R/'results/cloud-checkpoint-quality-cost-evidence-v0';dest.mkdir(exist_ok=False)
 with tarfile.open(archive) as tar:
  members=tar.getmembers();names=[x.name for x in members];assert len(names)==len(set(names))
  manifest=json.load(tar.extractfile('amp-recovery-results-manifest.json'));expected={x['path']:x for x in manifest['files']}
  assert set(names)==set(expected)|{'amp-recovery-results-manifest.json'}
  for m in members:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   p=(dest/m.name).resolve();assert dest.resolve() in p.parents;raw=tar.extractfile(m).read()
   if m.name in expected:
    e=expected[m.name];assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 verification=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),sha256=proof['sha256'],files=len(expected))
 dump(dest/'LOCAL-VERIFICATION.json',verification)
 pp=dest/'provenance/checkpoint-quality-cost-protocol.json';protocol=read(pp);assert sha(pp)==sha(R/'provenance/checkpoint-quality-cost-protocol.json')
 for name,h in protocol['sources'].items():assert sha(dest/name)==h
 stage=dest/'results/checkpoint-quality-cost-stage-v0';controller=read(stage/'result.json')
 assert controller['status']=='complete' and controller['optimizer_updates']==0 and controller['protocol_sha256']==sha(pp)
 assert [x['name'] for x in controller['runs']]==protocol['job_order'] and all(x['returncode']==0 for x in controller['runs'])
 records={};count=0;nll_count=0;replays=0
 for name,spec in protocol['runs'].items():
  run_dir=R/spec['local_run_dir'];train=read(run_dir/'result.json');assert sha(run_dir/'result.json')==spec['training_result_sha256']
  d=read(stage/name/'result.json');assert d['status']=='complete' and d['optimizer_updates']==0
  assert d['protocol_sha256']==sha(pp) and d['training_result_sha256']==spec['training_result_sha256']
  assert d['task_protocol_sha256']==sha(R/'data/task-quality-v0/protocol.json')
  assert [d[x] for x in ['k','seed','lr']]==[spec[x] for x in ['k','seed','lr']]
  assert [x['step'] for x in d['steps']]==protocol['step_order'][str(spec['seed'])]
  old=read(R/f'results/cloud-task-quality-evidence-v0/results/task-quality-stage-v0/{name}/result.json')
  prior={(x['item_id'],x['variant']):x for x in old['predictions'] if x['variant'] in protocol['variants']};assert len(prior)==192
  flattened=[]
  for x in d['steps']:
   step=x['step'];assert x==read(stage/name/f'step-{step}.json')
   assert x['checkpoint_sha256']==spec['checkpoints'][str(step)]==sha(run_dir/f'checkpoint-{step}.pt')
   assert abs(x['training_seconds']-sum(t['seconds'] for t in train['trace'][:step]))<1e-8
   assert x['training_target_exposures']==step*16384
   assert len(x['heldout_values'])==29 and np.isfinite(x['heldout_values']).all()
   assert abs(x['heldout_mean_nll']-np.mean(x['heldout_values']))<1e-10
   if step!=2:
    cal=[e for e in train['evaluations'] if e['step']==step and e['split']=='calibration'];assert len(cal)==1
    assert len(cal[0]['values'])==len(x['calibration_values']) and np.max(np.abs(np.array(cal[0]['values'])-x['calibration_values']))<=1e-6
    assert x['calibration_replay_max_abs_error']<=1e-6
   else:assert x['calibration_replay_max_abs_error'] is None
   if step==256:
    hp=R/f'results/cloud-amp-confirmation-evidence-v1/results/amp-recovery-stage-v1/heldout-k{spec["k"]}-seed{spec["seed"]}/result.json'
    h=read(hp);assert h['training_result_sha256']==spec['training_result_sha256']
    assert np.max(np.abs(np.array(h['values'])-x['heldout_values']))<=1e-6
    assert x['heldout_256_replay_error']<=1e-6 and x['task_256_max_logit_error']<=1e-6
   else:assert x['heldout_256_replay_error'] is None and x['task_256_max_logit_error'] is None
   preds={(p['item_id'],p['variant']):p for p in x['predictions']};assert len(x['predictions'])==len(preds)==192 and preds.keys()==prior.keys()
   for key,p in preds.items():
    oldp=prior[key];assert p['step']==step and p['gold']==oldp['gold']
    assert len(p['choice_logits'])==4 and np.isfinite(p['choice_logits']).all()
    assert p['prediction']==int(np.argmax(p['choice_logits'])) and p['correct']==(p['prediction']==p['gold'])
    if step==256:
     assert p['prediction']==oldp['prediction'] and np.max(np.abs(np.array(p['choice_logits'])-oldp['choice_logits']))<=1e-6;replays+=1
   flattened+=x['predictions'];count+=len(preds);nll_count+=len(x['heldout_values'])+len(x['calibration_values'])
   records[name,step]=x
  assert flattened==[json.loads(s) for s in (stage/name/'predictions.jsonl').read_text().splitlines()]
  assert len(flattened)==d['task_predictions']==768 and d['nll_window_forwards']==sum(len(x['heldout_values'])+len(x['calibration_values']) for x in d['steps'])
 assert count==protocol['planned_scored_forwards']==3072 and replays==768
 ids=sorted({p['item_id'] for p in records['dense0',256]['predictions']});assert len(ids)==96
 arrays={};points=[]
 for kind in ['dense','sparse']:
  for step in protocol['steps']:
   xs=[records[kind+str(s),step] for s in [0,1]]
   a={'nll':np.array([x['heldout_values'] for x in xs])}
   for v in protocol['variants']:
    maps=[{p['item_id']:p['correct'] for p in x['predictions'] if p['variant']==v} for x in xs]
    a[v]=np.array([[m[i] for i in ids] for m in maps],dtype=float)
   arrays[kind,step]=a
   points.append(dict(kind=kind,step=step,seconds=float(np.mean([x['training_seconds'] for x in xs])),
                      nll=float(a['nll'].mean()),ppl=float(np.exp(a['nll'].mean())),
                      acc8k=float(a['long8192'].mean()*100),acc16k=float(a['long16384'].mean()*100),
                      correct8k=[int(z.sum()) for z in a['long8192']],correct16k=[int(z.sum()) for z in a['long16384']]))
 lookup={(p['kind'],p['step']):p for p in points};rng=np.random.default_rng(2026091630)
 samples={n:rng.integers(0,n,(10000,n)) for n in [29,96]};comparisons=[]
 for step in protocol['steps']:
  for refstep in sorted({step,256}):
   s=lookup['sparse',step];d=lookup['dense',refstep];c=dict(sparse_step=step,dense_step=refstep,time_saved_pct=100*(1-s['seconds']/d['seconds']),ppl_increase_pct=100*(math.exp(s['nll']-d['nll'])-1),acc8k_gap_pp=s['acc8k']-d['acc8k'],acc16k_gap_pp=s['acc16k']-d['acc16k'])
   for metric,n in [('nll',29),('long8192',96),('long16384',96)]:
    diff=(arrays['sparse',step][metric]-arrays['dense',refstep][metric]).mean(0);q=np.quantile(diff[samples[n]].mean(1),[.025,.975])
    c[metric+'_descriptive_interval']=(100*np.expm1(q) if metric=='nll' else 100*q).tolist()
   comparisons.append(c)
 # Every point is retained; frontier is limited to these observed schedules and metrics.
 for metric,sign in [('ppl',1),('acc8k',-1),('acc16k',-1)]:
  for p in points:p['frontier_'+metric]=not any(q['seconds']<=p['seconds'] and sign*q[metric]<=sign*p[metric] and (q['seconds']<p['seconds'] or sign*q[metric]<sign*p[metric]) for q in points)
 tolerances=[]
 for tol in protocol['ppl_relative_tolerances']:
  ref=lookup['dense',256];eligible=[p for p in points if p['ppl']<=ref['ppl']*(1+tol)]
  tolerances.append(dict(metric='ppl',tolerance=tol,cheapest_by_kind={k:min([p for p in eligible if p['kind']==k],key=lambda x:x['seconds'],default=None) for k in ['dense','sparse']}))
 for metric in ['acc8k','acc16k']:
  for tol in protocol['task_tolerances_pp']:
   ref=lookup['dense',256];eligible=[p for p in points if p[metric]>=ref[metric]-tol]
   tolerances.append(dict(metric=metric,tolerance=tol,cheapest_by_kind={k:min([p for p in eligible if p['kind']==k],key=lambda x:x['seconds'],default=None) for k in ['dense','sparse']}))
 out=R/'results/checkpoint-quality-cost-audit-v0';out.mkdir(exist_ok=False)
 audit=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),archive=verification,optimizer_updates=0,new_scored_forwards=count,new_intermediate_predictions=count-replays,replay_predictions=replays,nll_window_forwards=nll_count,cumulative_task_predictions=8096+count,controller_seconds=controller['seconds'],job_seconds=sum(read(stage/n/'result.json')['seconds'] for n in protocol['runs']),points=points,comparisons=comparisons,tolerances=tolerances)
 dump(out/'result.json',audit);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 fig,axes=plt.subplots(1,3,figsize=(15,4.5))
 for ax,metric,label in zip(axes,['ppl','acc8k','acc16k'],['Heldout perplexity (lower is better)','8K accuracy (%)','16K accuracy (%)']):
  for kind,color in [('dense','#2166ac'),('sparse','#d6604d')]:
   ps=[lookup[kind,s] for s in protocol['steps']];ax.plot([p['seconds'] for p in ps],[p[metric] for p in ps],'-o',color=color,label=kind+' (2-seed mean)')
   offset=(-16 if kind=='dense' else 8) if metric=='ppl' else (6 if kind=='dense' else -13)
   for p in ps:ax.annotate(str(p['step']),(p['seconds'],p[metric]),xytext=(3,offset),textcoords='offset points',fontsize=8)
   for step in protocol['steps']:
    for seed in [0,1]:
     x=records[kind+str(seed),step]
     val=math.exp(x['heldout_mean_nll']) if metric=='ppl' else arrays[kind,step]['long8192' if metric=='acc8k' else 'long16384'][seed].mean()*100
     ax.scatter(x['training_seconds'],val,color=color,alpha=.4,marker='x',s=24)
  ax.set_xlabel('Cumulative measured training seconds');ax.set_ylabel(label);ax.grid(alpha=.25);ax.margins(x=.1,y=.1)
  if metric!='ppl':ax.set_ylim(0,100)
 axes[0].legend(fontsize=8);fig.suptitle('Existing checkpoint trajectories: Qwen2.5-0.5B LoRA, 16K training\nPPL axis zoomed; 96 reused questions per length; steps 2 / 64 / 128 / 256')
 fig.tight_layout();fd=R/'docs/figures';fd.mkdir(exist_ok=True);fig.savefig(fd/'checkpoint-quality-cost-2026-09-16.png',dpi=180);plt.close(fig)
 lines=['# 既有检查点：效果与训练成本的完整曲线','','目标是更低成本达到可接受的接近效果，不要求稀疏超过密集。此前严格NLL门槛仅保留为历史筛查记录，不再作为所有应用的统一否决条件。下面同时保留所有密集、稀疏检查点；允许两者提前停止，避免只给稀疏挑早停点。','',
 '## 全部结果','','两种子均值。PPL越低越好，准确率越高越好；时间为原训练日志中截至该步的同步训练时间，不含本轮评测、安装、搜索或空闲账单。','',
 '|条件|训练步数|训练秒数|语料PPL|8K准确率|16K准确率|','|---|---:|---:|---:|---:|---:|']
 for p in points:lines.append(f"|{p['kind']}|{p['step']}|{p['seconds']:.2f}|{p['ppl']:.4f}|{p['acc8k']:.2f}%|{p['acc16k']:.2f}%|")
 lines+=['','![质量成本曲线](figures/checkpoint-quality-cost-2026-09-16.png)','','## 稀疏相对密集的取舍','','PPL增加百分比与准确率下降百分点是不同量，不能互换。时间节省为负表示更贵。','', '|稀疏步数|密集参照步数|节省训练时间|PPL增加|8K变化pp|16K变化pp|','|---:|---:|---:|---:|---:|---:|']
 for c in comparisons:lines.append(f"|{c['sparse_step']}|{c['dense_step']}|{c['time_saved_pct']:.2f}%|{c['ppl_increase_pct']:+.2f}%|{c['acc8k_gap_pp']:+.2f}|{c['acc16k_gap_pp']:+.2f}|")
 lines+=['','### 描述性配对区间','','共享窗口/问题重采样，先对两个种子平均；以下不是独立新数据上的等价结论。','', '|稀疏/密集步数|PPL增加95%区间|8K变化95%区间pp|16K变化95%区间pp|','|---|---|---|---|']
 for c in comparisons:
  parts=[f"[{c[k][0]:+.2f}, {c[k][1]:+.2f}]" for k in ['nll_descriptive_interval','long8192_descriptive_interval','long16384_descriptive_interval']]
  lines.append(f"|{c['sparse_step']}/{c['dense_step']}|{parts[0]}%|{parts[1]}|{parts[2]}|")
 lines+=['','## 不同容忍幅度下，最便宜的已测点','','参照为密集256步。下列3/5/10% PPL和3/5/10个百分点准确率仅为描述性预算情景，不代表用户已指定唯一阈值，也不是统计等价检验。每项只约束该指标，不能借此声称其他指标也接近。','', '|指标|允许劣化|密集最低成本点|稀疏最低成本点|','|---|---:|---|---|']
 for t in tolerances:
  def fmt(p):return '未覆盖' if p is None else f"{p['step']}步 / {p['seconds']:.2f}秒"
  unit=f"{t['tolerance']*100:.0f}%" if t['metric']=='ppl' else f"{t['tolerance']}pp"
  lines.append(f"|{t['metric']}|{unit}|{fmt(t['cheapest_by_kind']['dense'])}|{fmt(t['cheapest_by_kind']['sparse'])}|")
 lines+=['','## 每个种子的任务计数','','|条件|步数|8K两种子各答对/96|16K两种子各答对/96|','|---|---:|---|---|']
 for p in points:lines.append(f"|{p['kind']}|{p['step']}|{p['correct8k']}|{p['correct16k']}|")
 lines+=['','## 审计、时间与结论边界','',f"全部16检查点完成；本轮{count}次任务计分前向，其中{replays}次为旧256步重放、{count-replays}次为中途检查点的新预测；另{nll_count}次语料窗口前向。新增优化更新0；累计任务{8096+count}次前向。仍只有96道原题，8K/16K是长度变体。",f"控制器经过{controller['seconds']:.2f}秒，各评测作业合计{audit['job_seconds']:.2f}秒；这些是评测开销，不能算进训练速度收益或当成完整账单。",f"证据包SHA256：{proof['sha256']}；{len(expected)}个文件经独立SHA和大小核验。16个检查点哈希、原训练日志前缀时间、3072个选项argmax/标签逐项核对；256步任务选项logit与语料NLL全部在1e-6以内复现。",'模型是密集预训练Qwen2.5-0.5B上的16K LoRA继续适配，并非Qwen4或从头全参数原生稀疏训练。密集LR3e-4，稀疏LR1e-3，各两个初始化种子、相同数据次序；LR选择本身属于既有探索。','2/64/128步沿用256步原学习率计划，仅截断训练轨迹，不是单独优化的短训练计划。图中的前沿仅限这些已测配置；两种子的29语料窗口和96题均被反复观察，不能由事后挑点直接宣布独立确认或统计等价。描述性配对区间保存在audit/result.json中，以窗口/问题为重采样单位并先平均种子，不当成两个独立数据集。','比较的是训练时间；既有推理计时没有证明更快。本轮评测前向计时只是诊断信息。原始多次训练总耗时、所有失败搜索与本轮开销仍须另计；未读取RunPod账单。','']
 content='\n'.join(lines).replace('## 全部结果','## 直观结论\n\n固定256步训练，稀疏平均288.85秒，密集318.21秒，节时9.23%；语料PPL增加6.32%。这可以作为有明确代价的低成本近似结果，不需要稀疏在质量上反超。但8K/16K阅读分别低10.42/10.94个百分点，所以“效果接近”的表述目前应限定到语料指标，并同时披露任务差距。\n\n稀疏从256步减到128步，训练时间从288.85降到144.50秒；PPL从11.5633变成11.6733，约增加0.95%，说明稀疏自身后半程的语料收益较小。它相对密集256步省54.59%、PPL高7.33%，但其中大量节省来自早停，不能全部归因于稀疏。\n\n关键参照：密集64步仅79.29秒、PPL11.0173，也比稀疏128步更便宜且更好。因此本轮证明的是固定训练量下的计算节省与质量折中，尚未建立在两边均可提前停止时的最省钱方案。我们继续寻找收益更大的长度/保留块数组合；不把“必须质量反超”设为门槛，也不省略这一早停参照。\n\n'+'## 全部结果',1)
 (R/'docs/checkpoint-quality-cost-results-2026-09-16.md').write_text(content,encoding='utf-8')
 print(json.dumps(audit,ensure_ascii=True))
if __name__=='__main__':main()
