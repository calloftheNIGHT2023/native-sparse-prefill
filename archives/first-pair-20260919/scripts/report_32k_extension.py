"""Verify and summarize the queued 128->256 extension; requires parent local audit."""
import hashlib,json,math,tarfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 a=R/'exports/32k-extension-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
 dest=R/'results/cloud-32k-extension-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  entries=json.loads(t.extractfile('32k-extension-manifest.json').read())['files'];members=t.getmembers()
  assert len(entries)==proof['files'] and len({m.name for m in members})==len(members)
  assert {m.name for m in members}=={e['path'] for e in entries}|{'32k-extension-manifest.json'}
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   b=t.extractfile(m).read();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
   f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
 p=load(dest/'provenance/32k-extension-protocol.json');psha=sha(dest/'provenance/32k-extension-protocol.json')
 intent=load(R/'provenance/32k-extension-intent.json');assert sha(R/'provenance/32k-extension-intent.json')==p['intent_sha256']
 assert p['source_sha256']==intent['source_sha256']
 for n,h in p['source_sha256'].items():assert sha(dest/n)==h
 parent=R/'results/cloud-32k-continuation-evidence-v0';old=parent/'results/32k-continuation-stage-v0';parentaudit=load(R/'results/32k-continuation-audit-v0/result.json');assert parentaudit['status']=='complete'
 parentlock=load(old/'report-lock.json');assert intent['frozen_utc']<parentlock['created_utc']
 gains=[]
 for seed in p['seeds']:
  v=load(old/f'k32-seed{seed}'/'result.json');es={x['step']:x['mean_nll'] for x in v['evaluations'] if x['split']=='calibration'};gains.append(es[96]-es[128])
 assert gains==p['gate']['gains_96_to_128'] and min(gains)>0 and np.mean(gains)>=intent['minimum_mean_nll_gain']
 stage=dest/'results/32k-extension-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and len(control['jobs'])==12 and all(x['returncode']==0 for x in control['jobs'])
 assert min(j['started_utc'] for j in control['jobs'])>=load(old/'result.json')['finished_utc']
 import torch
 torch.set_num_threads(4);trains={};count=0;order=np.random.default_rng(2026091662).permutation(32).tolist()
 for key,j in p['parents'].items():
  cp=parent/j['checkpoint'];assert sha(cp)==j['checkpoint_sha256'];d=stage/key;v=load(d/'result.json');prior=load(old/key/'result.json')
  assert v['status']=='complete' and v['step']==256 and v['optimizer_updates_this_process']==128 and v['trace'][:128]==prior['trace']
  assert v['initial_sha256']==prior['initial_sha256'] and len(v['trace'])==256
  assert [x['step'] for x in v['trace']]==list(range(1,257))
  for x in v['trace'][128:]:assert x['lr']==.0001 and x['window_index']==order[(x['step']-1)%32] and math.isfinite(x['train_nll']) and x['seconds']>0
  assert abs(sum(x['seconds'] for x in v['trace'])-v['training_seconds'])<1e-8
  ver=load(d/'parent-verification.json');assert ver['replay_error']<=1e-6 and ver['parent_identity']==prior['identity']
  assert sha(d/'source.py')==p['source_sha256']['scripts/run_32k_extension.py']
  for f in d.glob('checkpoint-*.pt'):
   c=torch.load(f,map_location='cpu',weights_only=False);n=c['step'];assert n==c['data_cursor']==c['scheduler']['last_epoch'] and c['identity']==v['identity']
   assert c['extra']['rows']==v['trace'][:n] and all(torch.isfinite(t).all() for t in c['params'].values())
   assert c['optimizer']['param_groups'][0]['lr']==.0001 and all(float(t['step'])==n for t in c['optimizer']['state'].values())
   assert c['rng']['cuda'] is not None;count+=1
  if key.startswith('k32'):
   cut=load(d/'time-budget-cut.json');n=cut['compliant_step'];assert sum(x['seconds'] for x in v['trace'][:n])<=j['dense128_training_seconds']<sum(x['seconds'] for x in v['trace'][:n+1])
   assert (d/f'checkpoint-{n}.pt').exists() and (d/f'checkpoint-{n+1}.pt').exists()
  trains[key]=v
 meta=load(dest/p['task_data_path']/'tasks.json');lock=load(stage/'report-lock.json');assert lock['protocol_sha256']==psha
 reports={};predictions=0
 for t in lock['reports']:
  d=stage/t['name'];v=load(d/'result.json');assert v['status']=='complete' and v['optimizer_updates_this_process']==0 and v['identity']==t['identity']
  cp=(dest if t['origin']=='continuation' else parent)/t['checkpoint'];assert sha(cp)==t['checkpoint_sha256']
  rep=load(d/'replay-verification.json');assert rep['max_abs_error'] is None or rep['max_abs_error']<=1e-6
  task=load(d/'task-results.json');assert task['task_metadata_sha256']==p['task_metadata_sha256'] and task['task_tokens_sha256']==p['task_tokens_sha256']
  assert len(task['predictions'])==192
  for q,m in zip(task['predictions'],meta):
   assert (q['item_id'],q['variant'],q['gold'])==(m['item_id'],m['variant'],m['gold'])
   assert q['prediction']==int(np.argmax(q['choice_logits'])) and q['correct']==(q['prediction']==q['gold']) and np.isfinite(q['choice_logits']).all()
  es=[x for x in v['evaluations'] if x['split']=='report'];assert len(es)==1 and len(es[0]['values'])==9
  if t['origin']=='parent':
   previous=load(old/t['name']/'result.json');e=next(x for x in previous['evaluations'] if x['split']=='report');assert np.max(np.abs(np.array(e['values'])-es[0]['values']))<=1e-6
   assert load(old/t['name']/'task-results.json')['predictions']==task['predictions']
  scores={var:[int(q['correct']) for q in task['predictions'] if q['variant']==var] for var in ['short','no_context','long32768']}
  seconds=sum(x['seconds'] for x in trains[f"k{t['k']}-seed{t['seed']}"]['trace'][:t['step']])
  reports[t['name']]=dict(step=t['step'],seconds=seconds,nll=es[0]['values'],scores=scores);predictions+=192
 assert predictions==1536
 groups={}
 for tag,k,kind in [('dense128',0,'step128'),('dense256',0,'step256'),('sparse_budget',32,'budget'),('sparse256',32,'step256')]:
  vs=[reports[f'report-k{k}-seed{s}-{kind}'] for s in p['seeds']]
  groups[tag]=dict(steps=[v['step'] for v in vs],seconds=float(np.mean([v['seconds'] for v in vs])),ppl=math.exp(float(np.mean([v['nll'] for v in vs]))),accuracy={var:100*float(np.mean([v['scores'][var] for v in vs])) for var in ['short','no_context','long32768']},correct={var:[sum(v['scores'][var]) for v in vs] for var in ['short','no_context','long32768']})
 comparisons={}
 rng=np.random.default_rng(2026091695);ix9=rng.integers(0,9,(10000,9));ix64=rng.integers(0,64,(10000,64))
 for label,sparse,dense in [('equal_tokens256','step256','step256'),('near_equal_time','budget','step128')]:
  ss=[reports[f'report-k32-seed{s}-{sparse}'] for s in p['seeds']];ds=[reports[f'report-k0-seed{s}-{dense}'] for s in p['seeds']]
  diff=np.mean([x['nll'] for x in ss],axis=0)-np.mean([x['nll'] for x in ds],axis=0)
  c=dict(training_saving_percent=100*(1-np.mean([x['seconds'] for x in ss])/np.mean([x['seconds'] for x in ds])),ppl_increase_percent=100*float(np.expm1(diff.mean())),ppl_ci95=(100*np.percentile(np.expm1(diff[ix9].mean(axis=1)),[2.5,97.5])).tolist(),tasks={})
  for var in ['short','no_context','long32768']:
   diff=np.mean([x['scores'][var] for x in ss],axis=0)-np.mean([x['scores'][var] for x in ds],axis=0)
   c['tasks'][var]=dict(delta_pp=100*float(diff.mean()),ci95=(100*np.percentile(diff[ix64].mean(axis=1),[2.5,97.5])).tolist())
  comparisons[label]=c
 curves={key:[dict(step=e['step'],calibration_nll=e['mean_nll'],calibration_ppl=math.exp(e['mean_nll']),seconds=sum(x['seconds'] for x in v['trace'][:e['step']])) for e in v['evaluations'] if e['split']=='calibration'] for key,v in trains.items()}
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),archive=proof,verified_checkpoints=count,new_updates=512,new_independent_initializations=0,continued_trajectories=4,new_target_exposures=512*32768,cumulative_amp_updates=4992,cumulative_amp_diagnostic_updates=55,cumulative_task_predictions=15776,task_predictions=1536,training_seconds_this_stage=sum(sum(x['seconds'] for x in v['trace'][128:]) for v in trains.values()),controller_seconds=control['seconds'],groups=groups,comparisons=comparisons,calibration_curves=curves,reports=reports,gate_gains=gains,parent_zero_update_baselines={k:parentaudit['groups'][k] for k in ['baseline0','baseline32']},jobs=control['jobs'])
 save(R/'results/32k-extension-audit-v0/result.json',result);save(dest/'LOCAL-VERIFICATION.json',dict(status='verified',archive=proof,checkpoints=count,updates=512,predictions=1536))
 lines=['# 32K续训到256步：成本与质量结果','',f"完成UTC：{control['finished_utc']}。本轮继续四条既有轨迹128→256，共512新更新；LR继续0.0001、原32个窗口总共循环8遍。校准启动规则在128步新题评测前冻结；制定时已有部分校准曲线可见，属于校准驱动的自适应探索，不根据最终任务结果决定延长。",'', '|条件|步数|累计训练秒|PPL|短题|无原文|32K|','|---|---|---:|---:|---:|---:|---:|']
 for n,g in groups.items():lines.append(f"|{n}|{g['steps']}|{g['seconds']:.2f}|{g['ppl']:.4f}|{g['accuracy']['short']:.2f}%|{g['accuracy']['no_context']:.2f}%|{g['accuracy']['long32768']:.2f}%|")
 lines+=['','## 成本折中','']
 for n,c in comparisons.items():lines.append(f"- {n}：训练节时{c['training_saving_percent']:.2f}%，PPL增加{c['ppl_increase_percent']:.2f}%（描述性95% {c['ppl_ci95']}），32K变化{c['tasks']['long32768']['delta_pp']:+.2f}pp（{c['tasks']['long32768']['ci95']}）。")
 lines+=['','时间按历史父轨迹+本轮同步训练循环累计，不含加载/校准/存档/评测/空闲账单。等时点选择最后一个不超过对应密集128预算的完整更新。','', '这批RACE64与WikiText均在前阶段评测过，本轮是复用探索；两个种子、64问题和9相邻窗口不能单独证明统计等价。保留前阶段零更新基线，不要求稀疏质量反超，但成本优势需连同质量代价报告。模型仍为密集预训练Qwen2.5-0.5B的LoRA适配，不是Qwen4/从零原生全参数预训练。','', '## 校准轨迹','', '|轨迹|步数|累计训练秒|校准PPL|','|---|---:|---:|---:|']
 for key,curve in curves.items():
  for e in curve:lines.append(f"|{key}|{e['step']}|{e['seconds']:.2f}|{e['calibration_ppl']:.4f}|")
 lines+=['','## 审计与保存','',f"证据包{proof['files']}文件、{count}检查点逐项审计。父断点校准重放≤1e-6；两个密集128参照的语料与任务预测复现前阶段。新增512科学更新、1536任务计分前向；累计4992科学更新、55诊断更新、15776计分前向。新增独立初始化为0。",'',f"本轮新增训练循环{result['training_seconds_this_stage']:.2f}秒、控制器{result['controller_seconds']:.2f}秒；实际账单未知。原始包SHA `{proof['sha256']}`。",'', '机器结果results/32k-extension-audit-v0/result.json；原始证据exports/32k-extension-evidence-v0.tar.gz；父报告docs/32k-continuation-results-2026-09-16.md。']
 (R/'docs/32k-extension-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='verified',groups=groups,comparisons=comparisons,checkpoints=count)))
if __name__=='__main__':main()
