"""Audit fixed-model evaluation and paired article uncertainty on fresh articles."""
import json,hashlib,tarfile,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 a=R/'exports/fresh256-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
 dest=R/'results/cloud-fresh256-evidence-v0';dest.mkdir(exist_ok=True)
 with tarfile.open(a) as t:
  entries=json.loads(t.extractfile('fresh256-manifest.json').read())['files'];names=[m.name for m in t.getmembers()]
  assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={x['path'] for x in entries}|{'fresh256-manifest.json'}
  for e in entries:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   raw=t.extractfile(m).read();assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 p=load(dest/'provenance/fresh256-confirmation-protocol.json');assert sha(dest/'provenance/fresh256-confirmation-protocol.json')==sha(R/'provenance/fresh256-confirmation-protocol.json')
 for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h
 stage=dest/'results/fresh256-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and len(control['jobs'])==6 and control['task_predictions']==4608 and control['scientific_updates']==control['diagnostic_updates']==0
 meta=load(dest/'data/fresh256-confirmation-v0/tasks.json');manifest=load(dest/'data/fresh256-confirmation-v0/manifest.json');assert len(meta)==768
 articles={x['article_hash'] for x in meta};assert len(articles)==256 and articles.isdisjoint(manifest['excluded_article_hashes'])
 previous=set()
 for name in ['data/task-quality-v0/pilot.json','data/task-quality-v0/formal.json','data/32k-adaptation-v0/tasks.json','data/32k-continuation-tasks-v0/tasks.json']:
  previous.update(x['article_hash'] for x in load(R/name) if 'article_hash' in x)
 assert articles.isdisjoint(previous)
 tokens=np.load(dest/'data/fresh256-confirmation-v0/tasks.npz');assert len(tokens['offsets'])==769 and tokens['offsets'][-1]==len(tokens['input_ids'])
 for i,x in enumerate(meta):assert int(tokens['offsets'][i+1]-tokens['offsets'][i])==x['length']
 parent=R/'results/cloud-expanded76-evidence-v0/results/expanded76-stage-v0';pa=load(R/'results/expanded76-audit-v0/result.json');assert pa['status']=='verified' and pa['control']['protocol_sha256']==p['parent_protocol_sha256']
 locks=load(stage/'checkpoint-lock.json')['models'];assert [[m['k'],m['seed'],m['step']] for m in locks]==p['models']
 records=[];arrays={};variants=p['variants'];item_order={t:[x['item_id'] for x in meta if x['variant']==t] for t in variants}
 assert item_order['short']==item_order['no_context']==item_order['long32768'] and len(set(item_order['short']))==256
 for j,m in zip(control['jobs'],locks):
  assert j['status']=='complete' and j['returncode']==0 and all(j[k]==m[k] for k in ['k','seed','step','path','sha256'])
  k,seed,step=j['k'],j['seed'],j['step'];d=stage/j['name'];v=load(d/'result.json');cp=parent/f'k{k}-seed{seed}/checkpoint-{step}.pt';assert sha(cp)==m['sha256']
  assert v['status']=='complete' and v['step']==step and v['optimizer_updates_this_process']==0 and not v['trace'] and sha(d/'source.py')==p['source_sha256']['scripts/eval_fresh256.py']
  original=load(parent/f'report-k{k}-seed{seed}-step{step}/result.json');assert v['identity']==original['identity'] and v['environment']==original['environment']
  assert v['evaluations'][0]['values']==original['evaluations'][0]['values'],'WikiText same-checkpoint replay changed'
  rep=load(d/'replay-verification.json');assert rep['max_abs_error']<=1e-6 and rep['checkpoint_sha256']==m['sha256']
  pred=load(d/'task-results.json');assert pred['checkpoint_sha256']==m['sha256'] and pred['task_metadata_sha256']==manifest['task_metadata_sha256'] and pred['task_tokens_sha256']==manifest['task_tokens_sha256']
  assert len(pred['predictions'])==768 and pred['predictions']==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
  by={t:[] for t in variants}
  for x,y in zip(pred['predictions'],meta):
   assert all(x[z]==y[z] for z in ['item_id','variant','gold']) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
   assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']);by[x['variant']].append(int(x['correct']))
  arrays[(k,seed,step)]={t:np.array(by[t],dtype=float) for t in variants}
  records.append(dict(k=k,seed=seed,step=step,accuracy={t:float(np.mean(by[t])) for t in variants},correct={t:sum(by[t]) for t in variants},questions=256))
 seeds=[2026091660,2026091661];rng=np.random.default_rng(2026091696);indices=rng.integers(0,256,size=(20000,256));comparisons={}
 for t in variants:
  diff=np.mean([arrays[(32,s,128)][t]-arrays[(0,s,128)][t] for s in seeds],axis=0)
  comparisons[t]=dict(delta_pp=float(100*diff.mean()),paired_article_ci95_pp=(100*np.quantile(diff[indices].mean(axis=1),[.025,.975])).tolist(),discordant_articles=int(np.count_nonzero(diff)))
 aggregates=[]
 for k,step in [(0,0),(32,0),(0,128),(32,128)]:
  subset=[x for x in records if x['k']==k and x['step']==step]
  acc={t:float(np.mean([x['accuracy'][t] for x in subset])) for t in variants};aggregates.append(dict(k=k,step=step,accuracy=acc,long_minus_no_context_pp=100*(acc['long32768']-acc['no_context'])))
 result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,records=records,aggregates=aggregates,comparisons=comparisons,parent_tradeoff=pa['tradeoff'],source_split_counts=p['source_split_counts'],scope=p['scope'],uncertainty_scope=p['analysis'])
 save(R/'results/fresh256-audit-v0/result.json',result)
 lines=['# 新256篇文章：固定模型的独立题目确认','','256篇文章与此前项目评测文章按内容哈希去重。来源为RACE middle验证集213篇、测试集43篇；没有用于本项目LoRA训练，但原预训练污染未知。每篇短文、无原文、32K填充上下文三个变体。该任务是带TARGET标记的扩展，不是官方RACE分数。','', '|模型|步数|短文|无原文|32K|32K减无原文|','|---|---:|---:|---:|---:|---:|']
 for x in aggregates:
  ac=x['accuracy'];lines.append(f"|{'dense' if x['k']==0 else 'K32'}|{x['step']}|{ac['short']:.2%}|{ac['no_context']:.2%}|{ac['long32768']:.2%}|{x['long_minus_no_context_pp']:+.2f}pp|")
 lines+=['','|128步稀疏减密集|准确率变化|配对文章95%区间|','|---|---:|---|']
 for t,v in comparisons.items():lines.append(f"|{t}|{v['delta_pp']:+.2f}pp|[{v['paired_article_ci95_pp'][0]:+.2f}, {v['paired_article_ci95_pp'][1]:+.2f}]pp|")
 tr=pa['tradeoff'];lines+=['',f"这些模型的训练成本来自已审计父实验：同128步训练节时{tr['training_time_saving_percent']:.2f}%，WikiText PPL代价{tr['ppl_cost_percent']:.2f}%。本轮只评测，不增加训练更新。",'', '每篇文章内先对两颗初始化种子的差值取平均，再按文章进行20000次配对bootstrap。区间条件于这两颗种子，不代表广泛模型/任务不确定性；区间包含0也不能证明统计等价。始终报告零更新模型和无原文对照；如果长文不优于无原文，本任务不足以证明长距离证据利用。','',f"原始包{proof['files']}文件逐一核验；4608次任务计分前向，6模型。所有父检查点SHA、旧WikiText逐窗口损失复现、校准重放和新题argmax已核验。SHA {proof['sha256']}。",'', '## 时间轴','','UTC。','','|作业|开始|结束|','|---|---|---|']
 for j in control['jobs']:lines.append(f"|{j['name']}|{j['started_utc']}|{j['finished_utc']}|")
 (R/'docs/fresh256-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='verified',comparisons=comparisons)))
if __name__=='__main__':main()
