"""Compare audited stages on identical evaluation items; exploratory, not a new run."""
import json,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sources={n:R/f'results/{n}-audit-v0/result.json' for n in ['32k-continuation','32k-extension','expanded76']}
 old,ext,new=[load(sources[n]) for n in sources]
 assert old['status']=='complete' and ext['status']=='complete' and new['status']=='verified'
 meta=load(R/'data/32k-continuation-tasks-v0/tasks.json');assert sha(R/'data/32k-continuation-tasks-v0/tasks.json')==sha(R/'data/32k-expanded-training-v0/tasks.json')
 rows=[];scores={};seeds=[2026091660,2026091661];variants=['short','no_context','long32768']
 for k in [0,32]:
  group=old['groups']['dense128' if k==0 else 'sparse128'];newgroup=next(x for x in new['summary'] if x['k']==k and x['step']==128)
  rows.append(dict(windows=32,k=k,steps=128,train_seconds=group['training_seconds'],ppl=group['ppl'],accuracy=group['accuracy']))
  rows.append(dict(windows=76,k=k,steps=128,train_seconds=newgroup['train_seconds'],ppl=newgroup['ppl'],accuracy=newgroup['accuracy']))
  for seed in seeds:
   scores[(32,k,seed)]=old['reports'][f'report-k{k}-seed{seed}-step128']['scores']
   pred=load(R/f'results/cloud-expanded76-evidence-v0/results/expanded76-stage-v0/report-k{k}-seed{seed}-step128/task-results.json')['predictions']
   assert len(pred)==len(meta)
   assert all(x['item_id']==y['item_id'] and x['variant']==y['variant'] and x['gold']==y['gold'] for x,y in zip(pred,meta))
   scores[(76,k,seed)]={t:[int(x['correct']) for x in pred if x['variant']==t] for t in variants}
 rng=np.random.default_rng(2026091698);indices=rng.integers(0,64,size=(20000,64));contrasts=[]
 def summarize(name,variant,diff):
  contrasts.append(dict(name=name,variant=variant,delta_pp=float(100*diff.mean()),paired_article_ci95_pp=(100*np.quantile(diff[indices].mean(axis=1),[.025,.975])).tolist()))
 for t in variants:
  means={(w,k):np.mean([scores[(w,k,s)][t] for s in seeds],axis=0) for w in [32,76] for k in [0,32]}
  summarize('dense:76-minus32',t,means[(76,0)]-means[(32,0)])
  summarize('sparse:76-minus32',t,means[(76,32)]-means[(32,32)])
  summarize('sparse-minus-dense:76',t,means[(76,32)]-means[(76,0)])
  summarize('change-in-gap:76-minus32',t,(means[(76,32)]-means[(76,0)])-(means[(32,32)]-means[(32,0)]))
 # Plot selected audited endpoints, with zero-update reference retained.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.3));colors={0:'#285A9F',32:'#CD673B'}
 for k in [0,32]:
  b=old['groups']['baseline0' if k==0 else 'baseline32'];g128=next(x for x in rows if x['windows']==32 and x['k']==k);g256=ext['groups']['dense256' if k==0 else 'sparse256'];g76=next(x for x in rows if x['windows']==76 and x['k']==k)
  xs=[0,g128['train_seconds'],g256['seconds']];ys=[b['ppl'],g128['ppl'],g256['ppl']];acc=[b['accuracy']['long32768']*100,g128['accuracy']['long32768']*100,g256['accuracy']['long32768']]
  label='Dense' if k==0 else 'Sparse K32'
  for ax,values,value in [(axes[0],ys,g76['ppl']),(axes[1],acc,g76['accuracy']['long32768']*100)]:
   ax.plot(xs,values,'o-',color=colors[k],label=label+' / 32 windows')
   ax.scatter([g76['train_seconds']],[value],s=100,marker='D',color=colors[k],edgecolors='black',label=label+' / 76 windows, 128 steps',zorder=5)
   ax.set_xlabel('Training-loop time per seed (seconds)');ax.grid(alpha=.22)
 axes[0].set_ylabel('WikiText perplexity (lower is better)');axes[1].set_ylabel('32K task accuracy (%)');axes[0].set_title('Fixed evaluation text');axes[1].set_title('Same 64 articles reused across stages')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8);fig.suptitle('Measured training cost and quality: retain every fixed endpoint');fig.tight_layout(rect=[0,.14,1,.94])
 folder=R/'docs/figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'expanded76-comparison-2026-09-16.png',dpi=180);fig.savefig(folder/'expanded76-comparison-2026-09-16.pdf');plt.close(fig)
 out=dict(utc=datetime.now(timezone.utc).isoformat(),sources={n:dict(path=p.relative_to(R).as_posix(),sha256=sha(p)) for n,p in sources.items()},rows=rows,contrasts=contrasts,scope='Post-hoc descriptive analysis of identical reused64 articles, conditioned on2 reused seeds. Training coverage and sample order both differ; no pure causal coverage conclusion. No new GPU work.')
 target=R/'results/expanded76-comparison-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['# 扩大文本后的变化：同128步对照','','本报告只分析已核验的结果，不增加训练或新题预测。所有表格使用同一批64篇RACE文章和相同WikiText评测格式。','', '|训练窗口|方法|步数|训练秒|PPL|短文|无原文|32K|','|---|---|---:|---:|---:|---:|---:|---:|']
 for x in sorted(rows,key=lambda x:(x['windows'],x['k'])):
  a=x['accuracy'];lines.append(f"|{x['windows']}|{'dense' if x['k']==0 else 'K32'}|128|{x['train_seconds']:.2f}|{x['ppl']:.4f}|{a['short']:.2%}|{a['no_context']:.2%}|{a['long32768']:.2%}|")
 lines+=['','在相同128步下，稀疏长题只下降0.78125个百分点，密集提高4.6875个百分点。差距从7.03125扩大到12.5个百分点，主要由密集成绩提高造成。不能把旧32窗口256步的最好差距和新76窗口128步直接当成同训练量比较。','', 'PPL方面，两种方法的绝对值都略有改善，密集改善更多；更大的相对质量代价不等于稀疏模型本身的PPL变差。训练循环节时仍约17%。','', '|同题配对变化（32K）|变化|文章bootstrap95%区间|','|---|---:|---|']
 for v in contrasts:
  if v['variant']=='long32768':lines.append(f"|{v['name']}|{v['delta_pp']:+.2f}pp|[{v['paired_article_ci95_pp'][0]:+.2f},{v['paired_article_ci95_pp'][1]:+.2f}]pp|")
 lines+=['','每篇先平均两颗种子的变化，再按64篇文章配对重采样20000次。这是事后描述性分析，区间条件于这两颗种子；训练覆盖和顺序一起变化，不能严格归因于新增44个窗口。','', '更关键的限制：零更新密集模型长题50%，高于当前两种训练模型；无原文成绩也高于32K成绩。因此当前任务不能单独证明长期上下文利用良好，也不能忽略不训练或较早停止的便宜参照。','', '正在进行的256篇新文章评测固定了模型和题目，会独立报告结果；这里不读取其部分预测，也不因该组新题成绩挑模型。','', '图：figures/expanded76-comparison-2026-09-16.png；PDF同名。完整机器对照results/expanded76-comparison-v0/result.json。']
 (R/'docs/expanded76-comparison-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='written',long_contrasts=[x for x in contrasts if x['variant']=='long32768'])))
if __name__=='__main__':main()
