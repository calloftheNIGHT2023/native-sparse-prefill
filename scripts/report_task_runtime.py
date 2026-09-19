"""Report all four fixed-weight/training-attention cells, without cherry-picking."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path(__file__).resolve().parents[1];E=R/'results/cloud-task-runtime-evidence-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(E/'LOCAL-VERIFICATION.json')['status']=='verified';stage=E/'results/task-runtime-stage-v0';control=read(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0
 original=E/'results/task-quality-stage-v0';models={};paths={}
 for seed in [0,1]:
  for tag,p in [('DD',original/f'dense{seed}'),('SS',original/f'sparse{seed}'),('DS',stage/f'dense_weights_sparse_eval{seed}'),('SD',stage/f'sparse_weights_dense_eval{seed}')]:
   d=read(p/'result.json');assert d['status']=='complete' and d['optimizer_updates']==0 and d['calibration_replay_max_abs_error']<=1e-6
   assert d['k']==(0 if tag[0]=='D' else 16) and d.get('attention_k',d['k'])==(0 if tag[1]=='D' else 16)
   assert d['seed']==2026091560+seed and d['selected_tasks']==['race_mc'] and len(d['predictions'])==384
   assert all(x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold']) for x in d['predictions'])
   models[tag,seed]=d;paths[tag,seed]=p
  for a,b in [('DD','DS'),('SS','SD')]:
   assert models[a,seed]['checkpoint_sha256']==models[b,seed]['checkpoint_sha256']
   assert models[a,seed]['training_result_sha256']==models[b,seed]['training_result_sha256']
   for x,y in zip(models[a,seed]['predictions'],models[b,seed]['predictions']):assert all(x[k]==y[k] for k in ['item_id','variant','length','gold'])
 lines=['# 训练注意力与使用注意力分离：固定权重诊断','','这是在阅读理解正式结果揭示长输入退步后开展的事后诊断，同一批题已被观察，不能叫独立确认。本轮0个训练更新，仅切换前向注意力，原模型断点与提示不变。','','D=密集，S=K16稀疏。单元格为两种子平均正确率，96题/输入条件。','','|输入|D训练/D使用|D训练/S使用|S训练/D使用|S训练/S使用|','|---|---:|---:|---:|---:|']
 results=[];contrasts=[];depths=[]
 def rows(tag,seed,variant):return {x['item_id']:x for x in models[tag,seed]['predictions'] if x['variant']==variant}
 for variant in ['short','no_context','long8192','long16384']:
  metrics={};arrays={}
  for tag in ['DD','DS','SD','SS']:
   ar=[]
   for seed in [0,1]:
    x=rows(tag,seed,variant);assert len(x)==96;ar.append(np.array([x[i]['correct'] for i in sorted(x)],dtype=float))
   arrays[tag]=np.stack(ar);metrics[tag]=float(arrays[tag].mean())
  results.append(dict(variant=variant,accuracy=metrics,seed_accuracy={tag:arrays[tag].mean(1).tolist() for tag in arrays}))
  lines.append('|'+variant+'|'+'|'.join(f'{100*metrics[tag]:.2f}%' for tag in ['DD','DS','SD','SS'])+'|')
  for a,b in [('SD','DD'),('SD','SS'),('DS','DD'),('SS','DS')]:
   diff=(arrays[a]-arrays[b]).mean(0);rng=np.random.default_rng(2026091601);samples=rng.integers(0,96,(10000,96));interval=100*np.quantile(diff[samples].mean(1),[.025,.975])
   contrasts.append(dict(variant=variant,comparison=a+' minus '+b,gap_pp=100*float(diff.mean()),descriptive_95pct_interval_pp=interval.tolist(),supports_5pp_descriptively=bool(interval[0]>-5 and (arrays[b].mean(1)>.25).all())))
  if variant.startswith('long'):
   for depth in [0,1,2]:
    mask=np.array([i%3==depth for i in range(96)]);depths.append(dict(variant=variant,filler_fraction=[.1,.5,.9][depth],accuracy={tag:float(arrays[tag][:,mask].mean()) for tag in arrays}))
 lines+=['','主要比较：S训练/D使用对D训练/D使用，检验稀疏训练所得参数在密集使用时是否保留任务效果。其次比较S训练的同一权重在D/S使用下的差别。','','|输入|S训练/D使用 − D训练/D使用(pp)|描述性95%区间(pp)|5pp范围证据|','|---|---:|---|---|']
 for c in contrasts:
  if c['comparison']=='SD minus DD':lines.append(f"|{c['variant']}|{c['gap_pp']:+.2f}|[{c['descriptive_95pct_interval_pp'][0]:+.2f}, {c['descriptive_95pct_interval_pp'][1]:+.2f}]|{'支持（描述性）' if c['supports_5pp_descriptively'] else '不足'}|")
 lines+=['','## 按关键段落位置拆分','','|输入|无关文本中插入位置|D/D|D/S|S/D|S/S|','|---|---|---:|---:|---:|---:|']
 for d in depths:lines.append(f"|{d['variant']}|{100*d['filler_fraction']:.0f}%|"+'|'.join(f"{100*d['accuracy'][t]:.2f}%" for t in ['DD','DS','SD','SS'])+'|')
 lines+=['','每位置仅32题；区间按题配对抽样并共同保留两个种子，不能把同题不同种子当独立题。模板使用明确标记的TARGET段落，不等于一般长文本理解。两个种子的训练数据顺序相同。看到结果后切换模式属于机制探索，需要新题和更多模型确认。',
  '', '原来的训练耗时约少9.2%仍只适用于那两条256步LoRA轨迹。稀疏训练+dense使用即使保留质量，也不意味着推理加速；本轮不重新宣称prefill收益。D/S对D/D可观察使用稀疏路由本身的影响；S/S对D/S还同时包含此前适配方式和峰值学习率的差异，不能只归因于某个训练机制。',
  '', '## UTC时间轴','','|任务|开始|结束|退出码|','|---|---|---|---:|']
 for x in control['runs']:lines.append(f"|{x['name']}|{x['started_utc']}|{x['finished_utc']}|{x['returncode']}|")
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,scope='Post-hoc fixed-weight attention intervention on previously observed task examples',results=results,contrasts=contrasts,depths=depths,seconds=control['seconds'],inputs=[dict(path=p.relative_to(R).as_posix(),sha256=sha(p/'result.json')) for p in paths.values()])
 out=R/'results/task-runtime-audit-v0';out.mkdir(exist_ok=False);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());(R/'docs/task-runtime-results-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='complete',results=results,contrasts=[x for x in contrasts if x['comparison']=='SD minus DD'])))
if __name__=='__main__':main()
