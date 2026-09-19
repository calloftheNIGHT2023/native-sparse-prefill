"""Recompute calibration trajectories from immutable downloaded evidence."""
import hashlib,json,math,statistics
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
S=R/'results/cloud-amp-recovery-final-evidence-v0/results/amp-recovery-stage-v2'
O=R/'results/amp-recovery-curve-audit-v0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 O.mkdir(exist_ok=False);runs=[];inputs=[]
 for p in sorted(S.glob('*/result.json')):
  d=read(p)
  if d.get('phase')!='train':continue
  assert d['status']=='complete' and len(d['trace'])==256
  assert [e['step'] for e in d['evaluations']]==[0,64,128,256]
  assert all(e['split']=='calibration' for e in d['evaluations'])
  assert all(abs(statistics.mean(e['values'])-e['mean_nll'])<1e-12 for e in d['evaluations'])
  orders=[tuple(x['window_index'] for x in d['trace'][i:i+64]) for i in range(0,256,64)]
  assert len(set(orders))==1 and len(set(orders[0]))==64
  ev=d['evaluations'];trace=d['trace'];t=sum(x['seconds'] for x in trace)
  assert abs(t-d['training_seconds'])<1e-8
  row=dict(name=p.parent.name,**{k:d['identity'][k] for k in ['k','seed','lr']},
    calibration=[e['mean_nll'] for e in ev],calibration_values=[e['values'] for e in ev],
    train_epoch_mean=[statistics.mean(x['train_nll'] for x in trace[i:i+64]) for i in range(0,256,64)],
    training_seconds=t,final_used_lr=trace[-1]['lr'],
    last128_calibration_improvement=ev[2]['mean_nll']-ev[3]['mean_nll'])
  runs.append(row);inputs.append(dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 assert len(runs)==12
 pairs=[]
 for seed in sorted({r['seed'] for r in runs}):
  dense=next(r for r in runs if r['k']==0 and r['seed']==seed and r['lr']==.0003)
  for k in [4,16]:
   r=next(r for r in runs if r['k']==k and r['seed']==seed and r['lr']==.0003)
   gaps=[a-b for a,b in zip(r['calibration'],dense['calibration'])]
   ratio=r['training_seconds']/dense['training_seconds']
   pairs.append(dict(k=k,seed=seed,calibration_gap=gaps,
      initial_gap_reduction_fraction=1-gaps[-1]/gaps[0],
      gap_shrink_last128=gaps[2]-gaps[3],time_ratio=ratio,
      approximate_extra_steps_before_time_break_even=256/ratio-256,
      approximate_extra_steps_preserving_5pct_time_saving=.95*256/ratio-256))
 fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors={0:'#374151',4:'#d97706',16:'#2563eb'}
 for r in runs:
  label='dense' if r['k']==0 else f"K{r['k']}"
  if r['lr']==.0003:
   style='-' if r['seed']==2026091560 else '--'
   ax[0,0].plot([0,64,128,256],r['calibration'],style+'o',color=colors[r['k']],label=label if style=='-' else None)
   ax[1,1].plot([1,2,3,4],r['train_epoch_mean'],style+'o',color=colors[r['k']],label=label if style=='-' else None)
 for p in pairs:
  style='-' if p['seed']==2026091560 else '--'
  ax[0,1].plot([64,128,256],p['calibration_gap'][1:],style+'o',color=colors[p['k']],label=f"K{p['k']}" if style=='-' else None)
 ax[0,1].axhline(.03,color='gray',linestyle=':',label='0.03 exploratory margin')
 for k in [0,4,16]:
  rs=sorted((r for r in runs if r['seed']==2026091560 and r['k']==k),key=lambda r:r['lr'])
  ax[1,0].semilogx([r['lr'] for r in rs],[r['calibration'][-1] for r in rs],'o-',color=colors[k],label='dense' if k==0 else f'K{k}')
 for a,title,xlabel,ylabel in zip(ax.flat,['Calibration loss, selected LR','Gap to matched dense (after step 64)','Learning-rate boundary, seed 0','Online train loss on repeated window order'],['Updates','Updates','Peak learning rate','Epoch'],['NLL (lower is better)','NLL gap','Step-256 calibration NLL','Mean online NLL']):
  a.set(title=title,xlabel=xlabel,ylabel=ylabel);a.grid(alpha=.2);a.legend(fontsize=8)
 fig.suptitle('0.5B LoRA, 16K, calibration only; solid/dashed = two initialization seeds')
 fig.savefig(O/'curves.png',dpi=180);fig.savefig(O/'curves.pdf');plt.close(fig)
 result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),scope='Calibration-only post-hoc diagnostic. No report data loaded; no new optimizer updates.',runs=runs,pairs=pairs,inputs=inputs)
 (O/'result.json').write_text(json.dumps(result,indent=2)+'\n');(O/'source.py').write_bytes(Path(__file__).read_bytes())
 lines=['# AMP训练曲线复算与下一轮决策','','仅读取12条已归档训练轨迹和校准数据；不读取report，不产生新训练更新。以下均为开发集诊断，不是最终测试结论。','','|方法|种子|初始差距|64步差距|128步差距|256步差距|最后128步差距缩小|','|---|---|---:|---:|---:|---:|---:|']
 for p in pairs:lines.append(f"|K{p['k']}|{p['seed']}|"+'|'.join(f'{x:.6f}' for x in p['calibration_gap'])+f"|{p['gap_shrink_last128']:.6f}|")
 lines+=['','两个种子都继续缩小差距，因此不能把256步未达标解释为不可恢复。K16从初始约0.531降至0.074，K4从约1.343降至0.149；恢复的是相对同阶段密集基线的NLL差，不是准确率。训练集四轮也继续改善；这是每步更新前的在线损失均值，不能与校准NLL直接作过拟合差值。',
 '','然而，末段学习率降至峰值的约10%，而最优峰值仍是网格上界。后期变慢不能证明收敛，也不能据此线性预测还需要多少步。四个校准窗口和两个共享数据顺序的初始化种子不足以做可靠泛化区间。',
 '','## 时间账不能忽略','','假定后续每步时间保持当前均值，密集基线固定在256步；这一计算只用于预算，没有声称额外步数会达到质量门槛。若密集也继续训练，应重新比较双方完整轨迹。','','|方法|种子|不超过密集耗时可额外训练约几步|保留至少5%节时可额外训练约几步|','|---|---|---:|---:|']
 for p in pairs:lines.append(f"|K{p['k']}|{p['seed']}|{p['approximate_extra_steps_before_time_break_even']:.1f}|{p['approximate_extra_steps_preserving_5pct_time_saving']:.1f}|")
 lines+=['','K16只有约25步余量就会耗尽当前速度优势；保持5%节时则只有约11步。简单把训练翻倍，即使质量恢复，也不能据此宣称同质量更便宜。K4有较多时间余量，但质量缺口也更大。',
 '','## 下一轮：先排除学习率搜索边界','','优先对差距更小的K16与dense各补一个相同峰值学习率1e-3、相同256步的从头训练。保留相同初始化、数据顺序、确定性后向、优化器、调度和损失实现。两方法都获得同样新增调参机会；只读原校准集，旧report不再作为选择依据。',
 '','这只是学习率边界诊断，不是新方法贡献。若结果改善，再决定是否扩网格、重复种子和建立新保留评测；不能把本次开发集达标当论文成立。若大LR不稳定，保留失败并停止本轮，不自动无限重试。',
 '','图：../../results/amp-recovery-curve-audit-v0/curves.png；机器可复算数据：../../results/amp-recovery-curve-audit-v0/result.json。']
 (R/'docs/amp-recovery-curve-analysis-2026-09-15.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='complete',pairs=pairs)))
if __name__=='__main__':main()
