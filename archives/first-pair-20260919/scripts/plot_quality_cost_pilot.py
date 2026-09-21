"""Standalone scientific figures for the fixed paired quality/cost pilot."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/flashmoba-quality-cost-analysis-v0/result.json').read_text(encoding='utf-8'))
out=ROOT/'docs/figures';out.mkdir(exist_ok=True)
names=['dense','original_k4','barrier_k4','fp32_k4','fp32_k16']
labels=['Dense','Original K4','Barrier K4','FP32 pool K4','FP32 pool K16']
colors=['#333333','#c64e45','#dc9860','#288a87','#496fbd']
fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
for name,label,color in zip(names,labels,colors):
    rows=[r for r in data['rows'] if r['condition']==name]
    curves=np.asarray([[p['mean_nll'] for p in r['calibration_curve']] for r in rows])
    steps=[p['step'] for p in rows[0]['calibration_curve']]
    for values in curves:axes[0].plot(steps,values,color=color,linewidth=.7,alpha=.4)
    axes[0].plot(steps,curves.mean(axis=0),'-o',color=color,linewidth=1.6,markersize=4,label=label)
    ratios=np.asarray([r['steady_step_time_ratio'] for r in rows]);gap=np.asarray([r['paired_report_gap'] for r in rows])
    axes[1].scatter(ratios,gap,color=color,s=32,alpha=.6)
    axes[1].plot(ratios,gap,color=color,linewidth=.9)
    axes[1].scatter(ratios.mean(),gap.mean(),color=color,s=65,marker='D',label=label,zorder=4)
axes[0].set(title='Calibration learning curves',xlabel='Optimizer updates',ylabel='Mean next-token NLL')
axes[0].legend(fontsize=8)
axes[1].axhline(.03,color='#999',linestyle='--',linewidth=.9)
axes[1].axvline(.95,color='#999',linestyle='--',linewidth=.9)
axes[1].set(title='Held-out report quality vs. training time',xlabel='Median training-step time / paired dense',ylabel='Report NLL minus paired dense')
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen2.5-0.5B LoRA, 8K, 64 steps, two initialization seeds; RTX 6000 Ada',fontsize=11)
fig.savefig(out/'flashmoba-quality-cost-v0.png',dpi=180)
fig.savefig(out/'flashmoba-quality-cost-v0.pdf')
plt.close(fig)
