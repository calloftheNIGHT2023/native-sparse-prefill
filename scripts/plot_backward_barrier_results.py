"""Plot measured fixed-graph repeatability and selected-reference comparisons."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/figures'
OUT.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout='constrained')
for name, folder, color in [
    ('Original', 'flashmoba-long-backward-original-v0', '#c43b36'),
    ('Barrier candidate', 'flashmoba-long-backward-barrier-v0', '#167b75'),
]:
    data = json.loads((ROOT/'results'/folder/'result.json').read_text(encoding='utf-8'))
    assert data['status']=='complete' and data['actual_k4_selected_masks_match']
    for deterministic, style in [(True, '-'), (False, ':')]:
        rows = [r for r in data['rows'] if r['deterministic']==deterministic]
        label = name + (' / deterministic' if deterministic else ' / default')
        axes[0].plot([r['repeat'] for r in rows],
                     [100*r['qkv_repeat'][0]['relative_l2'] for r in rows],
                     style, color=color, linewidth=1.5, marker='o', markersize=3, label=label)
        axes[1].plot([r['repeat'] for r in rows],
                     [100*r['reference_relative_l2'] for r in rows],
                     style, color=color, linewidth=1.5, marker='o', markersize=3, label=label)
axes[0].set_title('Full Q gradient: repeat vs. first call')
axes[1].set_title('128 diagnostic rows: vs. FP64 reference')
axes[1].axhline(5., color='#777', linestyle='--', linewidth=.8, label='Predeclared 5% gate')
for ax in axes:
    ax.set_xlabel('Backward call index')
    ax.set_ylabel('Relative L2 difference (%)')
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
axes[1].legend(fontsize=7, loc='upper right')
fig.suptitle('Frozen real 8K QKV, fixed routes and upstream gradient; RTX 6000 Ada', fontsize=11)
fig.savefig(OUT/'flashmoba-backward-barrier-v0.png', dpi=180)
fig.savefig(OUT/'flashmoba-backward-barrier-v0.pdf')
plt.close(fig)
