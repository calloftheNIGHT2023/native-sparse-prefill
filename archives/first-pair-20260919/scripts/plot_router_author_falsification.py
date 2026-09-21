"""Plot measured epochs only; never extrapolate a stopped control."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main(a):
    fig,axs=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    paths=[ROOT/'results/router-author-control-v0/lr2',ROOT/'results/router-author-control-v0/lr1',*[p.resolve() for p in a.fits]]
    for d in paths:
        f=json.loads((d/'fit-result.json').read_text(encoding='utf-8'));curves=f['curves']
        if d.name=='lr2':label='Dense, lr=.01, seed123 (stops at19)'
        elif d.name=='lr1':label='Dense, lr=.002154, seed123'
        else:
            label=f"Native top8, lr={f['lr']:.4g}, seed{f['seed']}"
            if f['starting_epoch']==19:
                prefix=json.loads((ROOT/'results/router-author-exact-train-v0/fit-result.json').read_text(encoding='utf-8'))['curves'];curves=prefix+curves
        x=[v['epoch'] for v in curves];axs[0].plot(x,[100*v['development_accuracy'] for v in curves],label=label,lw=1.8);axs[1].plot(x,[v['development_nll'] for v in curves],label=label,lw=1.8)
    axs[0].axhline(99,color='gray',ls=':',lw=1);axs[0].set(ylabel='Development accuracy (%)',ylim=(-2,103));axs[1].set(ylabel='Development NLL',yscale='log')
    for ax in axs:ax.set_xlabel('Training epoch');ax.grid(alpha=.2)
    axs[1].legend(fontsize=7,loc='best');fig.suptitle('Ordinary native training: time and learning-rate controls',fontsize=12)
    a.output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(a.output/('curves.'+suffix),dpi=180)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fits',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
