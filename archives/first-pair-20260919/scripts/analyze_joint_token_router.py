"""Read-only checkpoint audit and route replacement; no fitted parameters."""
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import LanguageModel
from zoology.config import ModelConfig
from zoology_sparse_schedule import install
from frozen_routing import TokenIndexer,FrozenRouter,select
from joint_token_routing import JointTokenRouter
from run_frozen_router import evaluate,weights_sha,now,save

@torch.no_grad()
def answer_query_trace(model,indexers,data):
    """Post-fit labels locate diagnostic queries only; never update either model."""
    accum=[dict(rows=0,answer_queries=0,kl_all_sum=0.,kl_answer_sum=0.,teacher_entropy_sum=0.,teacher_source_value_mass=0.,exact_source_hits=0,learned_source_hits=0) for _ in range(2)]
    current={};hooks=[]
    def make_hook(i):
        def hook(mha,args):
            x=args[0];q,k,_=mha.Wqkv(x).chunk(3,-1);n=x.shape[1];causal=torch.ones(n,n,dtype=torch.bool).tril()
            full=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]);log_teacher=full.masked_fill(~causal,-torch.inf).log_softmax(-1).masked_fill(~causal,0.);teacher=log_teacher.exp()*causal
            score=indexers[i](x);log_student=score.masked_fill(~causal,-torch.inf).log_softmax(-1).masked_fill(~causal,0.);kl=(teacher*(log_teacher-log_student)).sum(-1)
            b,p=current['query_indices'];s=current['source_values'];z=accum[i];z['rows']+=kl.numel();z['answer_queries']+=len(p);z['kl_all_sum']+=float(kl.sum());z['kl_answer_sum']+=float(kl[b,p].sum());z['teacher_entropy_sum']+=float(-(teacher*log_teacher).sum())
            z['teacher_source_value_mass']+=float(teacher[b,p,s].sum());z['exact_source_hits']+=int(select(full)[b,p,s].sum());z['learned_source_hits']+=int(select(score)[b,p,s].sum())
        return hook
    for i,layer in enumerate(model.backbone.layers):hooks.append(layer.sequence_mixer.register_forward_pre_hook(make_hook(i)))
    try:
        for first in range(0,len(data['inputs']),32):
            x=data['inputs'][first:first+32];y=data['labels'][first:first+32];b,p=torch.where(y!=-100);source=[]
            for bi,pi in zip(b.tolist(),p.tolist()):
                si=next(j+1 for j in range(0,8,2) if x[bi,j]==x[bi,pi]);assert x[bi,si]==y[bi,pi];source.append(si)
            current.update(query_indices=(b,p),source_values=torch.tensor(source));model(x)
    finally:
        for hook in hooks:hook.remove()
    return [dict(layer=i,answer_queries=z['answer_queries'],mean_kl_all=z['kl_all_sum']/z['rows'],mean_kl_answer_queries=z['kl_answer_sum']/z['answer_queries'],mean_teacher_entropy=z['teacher_entropy_sum']/z['rows'],mean_teacher_source_value_mass=z['teacher_source_value_mass']/z['answer_queries'],exact_source_value_recall=z['exact_source_hits']/z['answer_queries'],learned_source_value_recall=z['learned_source_hits']/z['answer_queries']) for i,z in enumerate(accum)]

def main():
    src=ROOT/'results/joint-token-router-v0';r=json.loads((src/'result.json').read_text());assert r['status']=='complete'
    out=ROOT/'results/joint-token-analysis-v0';out.mkdir(exist_ok=False);torch.set_num_threads(4);timer=time.perf_counter()
    manifest=json.loads((src/'manifest.json').read_text())
    for row in manifest:assert hashlib.sha256((src/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    data=torch.load(src/'data-and-initialization.pt',map_location='cpu',weights_only=True);initial_hash=weights_sha(data['initial_state']);rows=[];audit=[];training_stamps=[];recomputed=0;frozen_checks=[]
    for run in r['runs']:
        folder=src/run['method'];ev=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()];step=[e for e in ev if e['event']=='optimizer_step'];epochs=[e for e in ev if e['event']=='epoch']
        assert [e['step'] for e in step]==list(range(1,12521)) and len(epochs)==40
        assert all(math.isfinite(e[k]) for e in step for k in ('ce','kl_sum_layers','main_gradient_norm','indexer_gradient_norm'))
        assert [e['utc'] for e in ev]==sorted(e['utc'] for e in ev);assert run['initial_backbone_hash']==initial_hash
        assert step[-1]['input_tokens']==25600000 and step[-1]['supervised_answers']==1600000
        training_stamps.append(dict(method=run['method'],start=ev[0]['utc'],end=ev[-1]['utc'],updates=12520))
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['steps']==12520 and ck['epoch']==39
        model=LanguageModel(ModelConfig(**ck['config']['model']));model.load_state_dict(ck['model']);install(model,'native');model.eval()
        ix=torch.nn.ModuleList([TokenIndexer(128,run['rank']) for _ in range(2)]);ix.load_state_dict(ck['indexers']);route=JointTokenRouter(model,ix,run['route']);route.collect_aux=False
        for key in ('fresh','swapped'):
            a=evaluate(model,data[key],'cpu');assert a['predictions']==run[key]['predictions'],(run['method'],key)
            recomputed+=a['answers'];audit.append(dict(method=run['method'],split=key,accuracy=a['accuracy'],answers=a['answers'],predictions_match=True))
        trace=answer_query_trace(model,ix,data['fresh']);route.restore()
        # Descriptive post-fit diagnostic; no changes to checkpoint or training decisions.
        exact=evaluate(model,data['fresh'],'cpu');exact_swap=evaluate(model,data['swapped'],'cpu')
        frozen=FrozenRouter(model,ix)
        learned=evaluate(model,data['fresh'],'cpu');learned_swap=evaluate(model,data['swapped'],'cpu')
        frozen.exact_layers={1};layer0=evaluate(model,data['fresh'],'cpu');frozen.exact_layers={0};layer1=evaluate(model,data['fresh'],'cpu');frozen.restore()
        assert weights_sha(model.state_dict())==run['final_backbone_hash']
        frozen_checks.append(dict(method=run['method'],exact=exact,exact_swap=exact_swap,learned=learned,learned_swap=learned_swap,only_layer0_learned=layer0,only_layer1_learned=layer1,answer_query_trace_under_original_route=trace))
        first99=next((c['epoch'] for c in run['curves'] if c['development_accuracy']>=.99),None)
        rows.append(dict(method=run['method'],rank=run['rank'],fresh=run['fresh']['accuracy'],swapped=run['swapped']['accuracy'],first_development_99_epoch=first99,final_kl=run['curves'][-1]['mean_kl_sum_layers'],wall_seconds=run['wall_seconds'],indexer_parameters=sum(p.numel() for p in ix.parameters())))
    assert r['backbone_optimizer_updates']==37560 and r['indexer_optimizer_updates']==37560
    assert weights_sha(torch.load(src/'exact_r16_shadow/initial-indexer.pt',map_location='cpu',weights_only=True))==weights_sha(torch.load(src/'learned_r16/initial-indexer.pt',map_location='cpu',weights_only=True))
    labels=data['fresh']['labels'];target=labels[labels!=-100].numpy().reshape(1024,4);reference=(np.asarray(r['runs'][0]['fresh']['predictions']).reshape(1024,4)==target).mean(1)
    intervals=[];rng=np.random.default_rng(2026091472)
    for run in r['runs'][1:]:
        delta=(np.asarray(run['fresh']['predictions']).reshape(1024,4)==target).mean(1)-reference
        boot=delta[rng.integers(0,1024,(5000,1024))].mean(1)
        intervals.append(dict(method=run['method'],fresh_paired_accuracy_difference=float(delta.mean()),sequence_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),scope='Conditional on these fitted checkpoints; not training-seed uncertainty'))
    summary=dict(utc=now(),manifest_files_verified=len(manifest),status='audited',rows=rows,paired_intervals=intervals,scientific_backbone_updates=37560,scientific_indexer_updates=37560,input_tokens=76800000,supervised_answers=4800000,initial_backbone_hash=initial_hash,matched_indexer16_initialization=True,original_cpu_predictions_checked=recomputed,cpu_all_predictions_match=True,analysis_optimizer_updates=0,training_stamps=training_stamps,analysis_wall_seconds=time.perf_counter()-timer,control_valid=r['control_valid'])
    save(out/'audit-and-summary.json',summary);save(out/'route-replacement.json',dict(utc=now(),optimizer_updates=0,scope='Post-fit descriptive interventions on fixed weights; changes may shift downstream activations, not intrinsic capacity proofs',runs=frozen_checks))
    print(json.dumps(summary,ensure_ascii=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(14,4))
        for run in r['runs']:
            x=[c['epoch'] for c in run['curves']];label=run['method'];axs[0].plot(x,[c['development_accuracy']*100 for c in run['curves']],label=label);axs[1].plot(x,[c['train_nll'] for c in run['curves']],label=label);axs[2].plot(x,[max(c['mean_kl_sum_layers'],1e-10) for c in run['curves']],label=label)
        for ax,title,ylabel in zip(axs,['Development accuracy','Task training loss','Indexer alignment loss'],['Accuracy (%)','NLL','KL (sum of layers)']):ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        axs[0].axhline(99,color='gray',linestyle=':',alpha=.6);axs[2].set_yscale('log');axs[0].legend(fontsize=7);fig.suptitle('Online token-indexer baseline | single seed, length 64, top-8 | dense KL teacher');fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);fig.savefig(out/'curves.pdf');plt.close(fig)
    except ImportError as exc:save(out/'plot-unavailable.json',dict(error=str(exc)))
if __name__=='__main__':main()
