"""Freeze six midpoint models and three endpoint implementation replays."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,torch
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    torch.set_num_threads(4)
    meta=load(R/'data/32k-expanded-training-v0/tasks.json')
    ids=[]
    for pos in [.1,.35,.65,.9]:ids.append(next(x['item_id'] for x in meta if x['variant']=='long32768' and x['evidence_fraction']==pos))
    jobs=[];files={}
    for k in [0,32,48]:
        label='gentle32k' if k==48 else 'expanded76'
        for seed,step,phase in [(2026091660,128,'replay'),(2026091660,64,'midpoint'),(2026091661,64,'midpoint')]:
            name=f'seed{seed}' if k==48 else f'k{k}-seed{seed}'
            parent=f'results/{label}-stage-v0/{name}'
            path=parent+f'/checkpoint-{step}.pt';local=R/f'results/cloud-{label}-evidence-v0'/path
            c=torch.load(local,map_location='cpu',weights_only=False)
            assert c['step']==c['data_cursor']==c['scheduler']['last_epoch']==step and len(c['params'])==192
            assert all(torch.isfinite(x).all() for x in c['params'].values())
            assert len(c['optimizer']['state'])==192 and all(float(x['step'])==step for x in c['optimizer']['state'].values())
            report=f'report-seed{seed}-step128' if k==48 else f'report-k{k}-seed{seed}-step128'
            ref=f'results/{label}-stage-v0/{report}/task-results.json'
            training=parent+'/result.json'
            jobs.append(dict(name=f'{phase}-k{k}-seed{seed}',phase=phase,k=k,seed=seed,step=step,path=path,sha256=sha(local),training_result=training,training_result_sha256=sha(R/f'results/cloud-{label}-evidence-v0'/training),reference=ref,reference_sha256=sha(R/f'results/cloud-{label}-evidence-v0'/ref),item_ids=ids if phase=='replay' else None))
    # Check all new evaluator paths against each replay before advancing that mode.
    jobs.sort(key=lambda x:(x['phase']!='replay',x['k'],x['seed']))
    for n in ['scripts/eval_midpoint64.py','scripts/run_midpoint64_stage.py','scripts/run_expanded76.py','scripts/run_gentle32k.py','scripts/run_flashmoba_realtext_precision.py','scripts/topk_m64_adapter.py','scripts/amp_recovery_state.py','scripts/chunked_lm_loss.py','experiments/topk-m64-v0/build/nsp_topk_m64_v0.so','experiments/topk-m64-v0/build-result.json']:
        files[n]=sha(R/n)
    data={n:sha(R/n) for n in ['data/32k-expanded-training-v0/config.json','data/gentle32k-v0/config.json','data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/tasks.npz','data/32k-expanded-training-v0/tasks.json']}
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,variants=['no_context','long32768'],source_sha256=files,data_sha256=data,calibration_max_abs_error=1e-6,replay_logit_max_abs_error=1e-6,maximum_seconds=1500,maximum_job_seconds=240,expected_task_predictions=792,expected_midpoint_predictions=768,expected_replay_predictions=24,optimizer_updates=0,scope='Post-hoc midpoint evaluation of all six frozen trajectories on reused64 development articles. No checkpoint selection, no new optimizer updates, no new independent validation. Same CUDA GPU/software required and endpoint logits/calibration must replay.')
    f=R/'provenance/midpoint64-protocol-v0.json';assert not f.exists();f.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(dict(jobs=len(jobs),sha256=sha(f),predictions=792,optimizer_updates=0)))
if __name__=='__main__':main()
