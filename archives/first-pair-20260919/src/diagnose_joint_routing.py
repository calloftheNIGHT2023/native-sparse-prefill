"""Development-only routing swaps and known local controls; no optimizer updates."""
import argparse,copy,json,shutil,time
from pathlib import Path
import torch
from run_joint_pilot import build_model,evaluate,save,sha,utc

ROOT=Path(__file__).resolve().parents[1]

def main(args):
    source=args.runs.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((source/'frozen-config.json').read_text()); torch.set_num_threads(cfg['threads'])
    tokens=torch.load(args.data/'tokens.pt',weights_only=True)['validation']
    started=utc(); rows=[]
    plan={'started_utc':started,'split':'validation','paragraphs':len(tokens),'updates':0,
        'backbones':['dense','sparse_step0','sparse_warmup'],'indexers':['sparse_step0','sparse_warmup'],
        'rules':['learned','learned_recent','sink_recent'],'additional_control':'self_only: no previous-token attention',
        'scope':'Post-pilot exploratory diagnostics. Forcing recent/sink is a known control, not a new contribution.'}
    save(out/'plan.json',plan); snapshots=out/'source-snapshot'; snapshots.mkdir()
    for name in ['diagnose_joint_routing.py','run_joint_pilot.py','joint_attention.py','routing_rules.py',
                 'gathered_core.py','indexer_calibration.py','sparse_reference.py','trace_math.py']:
        shutil.copy2(ROOT/'src'/name,snapshots/name)
    for init in cfg['initializations']:
        indexers={}
        for method in plan['indexers']:
            path=source/f'{init}__seed41__{method}/final-checkpoint.pt'
            stored=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            indexers[method]={k:v.clone() for k,v in stored['indexers'].items()}
            del stored
        for method in plan['backbones']:
            model,wrapper=build_model(copy.deepcopy(cfg),init,41)
            path=source/f'{init}__seed41__{method}/final-checkpoint.pt'
            stored=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            model.load_state_dict(stored['model']); del stored
            for p in model.parameters(): p.requires_grad_(False)
            dense=evaluate(model,wrapper,tokens,'dense',cfg)
            row={'initialization':init,'backbone':method,'indexer':None,'rule':'dense','development':dense}
            rows.append(row)
            independent=evaluate(model,wrapper,tokens,'self_only',cfg)
            rows.append({'initialization':init,'backbone':method,'indexer':None,'rule':'self_only','development':independent})
            for indexer_name,weights in indexers.items():
                wrapper.indexers.load_state_dict(weights)
                for rule in plan['rules']:
                    if rule=='sink_recent' and indexer_name!=plan['indexers'][0]: continue
                    wrapper.cfg['selection_rule']=rule
                    timer=time.perf_counter(); value=evaluate(model,wrapper,tokens,'sparse',cfg)
                    row={'initialization':init,'backbone':method,'indexer':indexer_name,'rule':rule,
                         'development':value,'elapsed_seconds':time.perf_counter()-timer,'utc':utc()}
                    rows.append(row)
                    with (out/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k!='development'}|{'late_nll':value['late_nll']}),flush=True)
            wrapper.restore()
    save(out/'summary.json',{'started_utc':started,'finished_utc':utc(),'conditions':len(rows),
        'paragraphs_per_condition':len(tokens),'results':rows,'optimizer_updates':0,'scope':plan['scope']})
    save(out/'manifest.json',[{'path':str(p.relative_to(out)).replace('\\','/'),'sha256':sha(p)} for p in out.rglob('*') if p.is_file()])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--runs',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); main(p.parse_args())
