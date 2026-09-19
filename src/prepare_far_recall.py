import argparse,json,shutil
from pathlib import Path
import torch
from far_recall import static_support,ancestors,family,near_control,symbolic_lookup
from run_joint_pilot import save,sha,utc
ROOT=Path(__file__).resolve().parents[1]
def main(args):
    cfg=json.loads(args.config.read_text(encoding='utf-8')); out=args.output
    out.mkdir(parents=True,exist_ok=False); started=utc(); torch.set_num_threads(4)
    layout,selected,mask=static_support(cfg)
    reach,history=ancestors(mask,cfg['sequence_length']-1,cfg['layers'])
    assert not reach[cfg['evidence_start']:cfg['evidence_end_exclusive']].any()
    metadata={}; datasets={}; seen=set(); audited=0
    for split,count in cfg['families'].items():
        xs=[]; ys=[]; records=[]
        for i in range(count):
            x,y,meta=family(cfg,split,i)
            assert meta['family_sha256'] not in seen; seen.add(meta['family_sha256'])
            assert torch.equal(x[:,reach],x[0:1,reach].expand(len(y),-1))
            assert not reach[meta['value_position']]
            assert torch.equal(symbolic_lookup(x,cfg),y)
            near,near_pos=near_control(x,meta,cfg)
            assert reach[near_pos] and torch.equal(symbolic_lookup(near,cfg),y)
            xs.append(x.to(torch.uint8)); ys.append(y); records.append(meta); audited+=len(y)
        datasets[split]=dict(inputs=torch.cat(xs),labels=torch.cat(ys))
        metadata[split]=records
    torch.save(datasets,out/'tokens.pt'); torch.save(dict(mask=mask,selected=selected,ancestors=reach),out/'graph.pt')
    save(out/'families.json',metadata); shutil.copy2(args.config,out/'frozen-config.json')
    for name in ['far_recall.py','prepare_far_recall.py','sparse_reference.py','routing_rules.py']:
        dest=out/'source-snapshot'/name; dest.parent.mkdir(exist_ok=True); shutil.copy2(ROOT/'src'/name,dest)
    sink=cfg['block_size']; non_sink=torch.where(reach & (torch.arange(len(reach))>=sink))[0]
    summary=dict(started_utc=started,finished_utc=utc(),examples=audited,
        counts={s:len(v['labels']) for s,v in datasets.items()},families=cfg['families'],
        seed_disjoint_families=True,exact_family_hashes_disjoint=True,full_parser_accuracy=1.,
        static_local_balanced_accuracy_upper_bound=1/cfg['num_values'],
        query_position=cfg['sequence_length']-1,ancestor_counts=[int(h.sum()) for h in history],
        non_sink_min_ancestor=int(non_sink.min()),non_sink_max_ancestor=int(non_sink.max()),
        evidence_interval=[cfg['evidence_start'],cfg['evidence_end_exclusive']],
        all_examples_counterfactually_identical_on_static_local_ancestors=True,
        neural_model_accuracy_measured=False,config_sha256=sha(args.config),scope=cfg['scope'])
    save(out/'audit.json',summary)
    save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in out.rglob('*') if p.is_file()])
    print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
