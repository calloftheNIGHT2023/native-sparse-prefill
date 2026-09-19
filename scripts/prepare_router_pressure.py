"""Prepare an existing MQAR configuration with 16 records and length 128; no model training."""
import hashlib,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import multiquery_ar,UPSTREAM,configuration,LanguageModel
from run_frozen_router import now,save,sha

out=ROOT/'data/router-pressure-kv16-v0';out.mkdir(exist_ok=False);splits={};hashes={};mapping_info={}
shutil.copy2(__file__,out/'prepare-source.py');shutil.copy2(UPSTREAM/'zoology/data/multiquery_ar.py',out/'upstream-generator.py')
for name,count,seed in [('train',10000,2026091494),('development',1000,2026091495),('test',1024,2026091496)]:
    torch.manual_seed(seed);segment=multiquery_ar(vocab_size=256,num_examples=count,input_seq_len=128,seed=seed,num_kv_pairs=16);data=dict(inputs=segment.inputs,labels=segment.labels);seen=set()
    for x,y in zip(data['inputs'],data['labels']):
        bank={int(x[i]):int(x[i+1]) for i in range(0,32,2)};assert len(bank)==len(set(bank.values()))==16;pos=torch.where(y!=-100)[0];assert len(pos)==16 and int(pos.min())>=32
        for p in pos:assert bank[int(x[p])]==int(y[p])
        h=hashlib.sha256(x.numpy().tobytes()).hexdigest();assert h not in seen;seen.add(h)
    for prior in hashes.values():assert not seen&prior
    splits[name]=data;hashes[name]=seen;mapping_info[name]=dict(rows=count,seed=seed,answers=count*16,input_tokens=count*128)
fresh=splits['test'];swapped=dict(inputs=fresh['inputs'].clone(),labels=torch.full_like(fresh['labels'],-100));swap_rows=[]
for i,(x,y) in enumerate(zip(fresh['inputs'],fresh['labels'])):
    p=int(torch.where(y!=-100)[0][-1]);src=next(j+1 for j in range(0,32,2) if x[j]==x[p]);alt=((src-1+2)%32)+1;swapped['inputs'][i,src]=x[alt];swapped['inputs'][i,alt]=x[src];swapped['labels'][i,p]=x[alt]
    assert int((x!=swapped['inputs'][i]).sum())==2 and torch.equal(x[32:],swapped['inputs'][i,32:]);swap_rows.append([p,src,alt])
torch.save(dict(**splits,swapped=swapped,swap_positions=swap_rows),out/'data.pt')
cfg=configuration(128)
for s in cfg.data.train_configs+cfg.data.test_configs:s.num_kv_pairs=16
model=LanguageModel(cfg.model);parameters=sum(p.numel() for p in model.parameters());del model
save(out/'config.json',cfg.model_dump(mode='json'))
save(out/'audit.json',dict(status='data_ready_not_trained',utc=now(),sequence_length=128,num_kv_pairs=16,proposed_topk=8,mandatory_local=2,remote_slots=6,parameters=parameters,splits=mapping_info,all_rows_unique_and_disjoint=True,all_labels_verified=True,swap_answers=1024,model_training_updates=0,data_sha256=sha(out/'data.pt'),scope='Existing MQAR family with greater record pressure; no new benchmark claim, no trained baseline yet. Multiple layers may still mediate information; record count alone is not a necessity theorem.'))
save(out/'manifest.json',[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]);print(json.dumps(dict(status='ready',model_training_updates=0,parameters=parameters,splits=mapping_info)))
