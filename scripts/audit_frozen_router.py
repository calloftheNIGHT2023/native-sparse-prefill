import hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from zoology_entry import LanguageModel
from zoology.config import ModelConfig
from zoology_sparse_schedule import install
from frozen_routing import TokenIndexer,FrozenRouter
from run_frozen_router import evaluate,weights_sha,now,save
torch.set_num_threads(4);out=ROOT/'results/frozen-router-audit-v0';out.mkdir(exist_ok=False)
src=ROOT/'results/frozen-router-capacity-v0';result=json.loads((src/'result.json').read_text());manifest=json.loads((src/'manifest.json').read_text())
for row in manifest:assert hashlib.sha256((src/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
state=torch.load(ROOT/'results/schedule-screen-cloud-v0/native/checkpoint.pt',map_location='cpu',weights_only=False)
model=LanguageModel(ModelConfig(**state['config']['model']));model.load_state_dict(state['model']);install(model,'native');model.eval()
data=torch.load(src/'data.pt',map_location='cpu',weights_only=True);rows=[];timer=time.perf_counter()
for r in result['runs']:
    folder=src/r['method'];events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()];steps=[x for x in events if x['event']=='optimizer_step']
    assert [x['step'] for x in steps]==list(range(1,5121));assert len([x for x in events if x['event']=='epoch'])==40
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);ix=torch.nn.ModuleList([TokenIndexer(128,r['rank']) for _ in range(2)]);ix.load_state_dict(ck['indexers']);routing=FrozenRouter(model,ix)
    for field,key,exact_layers in [('fresh','fresh',set()),('swapped','swapped',set()),('only_layer0','fresh',{1}),('only_layer1','fresh',{0})]:
        routing.exact_layers=exact_layers;a=evaluate(model,data[key],'cpu');assert a['predictions']==r[field]['predictions'],(r['method'],field)
        rows.append(dict(method=r['method'],field=field,accuracy=a['accuracy'],answers=a['answers'],predictions_equal=True))
    routing.restore()
exact=evaluate(model,data['fresh'],'cpu');assert exact['predictions']==result['controls']['exact']['fresh']['predictions']
cfg=json.loads((src/'config.json').read_text());assert weights_sha(model.state_dict())==cfg['backbone_hash']
report=dict(utc=now(),manifest_files_verified=len(manifest),scientific_indexer_updates=20480,backbone_updates=0,cpu_reevaluation_answers=sum(r['answers'] for r in rows)+exact['answers'],all_predictions_equal=True,backbone_hash=weights_sha(model.state_dict()),rows=rows,wall_seconds=time.perf_counter()-timer)
save(out/'audit.json',report);print(json.dumps(report))
