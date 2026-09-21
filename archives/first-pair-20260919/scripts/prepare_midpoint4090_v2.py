"""Freeze a full same-GPU trajectory after the cross-GPU short-task replay failed."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, tarfile, io
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=load(R/'provenance/midpoint64-protocol-v1.json')
    p['parent_protocol_sha256']=sha(R/'provenance/midpoint64-protocol-v1.json')
    p['created_utc']=datetime.now(timezone.utc).isoformat()
    p['revision_reason']='Strict v1 failed. Bounded diagnostic found 12/12 sampled32K outputs identical but one of12 no-context answers changed. Do not combine old and new GPU task scores; reevaluate all fixed0/64/128 checkpoints on the4090. This does not make v1 pass.'
    p['scope']='Exploratory full same-GPU reevaluation on64 reused development articles. All0/64/128 checkpoints, dense/K32/K48, two existing seeds at64/128; one functional zero-LoRA baseline per mode. Zero training. No best-checkpoint selection, independent validation, causal hardware claim, or cross-GPU speed comparison.'
    p.update(jobs=[],maximum_seconds=3600,maximum_job_seconds=360,expected_task_predictions=1920,expected_midpoint_predictions=768,expected_replay_predictions=0)
    files={}
    for step in [0,64,128]:
        for k in [0,32,48]:
            label='gentle32k' if k==48 else 'expanded76'
            mirror=R/f'results/cloud-{label}-evidence-v0'
            assert load(R/f'results/{label}-audit-v0/result.json')['status']=='verified'
            for seed in ([2026091660] if step==0 else [2026091660,2026091661]):
                name=f'seed{seed}' if k==48 else f'k{k}-seed{seed}'
                parent=f'results/{label}-stage-v0/{name}'
                cp=parent+f'/checkpoint-{step}.pt';training=parent+'/result.json'
                j=dict(name=f'trajectory-k{k}-seed{seed}-step{step}',phase='same_gpu_trajectory',k=k,seed=seed,step=step,path=cp,sha256=sha(mirror/cp),training_result=training,training_result_sha256=sha(mirror/training),item_ids=None)
                p['jobs'].append(j)
                files.update({cp:mirror/cp,training:mirror/training})
    for name in ['eval_midpoint64','run_midpoint64_stage']:
        old=R/f'scripts/{name}_v1.py';new=R/f'scripts/{name}_v2.py'
        s=old.read_text().replace('midpoint64-protocol-v1','midpoint64-protocol-v2').replace('midpoint64-stage-v1','midpoint64-stage-v2').replace('midpoint64-queue-v1','midpoint64-queue-v2').replace('midpoint64-evidence-v1','midpoint64-evidence-v2').replace('eval_midpoint64_v1','eval_midpoint64_v2')
        if name=='run_midpoint64_stage':
            s=s.replace("            if j['phase']=='midpoint':assert replayed=={0,32,48}\n",'')
        else:
            s=s.replace("    predictions.append(pred)","    predictions.append(pred)\n    if len(predictions)%16==0:event('prediction_progress',completed=len(predictions),expected=128)")
        ast.parse(s);new.write_text(s,encoding='utf-8')
        p['source_sha256'].pop(f'scripts/{name}_v1.py');p['source_sha256'][f'scripts/{name}_v2.py']=sha(new)
    for j in p['jobs']:assert j['phase']=='same_gpu_trajectory' and j['item_ids'] is None
    assert len(p['jobs'])==15
    for n in list(p['source_sha256'])+list(p['data_sha256']):files[n]=R/n
    pp=R/'provenance/midpoint64-protocol-v2.json';assert not pp.exists();pp.write_text(json.dumps(p,indent=2)+'\n')
    files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/midpoint64-migration-v2.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));a.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(protocol_sha256=sha(pp),archive=proof,jobs=15,task_predictions=1920,updates=0)))
if __name__=='__main__':main()
