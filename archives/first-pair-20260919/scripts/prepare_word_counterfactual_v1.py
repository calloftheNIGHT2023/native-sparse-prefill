"""Freeze all existing dense/K32 trajectories after the dense-base task gate passed."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    gate=load(R/'results/word-counterfactual-audit-v0/result.json');assert gate['status']=='verified' and gate['short_gate']['passed'] and gate['long_gate']['passed']
    parent=load(R/'provenance/word-counterfactual-protocol-v0.json');traj=load(R/'provenance/midpoint64-protocol-v2.json')
    jobs=[];files={}
    for old in traj['jobs']:
        if old['k'] not in [0,32] or (old['k']==0 and old['step']==0):continue
        j=dict(old);j.update(name=f"word-k{j['k']}-seed{j['seed']}-step{j['step']}",phase='word_trajectory');jobs.append(j)
    assert len(jobs)==9
    s=(R/'scripts/eval_word_counterfactual_v0.py').read_text().replace('word-counterfactual-protocol-v0','word-counterfactual-protocol-v1')
    s=s.replace("     if not short_gate['passed']:break\n",'')
    s=s.replace("  if short_gate['passed']:long_gate=gate('long32768',protocol['long_gate']);event('long_ability_gate',**long_gate)","  long_gate=gate('long32768',protocol['long_gate']);event('long_ability_gate',**long_gate)")
    s=s.replace("assert len(predictions)==(192 if short_gate['passed'] else 128)","assert len(predictions)==192")
    f=R/'scripts/eval_word_counterfactual_v1.py';ast.parse(s);f.write_text(s)
    s=(R/'scripts/run_word_counterfactual_stage_v0.py').read_text().replace('word-counterfactual-protocol-v0','word-counterfactual-protocol-v1').replace('word-counterfactual-stage-v0','word-counterfactual-stage-v1').replace('word-counterfactual-queue-v0','word-counterfactual-queue-v1').replace('word-counterfactual-evidence-v0','word-counterfactual-evidence-v1').replace('eval_word_counterfactual_v0','eval_word_counterfactual_v1')
    s=s.replace("q.wait(timeout=remaining)","q.wait(timeout=min(remaining,p['maximum_job_seconds']))").replace("assert count in [p['expected_task_predictions_if_short_passes'],p['expected_task_predictions_if_short_fails']]","assert count==p['expected_task_predictions']")
    f=R/'scripts/run_word_counterfactual_stage_v1.py';ast.parse(s);f.write_text(s)
    sources={n:h for n,h in parent['source_sha256'].items() if 'word_counterfactual_v0' not in n and 'word_counterfactual_stage_v0' not in n}
    for n in ['scripts/eval_word_counterfactual_v1.py','scripts/run_word_counterfactual_stage_v1.py','scripts/prepare_word_counterfactual_v1.py']:sources[n]=sha(R/n)
    data=dict(parent['data_sha256']);reference='results/word-counterfactual-stage-v0/dense-base-ability/result.json';local=R/'results/cloud-word-counterfactual-evidence-v0'/reference;data[reference]=sha(local);files[reference]=local
    p=dict(parent);p.update(created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,reference=reference,reference_sha256=sha(local),parent_gate_audit_sha256=sha(R/'results/word-counterfactual-audit-v0/result.json'),maximum_seconds=2700,maximum_job_seconds=240,expected_task_predictions=1728,maximum_gpu_cost_usd_excluding_setup_storage=2700/3600*.74,scope='Allfixed0/64/128 dense/K32 checkpoints; dense0 reuses verified same4090 ability-gate predictions.32 development proceduralbackground families, paired facts. All192 prompts evaluated for every model regardless of itsgate score. No training, no independent confirmation, no zero-checkpoint doublecounting. Short/no-context prompts have only4/1 unique token sequences and are descriptive capability checks, not64 independent questions.',primary='Long-context accuracy and fraction of32 families where both factual versions are correct, plotted at allthree checkpoints. Pair family bootstrap jointly across facts andseeds; no selecting best checkpoint.')
    pp=R/'provenance/word-counterfactual-protocol-v1.json';assert not pp.exists();save(pp,p)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/word-counterfactual-launch-v1.tar.gz';entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof,jobs=9,predictions=1728)))
if __name__=='__main__':main()
