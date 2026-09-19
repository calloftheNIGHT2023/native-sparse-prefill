"""Freeze one-checkpoint three-mode profiler diagnosis after cost screen failure."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
    audit=load(R/'results/word-suffix-cost-audit-v0/result.json');assert audit['status']=='verified' and not audit['summary']['cost_screen_passed']
    p=load(R/'provenance/word-suffix-cost-protocol-v0.json');job=dict(next(j for j in p['jobs'] if j['k']==32 and j['seed']==2026091660));job.update(name='profile-k32-seed2026091660',phase='profile')
    p['jobs']=[job];p['references']=[x for x in p['references'] if x['training_k']==job['k'] and x['seed']==job['seed']];item=p['item_ids'][0];p['item_ids']=[item]
    p['schedule']=[dict(item_id=item,mode=m,phase=phase,round=-1,order=i*3+j) for i,phase in enumerate(['warmup','profile']) for j,m in enumerate(['DD','SS','SD'])]
    names=['run_expanded76.py','amp_recovery_state.py','chunked_lm_loss.py','run_flashmoba_realtext_precision.py','word_suffix_attention.py','profile_word_suffix_v0.py','run_word_suffix_profile_stage_v0.py','prepare_word_suffix_profile_v0.py','report_word_suffix_profile_v0.py']
    sources={'scripts/'+n:sha(R/'scripts'/n) for n in names};data={n:h for n,h in p['data_sha256'].items() if not n.startswith('results/')};files={}
    for x in p['references']:data[x['path']]=x['sha256'];files[x['path']]=R/'results/cloud-word-suffix-cost-evidence-v0'/x['path']
    p.pop('cost_screen');p.update(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,data_sha256=data,maximum_seconds=600,maximum_job_seconds=540,expected_task_predictions=6,maximum_gpu_cost_usd_excluding_setup_storage=600*.74/3600,parent_audit_sha256=sha(R/'results/word-suffix-cost-audit-v0/result.json'),scope='Three modes on one fixed trainedK32 checkpoint andone fixed development prompt, one warmup andone profiler pass each.0optimizer updates,6logged forwards. Exactexistinglogit replay1e-6. Raw CPU/CUDA traces with filehashes. Heuristic kernel grouping is diagnostic only, sumkernel time isnot walltime; use completed interleaved uninstrumented benchmark for speed. No performance tuning, newmethod, independentquality ortraining speedclaim. Original32K training peak39-40GiB exceeds24GiBcurrentcard; any subsequent training-cost screen must explicitly use matchedmemory-saving settings, notpretend to repeatold6000Ada protocol.')
    pp=R/'provenance/word-suffix-profile-protocol-v0.json';assert not pp.exists();save(pp,p)
    for n in list(sources)+list(data):files.setdefault(n,R/n)
    files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/word-suffix-profile-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
