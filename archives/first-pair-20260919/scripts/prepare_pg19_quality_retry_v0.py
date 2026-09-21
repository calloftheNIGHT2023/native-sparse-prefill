"""Immutable retry of the same scientific protocol after target-dtype failure."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pp=R/'provenance/pg19-quality-retry-protocol-v0.json';assert not pp.exists()
    oldp=R/'provenance/pg19-quality-protocol-v0.json';p=load(oldp)
    failed=load(R/'results/cloud-pg19-quality-evidence-v0/results/pg19-quality-stage-v0/base/result.json')
    assert failed['status']=='failed' and "not implemented for 'Int'" in failed['error']
    for oldname in ['eval_pg19_quality_v0.py','run_pg19_quality_stage_v0.py','report_pg19_quality_v0.py','collect_pg19_quality_v0.py']:
        name=oldname.replace('pg19_quality','pg19_quality_retry');path=R/'scripts'/name;assert not path.exists()
        code=(R/'scripts'/oldname).read_text(encoding='utf-8').replace('pg19-quality','pg19-quality-retry').replace('pg19_quality','pg19_quality_retry')
        if oldname.startswith('eval_'):
            anchor="def inputs(w):return torch.tensor(w[:-1][None],device='cuda'),torch.tensor(w[1:],device='cuda')"
            assert code.count(anchor)==1
            code=code.replace(anchor,"def inputs(w):return torch.tensor(w[:-1][None],device='cuda',dtype=torch.long),torch.tensor(w[1:],device='cuda',dtype=torch.long)")
        ast.parse(code);path.write_text(code,encoding='utf-8')
    p['created_utc']=datetime.now(timezone.utc).isoformat()
    p['retry']=dict(parent_protocol_sha256=sha(oldp),reason='Explicit torch.long for input and target IDs; int32 PG19 archive was valid storage but cross_entropy targets require int64. No quality outputs completed; four calibration NLLs and one failed book body forward in original attempt. No changes to books, checkpoints, masks, metrics, margins or maximum limits.')
    p['source_sha256']={n:h for n,h in p['source_sha256'].items() if not any(s in n for s in ['eval_pg19_quality_v0.py','run_pg19_quality_stage_v0.py','report_pg19_quality_v0.py','collect_pg19_quality_v0.py'])}
    for n in ['prepare_pg19_quality_retry_v0.py','eval_pg19_quality_retry_v0.py','run_pg19_quality_retry_stage_v0.py','report_pg19_quality_retry_v0.py','collect_pg19_quality_retry_v0.py']:p['source_sha256']['scripts/'+n]=sha(R/'scripts'/n)
    p['data_sha256'][oldp.relative_to(R).as_posix()]=sha(oldp)
    save(pp,p);files={n:R/n for n in list(p['source_sha256'])+list(p['data_sha256'])};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/pg19-quality-retry-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
