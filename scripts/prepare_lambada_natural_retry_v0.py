"""Retry only a NumPy-bool JSON serialization error; keep every scientific choice."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io,ast,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    pp=R/'provenance/lambada-natural-retry-protocol-v0.json';assert not pp.exists();oldp=R/'provenance/lambada-natural-protocol-v0.json';p=load(oldp)
    for oldname in ['eval_lambada_natural_v0.py','run_lambada_natural_stage_v0.py','report_lambada_natural_v0.py','collect_lambada_natural_v0.py']:
        name=oldname.replace('lambada_natural','lambada_natural_retry');path=R/'scripts'/name;assert not path.exists()
        code=(R/'scripts'/oldname).read_text(encoding='utf-8').replace('lambada-natural','lambada-natural-retry').replace('lambada_natural','lambada_natural_retry').replace('data/lambada-natural-retry-v0','data/lambada-natural-v0')
        if oldname.startswith('eval_'):
            anchor="passed=full>=protocol['base_gate']['min_full_accuracy'] and full-short>=protocol['base_gate']['min_full_minus_short_accuracy']"
            assert code.count(anchor)==1;code=code.replace(anchor,"passed=bool(full>=protocol['base_gate']['min_full_accuracy'] and full-short>=protocol['base_gate']['min_full_minus_short_accuracy'])")
        if oldname.startswith('collect_'):code=code.replace("c['cumulative_task_predictions']==35583","c['cumulative_task_predictions']==36607")
        ast.parse(code);path.write_text(code,encoding='utf-8')
    # Actual failing expression, including NumPy scalar behavior, is serialized here.
    full=np.mean([True,False]);short=np.mean([False,False]);json.dumps(dict(passed=bool(full>=.2 and full-short>=.05)))
    p['created_utc']=datetime.now(timezone.utc).isoformat();p['retry']=dict(parent_protocol_sha256=sha(oldp),reason='Cast operational gate NumPy bool_ to native bool for JSON. Original base produced1024 saved predictions before serialization error; no trained model was evaluated. Those predictions remain counted separately. Full unchanged scientific protocol rerun, base repeat is technical, not extra independent evidence.')
    p['source_sha256']={n:h for n,h in p['source_sha256'].items() if n.split('/')[-1] not in ['eval_lambada_natural_v0.py','run_lambada_natural_stage_v0.py','report_lambada_natural_v0.py','collect_lambada_natural_v0.py']}
    for n in ['prepare_lambada_natural_retry_v0.py','eval_lambada_natural_retry_v0.py','run_lambada_natural_retry_stage_v0.py','report_lambada_natural_retry_v0.py','collect_lambada_natural_retry_v0.py']:p['source_sha256']['scripts/'+n]=sha(R/'scripts'/n)
    p['data_sha256'][oldp.relative_to(R).as_posix()]=sha(oldp);save(pp,p);files={n:R/n for n in list(p['source_sha256'])+list(p['data_sha256'])};files[pp.relative_to(R).as_posix()]=pp
    a=R/'exports/lambada-natural-retry-launch-v0.tar.gz';assert not a.exists();entries=[]
    with tarfile.open(a,'w:gz') as t:
        for n,f in sorted(files.items()):
            raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
    proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
