"""Freeze exact continuation of interrupted evaluations without changing comparisons."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,io
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 pp=R/'provenance/fresh-book256-resume-protocol-v0.json';assert not pp.exists();parent=load(R/'provenance/fresh-book256-protocol-v0.json');proof=load(R/'logs/fresh-book256-user-stop-backup.json');assert proof['local_backup_verified'] and sha(R/'exports/fresh-book256-user-stop-v0.tar.gz')==proof['archive']['sha256'];assert load(R/'logs/fresh-book256-user-stop-partial-audit.json')['saved_predictions']==6495
 oldroot=R/'results/cloud-fresh-book256-user-stop-v0';j0=parent['jobs'][6];rows=[json.loads(x) for x in (oldroot/'results/fresh-book256-stage-v0'/j0['name']/'task-predictions.jsonl').read_text().splitlines()];assert len(rows)==321;carry=R/'provenance/fresh-book256-resume-carry-v0.json';assert not carry.exists();save(carry,rows)
 jobs=[dict(x) for x in parent['jobs'][6:]];jobs[0].update(carry_path=carry.relative_to(R).as_posix(),carry_count=321)
 names=['eval_fresh_book256_resume_v0.py','run_fresh_book256_resume_stage_v0.py','report_fresh_book256_resume_v0.py','collect_fresh_book256_resume_v0.py','prepare_fresh_book256_resume_v0.py'];sources=dict(parent['source_sha256']);sources.update({'scripts/'+n:sha(R/'scripts'/n) for n in names})
 for n,h in parent['source_sha256'].items():assert sha(R/n)==h,n
 data=dict(parent['data_sha256'])
 for n in ['provenance/fresh-book256-protocol-v0.json','provenance/fresh-book256-resume-carry-v0.json','logs/fresh-book256-user-stop-partial-audit.json']:data[n]=sha(R/n)
 for n,h in parent['data_sha256'].items():assert sha(R/n)==h,n
 p=dict(parent,created_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256=sources,data_sha256=data,expected_task_predictions=3087,expected_new_task_predictions=2766,expected_diagnostic_replay_predictions=2,expected_nll_forwards=12,maximum_seconds=3600,maximum_job_seconds=1150,maximum_gpu_cost_usd_excluding_setup_storage=3600*.74/3600,parent_interruption_archive_sha256=proof['archive']['sha256'],primary=parent['primary'],scope=parent['scope']+' User resumed on another4090 Pod. Keep six complete original conditions and321 saved records; two exact saved-row logit replays at indices5,320 must pass max_abs<=1e-6 plus eachcheckpoint4calibrationNLL<=1e-6. Continue708 then1029+1029 tasks. Count2766new tasks and2diagnostic replay predictions separately, not3087 carried+new rows or9261whole-stage rows as new computation. Preserve original sources/protocol/interruption archive. New environment exactly pinned, no new training, no changed tests/thresholds. A replay failure stops without discarding prior records or loosening tolerance.')
 save(pp,p);files={n:R/n for n in list(sources)+list(data)};files[pp.relative_to(R).as_posix()]=pp;a=R/'exports/fresh-book256-resume-launch-v0.tar.gz';assert not a.exists();entries=[]
 with tarfile.open(a,'w:gz') as t:
  for n,f in sorted(files.items()):
   raw=f.read_bytes();entries.append(dict(path=n,sha256=sha(f),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('migration-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(sha256=sha(a),bytes=a.stat().st_size,files=len(entries));save(a.with_suffix('.json'),proof);print(json.dumps(dict(status='prepared',protocol_sha256=sha(pp),archive=proof)))
if __name__=='__main__':main()
