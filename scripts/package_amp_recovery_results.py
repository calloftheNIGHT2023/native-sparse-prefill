"""Snapshot finished jobs only; include failed attempts, logs and provenance."""
import argparse,hashlib,json,tarfile,io
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def main(a):
 assert not a.archive.exists();files=set();completed=[]
 for stage in (R/'results').glob('amp-recovery-stage-v*'):
  for d in stage.iterdir():
   if d.is_dir() and (d/'result.json').exists():
    completed.append(d.relative_to(R).as_posix());files.update(x for x in d.rglob('*') if x.is_file() and x.suffix!='.tmp')
  files.update(x for x in stage.iterdir() if x.is_file() and x.suffix in ['.json','.log','.txt'])
 for d in (R/'results').glob('amp-recovery-repeat-diagnostic-*'):files.update(x for x in d.rglob('*') if x.is_file())
 for pat in ['scripts/*amp_recovery*.py','scripts/diagnose_amp_recovery*.py','data/flashmoba-amp-recovery-v0/*.json','provenance/amp-recovery*','docs/flashmoba-amp-recovery*.md']:
  files.update(x for x in R.glob(pat) if x.is_file())
 rows=[];skipped=[]
 # Read each file once: the archive member and its digest share identical bytes.
 with tarfile.open(a.archive,'w:gz') as t:
  for p in sorted(files):
   raw=p.read_bytes();name=p.relative_to(R).as_posix()
   if p.suffix=='.json':
    try:json.loads(raw)
    except Exception:skipped.append(name);continue
   row=dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());rows.append(row)
   m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
  manifest=dict(utc=datetime.now(timezone.utc).isoformat(),completed_job_directories=completed,files=rows,skipped_partial_json=skipped)
  raw=(json.dumps(manifest,indent=2)+'\n').encode();m=tarfile.TarInfo('amp-recovery-results-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 proof=dict(archive=str(a.archive),sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),bytes=a.archive.stat().st_size,files=len(rows),completed_jobs=len(completed),utc=datetime.now(timezone.utc).isoformat())
 a.archive.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);main(p.parse_args())
