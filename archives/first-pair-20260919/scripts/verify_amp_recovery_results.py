"""Validate one results archive into an isolated local evidence directory."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from datetime import datetime,timezone
def main(a):
 proof=json.loads(a.archive.with_suffix('.json').read_text());assert hashlib.sha256(a.archive.read_bytes()).hexdigest()==proof['sha256']
 with tarfile.open(a.archive) as t:
  members=t.getmembers();assert len({m.name for m in members})==len(members)
  assert all(m.isfile() and (a.output/m.name).resolve().is_relative_to(a.output.resolve()) for m in members)
  manifest=json.load(t.extractfile('amp-recovery-results-manifest.json'))
  assert {m.name for m in members}=={x['path'] for x in manifest['files']}|{'amp-recovery-results-manifest.json'}
  for row in manifest['files']:
   raw=t.extractfile(row['path']).read();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  a.output.mkdir(parents=True,exist_ok=False);t.extractall(a.output,filter='data')
 for row in manifest['files']:assert hashlib.sha256((a.output/row['path']).read_bytes()).hexdigest()==row['sha256']
 p=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),sha256=proof['sha256'],files=len(manifest['files']),completed_jobs=len(manifest['completed_job_directories']))
 (a.output/'LOCAL-VERIFICATION.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
