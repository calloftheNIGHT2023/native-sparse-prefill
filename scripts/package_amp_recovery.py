"""Build and verify a portable delta; large base model stays separately hashed."""
import hashlib,json,tarfile
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=set()
 for pat in ['scripts/*amp_recovery*.py','data/flashmoba-amp-recovery-v0/*','results/flashmoba-amp-recovery-cpu-v1/*','results/flashmoba-amp-recovery-controller-cpu-v0/*']:
  names.update(p.relative_to(R).as_posix() for p in R.glob(pat) if p.is_file())
 names.update(['scripts/chunked_lm_loss.py','scripts/run_flashmoba_realtext_precision.py','data/flashmoba-qwen-precision-v0/manifest.json',
  'docs/flashmoba-amp-recovery-protocol-2026-09-15.md','docs/flashmoba-amp-recovery-handoff-2026-09-15.md','docs/flashmoba-long-cost-stop-and-resume-2026-09-15.md',
  'provenance/flashmoba-long-cost-recovery-freeze.txt','exports/flashmoba-barrier-wheels-v0/flash_moba-2.0.0-cp311-cp311-linux_x86_64.whl',
  'STATE.md','TIMELINE.md','logs/control-state.json','logs/amp-recovery-controller-cpu-check.log'])
 # Cross-check the model locally; this large asset need not be reuploaded if already on the Pod.
 model=[]
 for row in json.loads((R/'data/flashmoba-qwen-precision-v0/manifest.json').read_text())['files']:
  if row['path'].startswith('model/'):
   p=R/'data/flashmoba-qwen-precision-v0'/row['path'];assert sha(p)==row['sha256'];model.append(dict(path=p.relative_to(R).as_posix(),sha256=row['sha256'],bytes=p.stat().st_size))
 rows=[dict(path=n,bytes=(R/n).stat().st_size,sha256=sha(R/n)) for n in sorted(names)]
 manifest=R/'provenance/flashmoba-amp-recovery-input-manifest-v0.json';assert not manifest.exists()
 manifest.write_text(json.dumps(dict(files=rows,separate_model_assets=model),indent=2)+'\n');names.add(manifest.relative_to(R).as_posix())
 a=R/'exports/flashmoba-amp-recovery-input-v0.tar.gz';assert not a.exists()
 with tarfile.open(a,'w:gz') as t:
  for n in sorted(names):t.add(R/n,arcname=n)
 with tarfile.open(a) as t:
  members=t.getmembers();assert len(members)==len(names) and all(m.isfile() for m in members)
  for row in rows:
   raw=t.extractfile(row['path']).read();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 proof=dict(status='local_verified_gpu_pending',utc=datetime.now(timezone.utc).isoformat(),archive=a.name,sha256=sha(a),bytes=a.stat().st_size,files=len(rows),separate_model_files=len(model),uploaded=False,gpu_jobs_started=0)
 (R/'exports/flashmoba-amp-recovery-input-v0.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
