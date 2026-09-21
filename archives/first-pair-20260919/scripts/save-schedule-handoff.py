import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'provenance/sparse-schedule-handoff-2026-09-14';OUT.mkdir(exist_ok=False)
files=['STATE.md','TIMELINE.md','logs/control-state.json','logs/cloud-connection-schedule-2026-09-14.json',
    'docs/paper-evidence-program-2026-09-14.md','docs/sparse-schedule-screen-results-2026-09-14.md',
    'docs/sparse-schedule-mechanism-boundaries-2026-09-14.md','docs/runpod-schedule-environment-2026-09-14.md',
    'literature/sparse-schedule-2026-09-14/sources.json','literature/sparse-schedule-2026-09-14/evidence.md',
    'results/schedule-screen-analysis-v0/analysis.json','results/schedule-screen-analysis-v0/schedule-curves.png',
    'results/schedule-checkpoint-audit-v0/wrapper-anomaly.json','results/schedule-checkpoint-audit-v0/full-fresh-recomputation.json',
    'scripts/analyze-sparse-schedule.py','scripts/audit-schedule-checkpoints.py','scripts/finalize-sparse-schedule.py','requirements-schedule-cloud.txt']
manifest=[]
for name in files:
    dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    manifest.append(dict(path=name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
archives=[]
for name in ['sparse-schedule-code-v1.tar.gz','sparse-schedule-final-results.tar.gz','sparse-schedule-final-logs.tar.gz']:
    p=ROOT/'exports'/name;archives.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(OUT/'backup-index.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),archives=archives,
    results='results/schedule-screen-cloud-v0',full_step_logs_verified=62600,full_fresh_predictions_recomputed=20000,
    active_training_processes_last_checked=0,pod_stopped=False,actual_invoice_verified=False,novelty_confirmed=False,
    decision='Close single-layer early-dense warmup candidate; paper objective remains unachieved; do not scale this candidate.'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(snapshot_files=len(manifest),archives=archives)))
