"""Snapshot the two pressure batches' final reports, state and reproducible source files."""
import hashlib,json,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();out=ROOT/'provenance/router-pressure-session-final-2026-09-14';out.mkdir(exist_ok=False)
paths={ROOT/'STATE.md',ROOT/'TIMELINE.md',ROOT/'logs/control-state.json',ROOT/'logs/router-pressure-historical-controls.json',Path(__file__)}
for folder,patterns in [('docs',['router-pressure-*.md','router-freshdata-*.md']),('logs',['router-pressure-*.log','router-freshdata-*.log']),('scripts',['*router_pressure.py','*router_freshdata.py']),('tests',['test_router_pressure.py','test_router_freshdata.py']),('src',['run_router_pressure.py','router_pressure_models.py','run_router_freshdata.py'])]:
    for pattern in patterns:paths.update((ROOT/folder).glob(pattern))
for folder in ['results/router-pressure-analysis-v0','results/router-pressure-diagnostics-v0','results/router-freshdata-analysis-v0']:paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
paths.update(p for p in (ROOT/'literature/router-pressure-upstream-2026-09-14').rglob('*') if p.is_file())
for p in paths:
    assert p.is_file();dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
bundles=[]
for name in ['router-pressure-code-v0.tar.gz','router-pressure-results-v0.tar.gz','router-freshdata-code-v0.tar.gz','router-freshdata-results-v0.tar.gz']:
    p=ROOT/'exports'/name;bundles.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
(out/'raw-bundles.json').write_text(json.dumps(dict(utc=stamp,bundles=bundles),indent=2)+'\n',encoding='utf-8')
files=sorted(p for p in out.rglob('*') if p.is_file());manifest=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in files];(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
bundle=ROOT/'exports/router-pressure-session-final-v0.tar.gz'
with tarfile.open(bundle,'w:gz') as t:t.add(out,arcname=out.relative_to(ROOT).as_posix())
with tarfile.open(bundle) as t:
    for row in manifest:
        member=out.relative_to(ROOT).as_posix()+'/'+row['path'];assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==row['sha256']
info=dict(utc=stamp,files=len(manifest),archive_sha256=sha(bundle),archive_bytes=bundle.stat().st_size,all_archive_members_verified=True)
(ROOT/'exports/router-pressure-session-final-v0.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');print(json.dumps(info))
