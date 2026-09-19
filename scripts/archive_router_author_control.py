"""Freeze audited author control before a separate sparse batch changes live state."""
import hashlib,json,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'provenance/router-author-control-final-2026-09-14';out.mkdir(exist_ok=False)
paths={ROOT/'STATE.md',ROOT/'TIMELINE.md',ROOT/'logs/control-state.json',Path(__file__)}
for folder,patterns in [('docs',['router-author-control-*.md','router-author-attention-*.md','router-author-version-*.md']),('logs',['router-author-*.log']),('scripts',['*router_author_control.py','diagnose_router_author_attention.py']),('tests',['test_router_author_control.py']),('src',['run_router_author_control.py','router_author_control.py','preflight_router_author.py'])]:
    for pattern in patterns:paths.update((ROOT/folder).glob(pattern))
for folder in ['results/router-author-analysis-v0','results/router-author-attention-v0','literature/router-author-version-audit-2026-09-14']:paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
for p in paths:
    dst=out/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
bundles=[]
for name in ['router-author-preflight-code-v0.tar.gz','router-author-control-code-v0.tar.gz','router-author-results-v0.tar.gz']:
    p=ROOT/'exports'/name;bundles.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
(out/'raw-bundles.json').write_text(json.dumps(bundles,indent=2)+'\n',encoding='utf-8')
manifest=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()];(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
bundle=ROOT/'exports/router-author-control-final-v0.tar.gz'
with tarfile.open(bundle,'w:gz') as t:t.add(out,arcname=out.relative_to(ROOT).as_posix())
with tarfile.open(bundle) as t:
    for row in manifest:assert hashlib.sha256(t.extractfile(out.relative_to(ROOT).as_posix()+'/'+row['path']).read()).hexdigest()==row['sha256']
info=dict(utc=datetime.now(timezone.utc).isoformat(),files=len(manifest),archive_sha256=sha(bundle),bytes=bundle.stat().st_size)
(ROOT/'exports/router-author-control-final-v0.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');print(json.dumps(info))
