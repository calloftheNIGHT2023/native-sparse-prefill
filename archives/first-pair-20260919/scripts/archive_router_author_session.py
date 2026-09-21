"""Final immutable handoff, including raw archive hashes, audit outputs, current state and timeline."""
import hashlib,json,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=ROOT/'provenance/router-author-session-final-2026-09-14';out.mkdir(exist_ok=False);paths={ROOT/'STATE.md',ROOT/'TIMELINE.md',ROOT/'logs/control-state.json',ROOT/'provenance/router-author-diagnostic-plans-v0.json',Path(__file__)}
for folder,patterns in [('docs',['router-author-*.md']),('scripts',['*router_author*.py']),('src',['*router_author*.py']),('tests',['test_router_author*.py']),('logs',['router-author-*.log'])]:
    for pattern in patterns:paths.update((ROOT/folder).glob(pattern))
for folder in ['results/router-author-analysis-v0','results/router-author-attention-v0','results/router-author-sparse-analysis-v0','results/router-author-reachability-v0','results/router-author-exact-mask-v0','results/router-author-exact-analysis-v0','literature/router-author-version-audit-2026-09-14']:
    assert (ROOT/folder).exists();paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
for path in paths:
    target=out/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
bundles=[]
for name in ['router-author-preflight-code-v0','router-author-control-code-v0','router-author-results-v0','router-author-control-final-v0','router-author-sparse-preflight-code-v0','router-author-sparse-code-v0','router-author-sparse-results-v0','router-author-exact-code-v0','router-author-exact-results-v0']:
    p=ROOT/'exports'/(name+'.tar.gz');assert p.exists();bundles.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
(out/'raw-bundles.json').write_text(json.dumps(bundles,indent=2)+'\n',encoding='utf-8');manifest=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()];(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');bundle=ROOT/'exports/router-author-session-final-v0.tar.gz'
with tarfile.open(bundle,'w:gz') as t:t.add(out,arcname=out.relative_to(ROOT).as_posix())
with tarfile.open(bundle) as t:
    for row in manifest:assert hashlib.sha256(t.extractfile(out.relative_to(ROOT).as_posix()+'/'+row['path']).read()).hexdigest()==row['sha256']
info=dict(utc=datetime.now(timezone.utc).isoformat(),files=len(manifest),archive_sha256=sha(bundle),bytes=bundle.stat().st_size,all_members_verified=True);(ROOT/'exports/router-author-session-final-v0.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');print(json.dumps(info))
