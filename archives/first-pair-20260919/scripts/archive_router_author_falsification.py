"""Archive this completed round and verify a separate cloud snapshot."""
import argparse,hashlib,json,shutil,subprocess,tarfile,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def create():
    audit=json.loads((R/'results/router-author-falsification-analysis-v0/audit.json').read_text(encoding='utf-8'));assert audit['status']=='passed'
    paths={R/'STATE.md',R/'TIMELINE.md',R/'logs/control-state.json'}
    for folder in ['results','docs','logs','provenance','src','scripts','literature']:
        for p in (R/folder).glob('*router-author-falsification*'):
            if p.name in ['archive_router_author_falsification.py']:paths.add(p)
            elif p.is_file():paths.add(p)
            else:paths.update(x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    for pattern in ['*router_author_falsification*.py']:
        for folder in ['src','scripts']:paths.update((R/folder).glob(pattern))
    manifest=R/'provenance/router-author-falsification-final-manifest-v0.json';archive=R/'exports/router-author-falsification-final-v0.tar.gz';assert not manifest.exists() and not archive.exists()
    files=[dict(path=p.relative_to(R).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths) if p.is_file()]
    save(manifest,dict(utc=now(),files=files,parent_archives='See logs/router-author-final-cloud-verification.json for prior-round immutable bundles. New snapshot includes this-round checkpoints, sources, logs, literature, analyses and report.'))
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for row in files:t.add(R/row['path'],arcname=row['path'])
        t.add(manifest,arcname=manifest.relative_to(R).as_posix())
    meta=dict(utc=now(),archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,files=len(files));save(R/'provenance/router-author-falsification-final-archive-v0.json',meta);print(json.dumps(meta))
def verify(a):
    archive=a.archive.resolve();assert sha(archive)==a.sha256
    dest=R/'provenance/router-author-falsification-final-cloud-snapshot-v0';assert not dest.exists();dest.mkdir()
    with tarfile.open(archive) as t:
        for m in t.getmembers():assert (dest/m.name).resolve().is_relative_to(dest)
        t.extractall(dest,filter='data')
    manifest=json.loads((dest/'provenance/router-author-falsification-final-manifest-v0.json').read_text(encoding='utf-8'))
    for row in manifest['files']:assert sha(dest/row['path'])==row['sha256']
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not compute,compute
    # Existing scientific raw files may be checked, never silently replaced by different content.
    protected_names={'checkpoint.pt','events.jsonl','fit-result.json','config.json','initialization.pt','evaluation-data.pt','evaluation.json'}
    for row in manifest['files']:
        target=R/row['path'];source=dest/row['path']
        if target.exists() and ('/source/' in row['path'] or target.name in protected_names):assert sha(target)==row['sha256'],row['path']
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==row['sha256']
    result=dict(utc=now(),status='passed',archive_sha256=a.sha256,files=len(manifest['files']),cloud_snapshot=str(dest),all_snapshot_hashes_verified=True,all_live_files_verified=True,gpu_compute_processes=[],pod_stopped=False,billing_checked=False)
    save(R/'logs/router-author-falsification-final-cloud-verification.json',result);print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-cloud',action='store_true');p.add_argument('--archive',type=Path);p.add_argument('--sha256');a=p.parse_args();verify(a) if a.verify_cloud else create()
