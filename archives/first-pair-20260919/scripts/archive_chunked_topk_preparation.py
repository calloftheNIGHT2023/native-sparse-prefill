"""Create a one-shot archive, verify every member, test its standalone CPU code."""
import hashlib,json,subprocess,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    archive=ROOT/'exports/chunked-topk-preparation-v0.tar.gz'
    verification=ROOT/'exports/chunked-topk-preparation-v0-verified'
    if archive.exists() or verification.exists():raise FileExistsError('Use a new archive version')
    fixed=['STATE.md','TIMELINE.md','logs/control-state.json','src/chunked_topk_attention.py',
           'src/zoology_sparse_schedule.py','tests/test_chunked_topk_attention.py',
           'scripts/benchmark_chunked_topk.py','scripts/preflight_chunked_topk_cuda.py',
           'scripts/verify_chunked_topk_checkpoints.py','scripts/verify_chunked_topk_fullmodel_gradients.py',
           'scripts/diagnose_chunked_topk_numerics.py','scripts/report_chunked_topk_preparation.py',
           'scripts/archive_chunked_topk_preparation.py','docs/sparse-efficiency-preparation-plan-2026-09-15.md',
           'docs/sparse-efficiency-numerical-addendum-2026-09-15.md','docs/chunked-topk-runbook-2026-09-15.md',
           'docs/chunked-topk-preparation-results-2026-09-15.md']
    paths={ROOT/p for p in fixed}
    for folder in (ROOT/'results').glob('chunked-topk-*'):
        if folder.is_dir():paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update((ROOT/'logs').glob('chunked-topk-*.log'))
    paths=sorted(paths)
    manifest=dict(utc=utc(),files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
        standalone_scope=['CPU operator tests','CUDA operator preflight','attention core benchmark'],
        full_model_replay_requires=dict(archive='exports/router-author-falsification-final-v0.tar.gz',
            sha256='f156526d396102236e2a48613d72b3bc02421e85b554ef27712cf14e46dea25f',
            note='Also requires pinned Zoology code/model dependencies from the existing project; no checkpoints duplicated here'),
        failed_trials_preserved=True)
    manifest_path=ROOT/'provenance/chunked-topk-preparation-manifest-v0.json';save(manifest_path,manifest)
    with tarfile.open(archive,'w:gz') as tar:
        for p in paths:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        tar.add(manifest_path,arcname=manifest_path.relative_to(ROOT).as_posix(),recursive=False)
    expected={r['path']:r['sha256'] for r in manifest['files']}
    expected[manifest_path.relative_to(ROOT).as_posix()]=sha(manifest_path)
    verification.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(expected)
        for member in members:
            pp=PurePosixPath(member.name)
            assert member.isfile() and not pp.is_absolute() and '..' not in pp.parts
            assert member.name in expected
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==expected[member.name]
        tar.extractall(verification,filter='data')
    assert all(sha(verification/p)==h for p,h in expected.items())
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_chunked_topk_attention.py','-v'],
        cwd=verification,capture_output=True,text=True,encoding='utf-8',timeout=120)
    log=ROOT/'logs/chunked-topk-portable-archive-tests-v0.log'
    log.write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Extracted archive CPU tests failed: '+str(log))
    proof=dict(utc=utc(),archive=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,
        sha256=sha(archive),files_verified=len(expected),every_archive_member_hash_verified=True,
        every_extracted_file_hash_verified=True,standalone_cpu_tests_returncode=test.returncode,
        tests_log=log.relative_to(ROOT).as_posix(),old_checkpoint_archive_unchanged=True,
        new_cloud_resources=0,scientific_optimizer_updates=0)
    save(ROOT/'provenance/chunked-topk-preparation-archive-v0.json',proof)
    print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
