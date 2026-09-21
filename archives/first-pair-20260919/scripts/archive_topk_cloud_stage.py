"""One-shot portable archive for this completed efficiency stage."""
import hashlib,json,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    archive=ROOT/'exports/topk-efficiency-cloud-stage-v0.tar.gz'
    if archive.exists():raise FileExistsError('Use a new archive version')
    fixed=['STATE.md','TIMELINE.md','.gitignore','logs/control-state.json','requirements-router-portable.txt',
        'configs/research-spending-cap-v1.json',
        'src/chunked_topk_attention.py','src/zoology_sparse_schedule.py','src/router_author_control.py','src/zoology_entry.py',
        'src/triton_selected_attention.py','src/triton_topk_selector.py','src/triton_streaming_selector.py',
        'scripts/benchmark_chunked_topk.py','scripts/benchmark_topk_cuda_graphs.py','scripts/benchmark_topk_fullmodel.py',
        'scripts/preflight_chunked_topk_cuda.py','scripts/verify_triton_selected_attention.py','scripts/verify_triton_topk_selector.py',
        'scripts/verify_streaming_topk_selector.py','scripts/verify_chunked_topk_checkpoints.py','scripts/evaluate_topk_precision.py',
        'scripts/report_topk_cloud_efficiency.py','scripts/archive_topk_cloud_stage.py','scripts/verify_topk_stage_archive.py',
        'tests/test_chunked_topk_attention.py','docs/topk-cloud-efficiency-results-2026-09-15.md',
        'docs/topk-cloud-literature-boundary-2026-09-15.md','docs/topk-cloud-runbook-2026-09-15.md',
        'results/router-author-falsification-lowlr-v0/checkpoint.pt','results/router-author-falsification-confirm-v0/checkpoint.pt',
        'results/router-author-falsification-evaluation-v0/evaluation.json','results/router-author-falsification-evaluation-v0/evaluation-data.pt']
    paths={ROOT/p for p in fixed}
    for pattern in ['topk-*','triton-*','chunked-topk-checkpoint-local-cuda-v0','chunked-cuda-gate-v0']:
        for directory in (ROOT/'results').glob(pattern):
            if directory.is_dir():paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for pattern in ['topk-*','triton-*','chunked-topk-checkpoint-local-cuda-v0.log','chunked-cuda-gate-v0.log',
                    'cloud-environment-freeze-v0.txt','operator-unit-tests-v0.log','model-dependencies-v0.log']:
        paths.update(p for p in (ROOT/'logs').glob(pattern) if p.is_file())
    for directory in ['third_party/zoology-1ad20d1','literature/topk-efficiency-cloud-2026-09-15']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest=dict(utc=datetime.now(timezone.utc).isoformat(),files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)],
        scope='This engineering stage, two frozen checkpoints and exposed eval set; no new training data or private keys')
    mp=ROOT/'provenance/topk-efficiency-cloud-stage-manifest-v0.json'
    mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with tarfile.open(archive,'w:gz') as t:
        for p in sorted(paths):t.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        t.add(mp,arcname=mp.relative_to(ROOT).as_posix(),recursive=False)
    expected={r['path']:r['sha256'] for r in manifest['files']};expected[mp.relative_to(ROOT).as_posix()]=sha(mp)
    with tarfile.open(archive,'r:gz') as t:
        ms=t.getmembers();assert len(ms)==len(expected)
        for m in ms:
            assert m.isfile() and m.name in expected and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
            assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==expected[m.name]
    proof=dict(utc=datetime.now(timezone.utc).isoformat(),archive=archive.relative_to(ROOT).as_posix(),
        sha256=sha(archive),bytes=archive.stat().st_size,all_members_verified=True,members=len(expected))
    (ROOT/'provenance/topk-efficiency-cloud-stage-archive-v0.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof))

if __name__=='__main__':main()
