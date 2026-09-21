"""Archive this completed/failed diagnostic stage and its exact candidate wheel."""
import hashlib
import json
import os
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = Path('/opt/native-sparse-flashmoba-env-v2')
BUILD = ROOT/'results/flashmoba-backward-environment-v5'
CAND = ROOT/'third_party/flash-moba-barrier-candidate-v0'
OUT = ROOT/'results/flashmoba-backward-package-v0'
OUT.mkdir(parents=True, exist_ok=False)
started = datetime.now(timezone.utc).isoformat()
(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
build = json.loads((BUILD/'environment.json').read_text())
assert build['status']=='complete'
assert (ROOT/'results/flashmoba-backward-stage-controller-v0/result.json').exists()
wheel_dir = ROOT/'exports/flashmoba-barrier-wheels-v0'
wheel_dir.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(json.loads((BUILD/'build-env.json').read_text()))
env['PATH'] = env['CUDA_HOME']+'/bin:'+str(ENV/'bin')+':'+env['PATH']
with (OUT/'wheel.log').open('w') as f:
    subprocess.run([str(ENV/'bin/python'), 'setup.py', 'bdist_wheel', '--dist-dir', str(wheel_dir)],
                   cwd=CAND, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600)
wheels = list(wheel_dir.glob('*.whl'))
assert len(wheels)==1
import zipfile
with zipfile.ZipFile(wheels[0]) as z:
    so = [n for n in z.namelist() if n.startswith('flash_moba_cuda') and n.endswith('.so')]
    assert len(so)==1
    assert hashlib.sha256(z.read(so[0])).hexdigest()==build['candidate_extension_sha256']
manifest = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                wheel_sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
                extension_sha256=build['candidate_extension_sha256'],
                wheel_path=str(wheels[0].relative_to(ROOT)),
                original_wheel_preserved=True,
                label='isolated synchronization candidate; use the recorded correctness results, not the package version, to assess validity')
(wheel_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
files = set()
for pattern in ['flashmoba-backward-*', 'flashmoba-long-backward-*', 'flashmoba-newpod-standard-*']:
    for directory in (ROOT/'results').glob(pattern):
        if directory.is_dir():
            for p in directory.rglob('*'):
                if p.is_file() and not any(x.startswith('migrated-') for x in p.relative_to(directory).parts):
                    files.add(p)
for directory in [wheel_dir, ROOT/'data/flashmoba-backward-fixed-v0', ROOT/'data/flashmoba-backward-training-v0']:
    files.update(p for p in directory.rglob('*') if p.is_file())
for pattern in ['*backward*.py', 'verify_flashmoba_official_v1.py']:
    files.update((ROOT/'scripts').glob(pattern))
files.add(ROOT/'patches/flashmoba-shared-index-barrier-candidate-v0.patch')
rows = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
mf=OUT/'archive-manifest.json'
mf.write_text(json.dumps(rows, indent=2)+'\n')
archive=Path('/workspace/flashmoba-backward-cloud-stage-v0.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(files | {mf}):
        t.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
proof=dict(archive=str(archive), bytes=archive.stat().st_size,
           sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=len(files)+1,
           created_utc=datetime.now(timezone.utc).isoformat())
(Path('/workspace')/'flashmoba-backward-cloud-stage-v0.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
