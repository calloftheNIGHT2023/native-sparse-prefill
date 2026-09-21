"""Single-shot, bounded small-evidence backup; no model calls or process signals."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TAG = 'babylm-optimization-stage-a-20260920-v0'


def main():
    c = json.loads((ROOT / 'logs/cloud-connection-one-epoch-current.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT / 'logs' / (TAG + '-observation-' + stamp)
    destination.mkdir(exist_ok=False)
    archive_name = TAG + '-' + stamp + '.tar.gz'
    previous = sorted((ROOT / 'logs').glob(TAG + '-observation-*/backup-receipt.json'))
    reuse_files = {}
    if previous:
        old = json.loads(previous[-1].read_text())
        for entry in old['files']:
            path = previous[-1].parent / 'raw' / entry['path']
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
                reuse_files[entry['path']] = (entry['sha256'], path)
    known_hashes = {name: item[0] for name, item in reuse_files.items()}
    code = r'''
from pathlib import Path
import hashlib,io,json,tarfile
from datetime import datetime,timezone
r=Path(ROOT_VALUE)
stage=r/'results/babylm-optimization-stage-a-20260920-v0'
launch=r/'logs/optimization-stage-a-20260920-v0-launch'
target=Path('/tmp')/ARCHIVE_VALUE
known=KNOWN_VALUE
records=[]; total=0
with tarfile.open(target,'w:gz') as t:
 for parent in (stage,launch):
  if not parent.exists(): continue
  for p in sorted(parent.rglob('*')):
   if not p.is_file() or p.suffix not in ('.json','.jsonl','.log'): continue
   blob=p.read_bytes(); original_bytes=len(blob); tail=0
   if p.suffix=='.jsonl' and blob and not blob.endswith(b'\n'):
    end=blob.rfind(b'\n')+1; tail=len(blob)-end; blob=blob[:end]
   total+=len(blob); assert total<=256*1024*1024,'Bounded log backup exceeded256MB'
   name=p.relative_to(r).as_posix(); digest=hashlib.sha256(blob).hexdigest(); reuse=known.get(name)==digest
   if not reuse:
    info=tarfile.TarInfo(name); info.size=len(blob); t.addfile(info,io.BytesIO(blob))
   records.append({'path':name,'bytes':len(blob),'sha256':digest,'reused_local_sha':reuse,'live_bytes_observed':original_bytes,'incomplete_jsonl_tail_bytes_excluded':tail})
 identities=[]
 state_path=stage/'stage.json'
 if state_path.exists():
  state=json.loads(state_path.read_text()); pids={state['pid']}
  pids.update(j['pid'] for j in state.get('jobs',[]) if j.get('status')=='running' and 'pid' in j)
  for pid in sorted(pids):
   proc=Path('/proc')/str(pid)/'stat'
   if not proc.exists(): identities.append({'pid':pid,'present':False}); continue
   fields=proc.read_text().rpartition(') ')[2].split()
   identities.append({'pid':pid,'present':True,'state':fields[0],'ppid':int(fields[1]),'pgid':int(fields[2]),'session':int(fields[3]),'start_ticks':int(fields[19])})
 receipt={'observed_utc':datetime.now(timezone.utc).isoformat(),'files':records,'total_bytes':total,'model_calls':0,'process_signals':0,'process_identities':identities}
 data=(json.dumps(receipt,indent=2)+'\n').encode(); info=tarfile.TarInfo('backup-manifest.json'); info.size=len(data);t.addfile(info,io.BytesIO(data))
print(json.dumps({'archive':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'observed_utc':receipt['observed_utc'],'files':len(records),'bytes':total}))
'''.replace('ROOT_VALUE', repr(c['root'])).replace('ARCHIVE_VALUE', repr(archive_name)).replace('KNOWN_VALUE', repr(known_hashes))
    # The generated payload is ASCII-safe shell code, never environment/token text.
    command = c['python'] + ' -c ' + shlex.quote('import base64;exec(base64.b64decode(' + repr(base64.b64encode(code.encode()).decode()) + '))')
    options = ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12', '-i', c['ssh_key_path']]
    remote = c['ssh_user'] + '@' + c['ssh_host']
    result = subprocess.run(['ssh', '-T', *options, '-p', str(c['ssh_port']), remote, command],
                            capture_output=True, text=True, timeout=60, check=True)
    meta = json.loads(result.stdout)
    local_archive = destination / archive_name
    subprocess.run(['scp', '-q', *options, '-P', str(c['ssh_port']), remote + ':' + meta['archive'], str(local_archive)],
                   check=True, timeout=90)
    assert hashlib.sha256(local_archive.read_bytes()).hexdigest() == meta['sha256']
    with tarfile.open(local_archive) as t:
        manifest = json.load(t.extractfile('backup-manifest.json'))
        for item in manifest['files']:
            out = (destination / 'raw' / item['path']).resolve()
            assert out.is_relative_to((destination / 'raw').resolve())
            if item.get('reused_local_sha'):
                blob = reuse_files[item['path']][1].read_bytes()
            else:
                member = t.getmember(item['path'])
                assert member.isfile()
                blob = t.extractfile(member).read()
            assert len(blob) == item['bytes'] and hashlib.sha256(blob).hexdigest() == item['sha256']
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('xb') as f:
                f.write(blob)
    manifest.update(archive_sha256_verified=True, individual_files_sha256_verified=True,
                    local_directory=destination.relative_to(ROOT).as_posix())
    (destination / 'backup-receipt.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stage = destination / 'raw/results' / TAG / 'stage.json'
    state = json.loads(stage.read_text()) if stage.exists() else {}
    print(json.dumps({'observed_utc':meta['observed_utc'], 'backup':str(destination), 'files_verified':len(manifest['files']),
                      'unchanged_files_reused':sum(bool(x.get('reused_local_sha')) for x in manifest['files']),
                      'status':state.get('status'), 'jobs':[{k:j.get(k) for k in ('name','status','replay_passed')} for j in state.get('jobs', [])]}))


if __name__ == '__main__':
    main()
