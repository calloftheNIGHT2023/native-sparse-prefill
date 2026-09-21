from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,tarfile
R=Path(__file__).resolve().parents[1]
def main():
 c=json.loads((R/'logs/cloud-connection-current.json').read_text());ssh=['ssh','-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15','-p',str(c['ssh_port']),c['ssh_user']+'@'+c['ssh_host']]
 code=r'''from pathlib import Path
from datetime import datetime,timezone
import os,signal,time,json,subprocess,tarfile,hashlib,io
r=Path(ROOT);pid=68048;o=r/'results/fresh-book256-stage-v0';assert o.is_dir()
if Path(f'/proc/{pid}/cmdline').exists():
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'run_fresh_book256_stage_v0.py' in cmd and os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM)
for i in range(10):
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if not gpu:break
 time.sleep(1)
assert not gpu,'GPU still busy; do not stop Pod yet'
p=json.loads((r/'provenance/fresh-book256-protocol-v0.json').read_text());rows=[]
for j in p['jobs']:
 d=o/j['name'];v=d/'result.json';f=d/'task-predictions.jsonl'
 if not d.exists():continue
 result=json.loads(v.read_text()) if v.exists() else {};rows.append(dict(name=j['name'],status=result.get('status','interrupted_by_user'),saved_predictions=len(f.read_text().splitlines()) if f.exists() else 0,has_result=v.exists()))
summary=dict(status='interrupted_by_user',utc=datetime.now(timezone.utc).isoformat(),gpu_compute_apps_empty=True,optimizer_updates=0,jobs=rows,scope='User stop; unfinished condition is not a scientific failure. Do not treat partial comparison as completed prespecified analysis.')
(r/'logs/fresh-book256-user-stop-remote.json').write_text(json.dumps(summary,indent=2)+'\n')
a=r/'exports/fresh-book256-user-stop-v0.tar.gz';assert not a.exists();files=list(o.rglob('*'))+[r/'provenance/fresh-book256-protocol-v0.json',r/'logs/fresh-book256-user-stop-remote.json',r/'logs/fresh-book256-controller-v0.log',r/'logs/fresh-book256-queue-v0.json']+[r/n for n in {**p['source_sha256'],**p['data_sha256']}];entries=[]
with tarfile.open(a,'w:gz') as t:
 for f in sorted(set(files)):
  if not f.is_file():continue
  raw=f.read_bytes();n=f.relative_to(r).as_posix();entries.append(dict(path=n,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)));m=tarfile.TarInfo(n);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
 raw=json.dumps(dict(files=entries)).encode();m=tarfile.TarInfo('user-stop-manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
proof=dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest(),bytes=a.stat().st_size,files=len(entries));a.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(summary=summary,archive=proof)))
'''.replace('ROOT',repr(c['root']))
 q=subprocess.run(ssh+['python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=70);assert q.returncode==0,q.stderr;result=json.loads(q.stdout);print(json.dumps(result),flush=True)
 scp=['scp','-i',c['ssh_key_path'],'-o','BatchMode=yes','-P',str(c['ssh_port'])]
 for ext in ['json','gz']:
  name='fresh-book256-user-stop-v0.tar.'+ext;subprocess.run(scp+[c['ssh_user']+'@'+c['ssh_host']+':'+c['root']+'/exports/'+name,str(R/'exports'/name)],check=True,timeout=150)
 a=R/'exports/fresh-book256-user-stop-v0.tar.gz';proof=result['archive'];assert hashlib.sha256(a.read_bytes()).hexdigest()==proof['sha256'];dest=R/'results/cloud-fresh-book256-user-stop-v0';dest.mkdir(exist_ok=False)
 with tarfile.open(a) as t:
  es=json.loads(t.extractfile('user-stop-manifest.json').read())['files'];assert len(es)==proof['files'];assert len(t.getnames())==len(set(t.getnames()))==len(es)+1
  for e in es:
   m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts;raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
 result.update(local_backup_verified=True,verified_utc=datetime.now(timezone.utc).isoformat());(R/'logs/fresh-book256-user-stop-backup.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(local_backup_verified=True,archive=proof)))
if __name__=='__main__':main()
