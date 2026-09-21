"""Deploy one immutable launch archive; do not retry uncertain remote launches."""
from pathlib import Path
import argparse,json,re,subprocess
R=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('stage');a=p.parse_args();stage=a.stage
    assert re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',stage)
    c=json.loads((R/'logs/cloud-connection-current.json').read_text(encoding='utf-8'));host=c['ssh_user']+'@'+c['ssh_host'];opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15']
    archive=R/f'exports/{stage}-launch-v0.tar.gz';proof=json.loads(archive.with_suffix('.json').read_text());receipt=R/f'logs/{stage}-launch-v0.json';assert not receipt.exists()
    subprocess.run(['scp',*opts,'-P',str(c['ssh_port']),str(archive),host+':'+c['root']+'/exports/'+archive.name],check=True,timeout=120)
    code='''from pathlib import Path
import json,tarfile,hashlib,subprocess
r=Path(ROOT);stage=STAGE;a=r/f'exports/{stage}-launch-v0.tar.gz'
assert hashlib.sha256(a.read_bytes()).hexdigest()==HASH
with tarfile.open(a) as t:
 entries=json.loads(t.extractfile('migration-manifest.json').read())['files']
 assert len(entries)==FILES
 for e in entries:
  n=e['path'];assert not Path(n).is_absolute() and '..' not in Path(n).parts
  m=t.getmember(n);assert m.isfile();raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
  f=r/n;f.parent.mkdir(parents=True,exist_ok=True)
  if f.exists():assert f.read_bytes()==raw,n
  else:f.write_bytes(raw)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not (r/f'results/{stage}-stage-v0').exists()
assert not (r/f'logs/{stage}-controller-v0.pid').exists()
with (r/f'logs/{stage}-controller-v0.log').open('w') as f:
 p=subprocess.Popen([PYTHON,'-u',str(r/('scripts/run_'+stage.replace('-','_')+'_stage_v0.py'))],cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/f'logs/{stage}-controller-v0.pid').write_text(str(p.pid))
print(json.dumps({'status':'launched','pid':p.pid,'stage':stage}))
'''.replace('ROOT',repr(c['root'])).replace('STAGE',repr(stage)).replace('HASH',repr(proof['sha256'])).replace('FILES',str(proof['files'])).replace('PYTHON',repr(c['python']))
    v=subprocess.run(['ssh',*opts,'-p',str(c['ssh_port']),host,'python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=45)
    assert v.returncode==0,v.stderr[-2000:];launch=json.loads(v.stdout);receipt.write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch))
if __name__=='__main__':main()
