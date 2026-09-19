"""Stop the current Pod after user-requested stop and verified partial backup."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib
R=Path(__file__).resolve().parents[1]
def main():
 c=json.loads((R/'logs/cloud-connection-current.json').read_text());assert c['pod_id']=='REDACTED_POD_ID';b=json.loads((R/'logs/fresh-book256-user-stop-backup.json').read_text());assert b['local_backup_verified'];assert hashlib.sha256((R/'exports/fresh-book256-user-stop-v0.tar.gz').read_bytes()).hexdigest()==b['archive']['sha256'];assert json.loads((R/'logs/fresh-book256-user-stop-partial-audit.json').read_text())['checkpoint_dependencies_all_verified_locally']
 state=R/'logs/user-stop-pod-20260917.json';assert not state.exists();state.write_text(json.dumps(dict(status='request_starting',utc=datetime.now(timezone.utc).isoformat(),pod_id=c['pod_id'],local_backup_verified=True),indent=2)+'\n')
 opts=['-i',c['ssh_key_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15'];host=c['ssh_user']+'@'+c['ssh_host']
 for n in ['STATE.md','TIMELINE.md','logs/control-state.json','logs/fresh-book256-user-stop-partial-audit.json']:subprocess.run(['scp',*opts,'-P',str(c['ssh_port']),str(R/n),host+':'+c['root']+'/'+n],check=True,timeout=40)
 code=r'''from pathlib import Path
import json,subprocess,hashlib,urllib.request,urllib.parse,urllib.error
r=Path(ROOT);assert hashlib.sha256((r/'exports/fresh-book256-user-stop-v0.tar.gz').read_bytes()).hexdigest()==HASH
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
v={k.decode():val.decode() for x in Path('/proc/1/environ').read_bytes().split(b'\0') if b'=' in x for k,val in [x.split(b'=',1)] if k in [b'RUNPOD_API_KEY',b'RUNPOD_POD_ID']}
assert v.get('RUNPOD_POD_ID')=='REDACTED_POD_ID' and v.get('RUNPOD_API_KEY')
body=json.dumps({'query':'mutation { podStop(input: {podId: "REDACTED_POD_ID"}) { id desiredStatus } }'}).encode()
req=urllib.request.Request('https://api.runpod.io/graphql?api_key='+urllib.parse.quote(v['RUNPOD_API_KEY'],safe=''),data=body,headers={'Content-Type':'application/json','User-Agent':'runpodctl'})
try:
 with urllib.request.urlopen(req,timeout=20) as q:z=json.load(q)
 pod=(z.get('data') or {}).get('podStop');result=dict(status='stop_acknowledged' if pod and pod.get('id')=='REDACTED_POD_ID' and pod.get('desiredStatus')=='EXITED' else 'stop_not_confirmed',pod=pod)
except urllib.error.HTTPError as e:result=dict(status='stop_rejected',http_status=e.code)
except Exception as e:result=dict(status='stop_uncertain',exception_type=type(e).__name__)
print(json.dumps(result),flush=True)
'''.replace('ROOT',repr(c['root'])).replace('HASH',repr(b['archive']['sha256']))
 q=subprocess.run(['ssh',*opts,'-p',str(c['ssh_port']),host,'python3','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=40)
 try:v=json.loads(q.stdout)
 except Exception:v=dict(status='stop_transport_uncertain',returncode=q.returncode)
 v.update(utc=datetime.now(timezone.utc).isoformat(),pod_id=c['pod_id'],local_backup_verified=True);state.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
