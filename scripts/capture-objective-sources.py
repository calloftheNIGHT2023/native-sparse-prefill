"""Save narrow primary-source screening evidence; downloads are not full readings."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parents[1]
out=root/'literature/objective-screen-2026-09-14'
out.mkdir(exist_ok=False)
sources={
 'value-aware.html':'https://aclanthology.org/2021.emnlp-main.753/',
 'ssa.html':'https://arxiv.org/html/2511.20102v1',
 'stem.html':'https://arxiv.org/html/2603.06274v1',
 'cobs.html':'https://arxiv.org/html/2607.09052v1',
 'causal-routing.html':'https://arxiv.org/html/2607.21692v1',
 'kvpop.html':'https://arxiv.org/html/2607.05061v2'
}
manifest=[]
for name,url in sources.items():
 started=datetime.now(timezone.utc).isoformat()
 try:
  data=urllib.request.urlopen(url,timeout=45).read()
  (out/name).write_bytes(data)
  item={'path':name,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'saved'}
 except Exception as ex:
  item={'path':name,'url':url,'status':'failed','error':str(ex)}
 manifest.append({'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),**item})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
