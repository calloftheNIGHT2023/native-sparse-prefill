import hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'literature/router-capacity-2026-09-14';root.mkdir(exist_ok=False)
sources=[('rtpurbo.html','https://arxiv.org/html/2605.16928v1','Sections 8.2 and 9.1; targeted overlap read, not full-paper proof audit'),('msa.html','https://arxiv.org/html/2606.13392v1','Sections 3.2, B.4 and C.3; teacher/support/training scope'),('ksa.html','https://github.com/awni/k_sparse_attention','README algorithm and training description; author claims not independent reproduction')]
rows=[]
for name,url,scope in sources:
    try:
        data=urllib.request.urlopen(url,timeout=25).read();(root/name).write_bytes(data);rows.append(dict(url=url,path=name,utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(data).hexdigest(),reading_scope=scope))
    except Exception as e:rows.append(dict(url=url,error=str(e),reading_scope=scope))
(root/'sources.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows))
