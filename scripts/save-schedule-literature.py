"""Persist public primary-source snapshots, not a claim of full-text reading."""
import concurrent.futures,hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'literature/sparse-schedule-2026-09-14';OUT.mkdir(parents=True,exist_ok=True)
SOURCES=[
('msa.html','https://arxiv.org/html/2606.13392v1','Sections 3.2, B.2-B.4; warmup and stop-gradient explicitly known.'),
('cerebras.html','https://www.cerebras.ai/blog/compressing-kv-cache-memory-by-half-with-sparse-attention','One-shot sparse layer selection heuristic; permanent dense-layer selection already known.'),
('liu-recall.html','https://kindxiaoming.github.io/blog/2026/sparse-attention-6/','Read article text; delayed recall and hyperparameter sensitivity already explored.'),
('liu-noise.html','https://kindxiaoming.github.io/blog/2026/sparse-attention-8/','Read article text; changing/randomizing embeddings already explored.'),
('longlora.pdf','https://proceedings.iclr.cc/paper_files/paper/2024/file/211ab571cc9f3802afa6ffff52ae3e5b-Paper-Conference.pdf','Relevant method and Table 6 only; not a full-text audit.'),
('rrattention.pdf','https://aclanthology.org/2026.acl-long.1199.pdf','Title/abstract and first pages only; inference scope, not a full-text audit.'),
('ksa.html','https://github.com/awni/k_sparse_attention','README algorithm and reported experiments read. Direct top-k-score gradient and training without dense warmup already publicly claimed; not independently reproduced.'),
('emergence-v2.html','https://arxiv.org/html/2505.17863v2','Abstract, Section 2.2 and Appendix C.2/D.3-D.4 checked. Plateau dynamics and two-layer induction roles are prior work; proofs not fully audited.')]
def fetch(row):
    name,url,scope=row;record=dict(file=name,url=url,reading_scope=scope,retrieved_utc=datetime.now(timezone.utc).isoformat())
    try:
        if (OUT/name).exists():
            prior=json.loads((OUT/'sources.json').read_text(encoding='utf-8'))
            old=next((r for r in prior if r['file']==name),None)
            if old and old.get('sha256')==hashlib.sha256((OUT/name).read_bytes()).hexdigest():return old
        request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 research-evidence-archive'})
        with urllib.request.urlopen(request,timeout=25) as response:data=response.read();record.update(status=response.status,content_type=response.headers.get('Content-Type'),final_url=response.url)
        (OUT/name).write_bytes(data);record.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as e:record['error']=repr(e)
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,SOURCES))
(OUT/'sources.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(records,ensure_ascii=False))
