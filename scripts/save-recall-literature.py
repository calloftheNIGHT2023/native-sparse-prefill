import hashlib,json,urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
class Plain(HTMLParser):
    def __init__(self): super().__init__(); self.text=[]
    def handle_data(self,data):
        if data.strip(): self.text.append(data.strip())
def main():
    out=ROOT/'literature/far-recall-check-2026-09-14'; out.mkdir(exist_ok=False)
    papers=[('mad','https://arxiv.org/html/2403.17844v2','Targeted sections 3.1.1-3.1.3 and Appendix B.1, not full-paper review'),
        ('ruler','https://arxiv.org/html/2404.06654v3','Introduction and retrieval task descriptions; not full-paper review'),
        ('zoology','https://arxiv.org/abs/2312.04927','Abstract only; HTML v2 URL returned404')]
    def fetch(p):
        name,url,scope=p
        raw=urllib.request.urlopen(url,timeout=45).read(); path=out/(name+'.html'); path.write_bytes(raw)
        text=Plain(); text.feed(raw.decode('utf-8')); (out/(name+'.txt')).write_text('\n'.join(text.text),encoding='utf-8')
        return dict(name=name,url=url,sha256=hashlib.sha256(raw).hexdigest(),reading_scope=scope)
    with ThreadPoolExecutor(max_workers=3) as pool: records=list(pool.map(fetch,papers))
    (out/'manifest.json').write_text(json.dumps(dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),sources=records),indent=2)+'\n')
    print(json.dumps(records))
if __name__=='__main__': main()
