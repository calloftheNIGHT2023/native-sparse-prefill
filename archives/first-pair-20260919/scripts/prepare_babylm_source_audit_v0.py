"""Download pinned BabyLM Strict-Small source only; no tokenization or training."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,hashlib
R=Path(__file__).resolve().parents[1]
REV='c92ab16b4f08858304b0815706065b3354d8fc0a'
def one(e):
 name=e['path'];assert '/' not in name and name.endswith('.train.txt')
 out=R/'data/babylm-2026-strict-small-raw-v0'/name
 url=f'https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small/resolve/{REV}/{name}'
 if out.exists():raw=out.read_bytes()
 else:
  raw=urllib.request.urlopen(url,timeout=90).read()
  assert len(raw)==e['size']
  if 'lfs' in e:assert hashlib.sha256(raw).hexdigest()==e['lfs']['oid']
  else:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['oid']
  out.write_bytes(raw)
 assert len(raw)==e['size']
 if 'lfs' in e:assert hashlib.sha256(raw).hexdigest()==e['lfs']['oid']
 else:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['oid']
 text=raw.decode('utf-8');lengths=sorted(len(s.split()) for s in text.splitlines() if s.strip());words=sum(lengths)
 return dict(path=name,url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),word_count_whitespace=words,nonempty_lines=len(lengths),line_word_p50=lengths[len(lengths)//2],line_word_p95=lengths[int(.95*(len(lengths)-1))],line_word_max=max(lengths),lines_at_least_2048_words=sum(n>=2048 for n in lengths),lines_at_least_8192_words=sum(n>=8192 for n in lengths),scope='Lines are observed text records, not verified document boundaries. Words are not tokenizer tokens. No long-context claim from concatenation.')
def main():
 out=R/'data/babylm-2026-strict-small-raw-v0';out.mkdir(exist_ok=True)
 tree=json.loads((R/'literature/babylm-reboot-2026-09-17/small-data-tree.json').read_text());entries=[e for e in tree if e['path'].endswith('.train.txt')];assert len(entries)==6
 with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,entries))
 result=dict(status='verified_raw_sources_only',utc=datetime.now(timezone.utc).isoformat(),dataset='BabyLM-community/BabyLM-2026-Strict-Small',revision=REV,files=rows,total_bytes=sum(x['bytes'] for x in rows),total_words_whitespace=sum(x['word_count_whitespace'] for x in rows),training_started=False,tokenizer_trained=False,scope='Raw corpus audit before experimental design. Distinct source lines are not document IDs, and nominal10M designation is not assumed to equal measured whitespace word count. No test data downloaded or used.')
 (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
