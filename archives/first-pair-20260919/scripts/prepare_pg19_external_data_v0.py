"""Select books without observing model outcomes; retain official generation and content hashes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,random,base64,requests,numpy as np
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'data/pg19-external-v0';assert not (out/'manifest.json').exists()
    listing=json.loads((out/'official-test-listing.json').read_text());assert len(listing['items'])==100 and not listing.get('nextPageToken')
    items=sorted(listing['items'],key=lambda x:x['name']);random.Random(2026091702).shuffle(items)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
    prior=np.load(R/'data/32k-expanded-training-v0/train-calibration.npz');old_report=np.load(R/'data/32k-expanded-training-v0/report.npz')['report']
    known=[np.asarray(w,dtype=np.int32).tobytes() for a in [prior['train'],prior['calibration'],old_report] for w in a]
    records=[];selected=[];windows=[];seen=set();rawdir=out/'raw';rawdir.mkdir(exist_ok=True)
    for item in items:
        url='https://storage.googleapis.com/deepmind-gutenberg/'+item['name'];f=rawdir/Path(item['name']).name
        if not f.exists():
            response=requests.get(url,params={'generation':item['generation']},timeout=40);response.raise_for_status();f.write_bytes(response.content)
        raw=f.read_bytes();assert len(raw)==int(item['size']) and base64.b64encode(hashlib.md5(raw).digest()).decode()==item['md5Hash']
        text=raw.decode('utf-8');ids=tok.encode(text,add_special_tokens=False)
        rec=dict(book_id=f.stem,url=url,generation=item['generation'],raw_sha256=sha(f),raw_bytes=len(raw),tokens=len(ids),selected=False)
        if len(ids)<36865:rec['reason']='insufficient_tokens_after_fixed4096offset'
        else:
            w=np.asarray(ids[4096:36865],dtype=np.int32);assert len(w)==32769;h=hashlib.sha256(w.tobytes()).hexdigest()
            # Screening exact 128-token passages at fixed64-token strides. This is
            # not a claim to exhaustive near-duplicate or pretraining exclusion.
            matches=[]
            for offset in range(0,len(w)-128+1,64):
                needle=w[offset:offset+128].tobytes()
                if any(needle in old for old in known):matches.append(offset)
            if matches:rec.update(reason='exact128token_passage_overlap',overlap_offsets=matches)
            elif h in seen:rec['reason']='duplicate_window'
            else:
                seen.add(h);rec.update(selected=True,window_index=len(windows),token_offset=4096,window_sha256=h)
                windows.append(w);selected.append(dict(rec))
        records.append(rec)
        print(json.dumps(dict(book=f.stem,selected=rec['selected'],selected_count=len(windows))),flush=True)
        if len(windows)==16:break
    assert len(windows)==16
    np.savez_compressed(out/'windows.npz',windows=np.stack(windows))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),dataset='Official Google DeepMind PG19 test',source='https://github.com/google-deepmind/pg19',listing_sha256=sha(out/'official-test-listing.json'),selection_seed=2026091702,selection='Lexicographically sort100 officialtest names, shuffle Python Random2026091702, choose first16 unique eligible books; >=36865tokens, fixed offset4096 and32769tokens; fixed128token passage screen against current CPT train/cal/old report; no model output used.',selected=selected,considered=records,windows_sha256=sha(out/'windows.npz'),prior_data_sha256={n:sha(R/n) for n in ['data/32k-expanded-training-v0/train-calibration.npz','data/32k-expanded-training-v0/report.npz']},scope='First project evaluation on these external book windows. Not full official PG19 benchmark, not official word-level PPL: report Qwen-token NLL/PPL for a prespecified16-book sample. Books may have appeared in unknown base-model pretraining. Exact passage screening at fixed offsets is not exhaustive near-duplicate elimination. New corpus evaluation conditional on already selected method and existing two training seeds; not a claim of broad downstream or seed generality.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='prepared',books=16,considered=len(records),manifest_sha256=sha(out/'manifest.json'))))
if __name__=='__main__':main()
