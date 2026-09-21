"""Freeze a natural last-word task, with no model-dependent item selection."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,random,numpy as np,requests
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'data/lambada-natural-v0';assert not (out/'manifest.json').exists();raw=(out/'source.jsonl').read_bytes();rows=[json.loads(s) for s in raw.splitlines()];assert len(rows)==5153
    url='https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/lambada/lambada_openai.yaml';r=requests.get(url,timeout=30);r.raise_for_status();(out/'reference-task.yaml').write_bytes(r.content)
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
    order=list(range(len(rows)));random.Random(2026091705).shuffle(order);selected=order[:512];meta=[];flat=[];offsets=[0]
    train=' '.join((R/'data/flashmoba-pool-training-v0/train.txt').read_text(encoding='utf-8').split())
    for i in selected:
        text=rows[i]['text'];context=' '.join(text.split(' ')[:-1]);target=' '+text.split(' ')[-1];assert context+target==text
        ctx=tok.encode(context,add_special_tokens=False);whole=tok.encode(text,add_special_tokens=False);assert whole[:len(ctx)]==ctx and 0<len(ctx)<len(whole)<=2048
        assert ' '.join(text.split()) not in train,'Sample overlap: fail rather than replace a selected item'
        ids=np.asarray(whole,dtype=np.int32);flat.extend(whole);offsets.append(len(flat));meta.append(dict(item_id=i,context_length=len(ctx),target_length=len(whole)-len(ctx),length=len(whole),text_sha256=hashlib.sha256(text.encode()).hexdigest(),input_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    assert len({x['text_sha256'] for x in meta})==512
    np.savez_compressed(out/'tasks.npz',input_ids=np.array(flat,dtype=np.int32),offsets=np.array(offsets,dtype=np.int64));(out/'tasks.json').write_text(json.dumps(meta,indent=2)+'\n')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_url='https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl',source_sha256=sha(out/'source.jsonl'),source_rows=5153,reference_task_url=url,reference_task_sha256=sha(out/'reference-task.yaml'),paper='https://aclanthology.org/P16-1144/',sample_seed=2026091705,sample_indices=selected,records=512,maximum_length=max(x['length'] for x in meta),tokenizer_model_manifest_sha256=sha(R/'data/flashmoba-qwen-precision-v0/manifest.json'),train_text_sha256=sha(R/'data/flashmoba-pool-training-v0/train.txt'),scope='First project evaluation of this fixed512-item LAMBADA OpenAI sample. Natural original last word, full vocabulary greedy token-exact correctness; no four-choice restriction or instruction prompt. Split text at last literal space following the cited harness task; verify concatenated token prefix consistency for every item, no truncation. Base additionally scored with last32 context tokens; others full-context only. This is short/mid-length discourse retention, not32K factual retrieval or full official benchmark. Normalized entire sample absent from current CPT train text; not exhaustive near-duplicate removal, base pretraining exposure unknown. Source lacks book IDs so item bootstrap may understate within-book dependence; two existing seeds.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(status='prepared',records=512,maximum_length=manifest['maximum_length'],manifest_sha256=sha(out/'manifest.json'))))
if __name__=='__main__':main()
