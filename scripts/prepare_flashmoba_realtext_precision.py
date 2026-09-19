"""Pin existing public model/data assets and deterministic new 2049-token windows."""
import argparse,hashlib,json,shutil,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def main(args):
    started=utc();tick=time.perf_counter()
    cfg=json.loads(args.config.read_text())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source_manifest=json.loads((ROOT/cfg['asset_manifest']).read_text())
    selected=[]
    for r in source_manifest:
        p=ROOT/r['path'].replace('\\','/')
        if p.parent==ROOT/cfg['model_source'] or p==ROOT/cfg['dataset_source']:
            assert sha(p)==r['sha256'],p
            selected.append(r)
    model_dir=out/'model';model_dir.mkdir()
    for p in (ROOT/cfg['model_source']).iterdir():
        if p.is_file():shutil.copy2(p,model_dir/p.name)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    texts=pq.read_table(ROOT/cfg['dataset_source'],columns=['text']).column('text').to_pylist()
    if cfg.get('text_protocol')=='natural_newlines':
        stream=tokenizer('\n\n'.join(texts),add_special_tokens=False)['input_ids']
    else:
        stream=[]
        for text in texts:
            if text.strip():stream.extend(tokenizer(text,add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id])
    width=max(cfg['lengths'])+1
    windows=np.asarray(stream[:len(stream)//width*width],dtype=np.int64).reshape(-1,width)
    chosen=np.random.default_rng(cfg['selection_seed']).permutation(len(windows))[:cfg['contexts']]
    assert len(chosen)==cfg['contexts']
    np.save(out/'tokens.npy',windows[chosen])
    shutil.copy2(args.config,out/'config.json')
    shutil.copy2(__file__,out/'preparation-source.py')
    manifest=dict(started_utc=started,finished_utc=utc(),seconds=time.perf_counter()-tick,
        source_assets=selected,chosen_nonoverlapping_window_indices=chosen.tolist(),
        available_windows=len(windows),stream_token_count=len(stream),text_protocol=cfg.get('text_protocol','insert_eos_per_nonempty_row'),
        separator_eos=None if cfg.get('text_protocol')=='natural_newlines' else tokenizer.eos_token_id,
        lengths_are_nested_prefixes=True,earlier_work_used_same_test_corpus=True,
        files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=args.archive.resolve()
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=out.relative_to(ROOT).as_posix())
    proof=dict(utc=utc(),archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,contexts=len(chosen))
    (ROOT/'provenance'/archive.name.replace('.tar.gz','.json')).write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/flashmoba-realtext-precision-v0.json')
    p.add_argument('--output',type=Path,default=ROOT/'data/flashmoba-realtext-precision-v0')
    p.add_argument('--archive',type=Path,default=ROOT/'exports/flashmoba-realtext-input-v0.tar.gz');main(p.parse_args())
