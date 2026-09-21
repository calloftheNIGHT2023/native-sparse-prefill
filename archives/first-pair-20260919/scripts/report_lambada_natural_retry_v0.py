"""Audit natural target-word scoring and prespecified paired retention comparison."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    name='lambada-natural-retry';a=R/f'exports/{name}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes'];dest=R/f'results/cloud-{name}-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        mn=name+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t];assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts;raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/f'provenance/{name}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{name}-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    data=dest/'data/lambada-natural-v0';meta=load(data/'tasks.json');arr=np.load(data/'tasks.npz');flat=arr['input_ids'];off=arr['offsets'];assert len(meta)==512 and len(off)==513
    manifest=load(data/'manifest.json');raw=[json.loads(s) for s in (data/'source.jsonl').read_text(encoding='utf-8').splitlines()]
    import random
    order=list(range(5153));random.Random(2026091705).shuffle(order);assert manifest['sample_indices']==order[:512]==[x['item_id'] for x in meta]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
    for i,m in enumerate(meta):
        text=raw[m['item_id']]['text'];context=' '.join(text.split(' ')[:-1]);seq=flat[off[i]:off[i+1]]
        assert len(seq)==m['length'] and np.array_equal(seq,tok.encode(text,add_special_tokens=False)) and len(tok.encode(context,add_special_tokens=False))==m['context_length'] and len(seq)-m['context_length']==m['target_length'] and hashlib.sha256(seq.tobytes()).hexdigest()==m['input_sha256']
    out=dest/f'results/{name}-stage-v0';control=load(out/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
    base=load(out/'base-ability/result.json');gate=base['base_gate'];assert len(control['jobs'])==(9 if gate['passed'] else 1)
    records=[];by={};count=0
    for j in p['jobs'][:len(control['jobs'])]:
        d=out/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['identity']==j['identity'] and v['evaluation_k']==0 and v['optimizer_updates']==0 and v['environment']['gpu']==p['evaluation_gpu']
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_lambada_natural_retry_v0.py')==sha(d/'source.py') and v['checkpoint_sha256']==j['sha256']
        mirror=R/('results/cloud-expanded76-evidence-v0' if j['step']==0 else 'results/cloud-matched-restore-training-evidence-v0')
        assert sha(mirror/j['path'])==j['sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        assert max(abs(x-y) for x,y in zip(v['calibration_values'],j['calibration_values']))<=1e-6 and len(v['calibration_values'])==4 and v['calibration_max_abs_error']<=1e-6
        params=torch.load(mirror/j['path'],map_location='cpu',weights_only=False)['params'];selected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']} if j['restore_qk'] else set();assert set(v['selected_restore_B'])==selected and v['other_params_unchanged']
        for n in selected:params[n].zero_()
        assert hashlib.sha256(b''.join(x.numpy().tobytes() for x in params.values())).hexdigest()==v['parameter_digest'];del params
        preds=v['predictions'];assert preds==[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()] and len(preds)==v['task_predictions']
        variants=['full','short32'] if j['phase']=='base_gate' else ['full'];assert len(preds)==512*len(variants)
        for vi,variant in enumerate(variants):
            rows=preds[vi*512:(vi+1)*512]
            for i,(x,m) in enumerate(zip(rows,meta)):
                seq=flat[off[i]:off[i+1]];assert all(x[k]==value for k,value in m.items()) and x['variant']==variant and x['target_token_ids']==seq[m['context_length']:].tolist() and x['context_tokens_used']==(m['context_length'] if variant=='full' else min(32,m['context_length']))
                assert len(x['predicted_token_ids'])==len(x['token_nll'])==m['target_length'] and np.isfinite(x['token_nll']).all() and all(z>=0 for z in x['token_nll']) and abs(sum(x['token_nll'])-x['target_nll_sum'])<1e-4
                assert x['correct']==(x['predicted_token_ids']==x['target_token_ids'])
            vals=np.array([x['correct'] for x in rows],dtype=float);nll=np.array([x['target_nll_sum'] for x in rows]);records.append(dict(name=j['name'],variant=variant,accuracy=float(vals.mean()),word_perplexity=float(np.exp(nll.mean())),correct=int(vals.sum())))
            by[(j.get('training_k',0),j['seed'],j['step'],j['restore_qk'],variant)]=(vals,nll)
        count+=len(preds)
    b=by[(0,p['jobs'][0]['seed'],0,False,'full')][0];s=by[(0,p['jobs'][0]['seed'],0,False,'short32')][0]
    assert abs(gate['full_accuracy']-b.mean())<1e-12 and abs(gate['short32_accuracy']-s.mean())<1e-12 and gate['passed']==bool(b.mean()>=.20 and b.mean()-s.mean()>=.05)
    assert count==control['task_predictions']==(5120 if gate['passed'] else 1024)
    contrasts=[];passed=None
    if gate['passed']:
        idx=np.random.default_rng(p['primary']['bootstrap_seed']).integers(0,512,(p['primary']['draws'],512))
        for label,ar,br in [('primary',True,False),('symmetric_restored',True,True),('original',False,False)]:
            for metric,mi in [('accuracy',0),('target_word_nll',1)]:
                delta=np.stack([by[(32,seed,128,ar,'full')][mi]-by[(0,seed,128,br,'full')][mi] for seed in [2026091660,2026091661]]).mean(axis=0);ci=np.quantile(delta[idx].mean(axis=1),[.025,.975]);contrasts.append(dict(contrast=label,metric=metric,difference=float(delta.mean()),conditional_item95ci=ci.tolist()))
        passed=next(x for x in contrasts if x['contrast']=='primary' and x['metric']=='accuracy')['conditional_item95ci'][0]>-.05
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=count,calibration_nll_forwards=len(control['jobs'])*4,base_gate=gate,records=records,contrasts=contrasts,natural_accuracy_screen_pass=passed,scope=p['scope']);target=R/f'results/{name}-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# LAMBADA真实原文末词预测：固定512题','','全词表逐token贪心一致性要求，整个末词都对才算正确；与四选一不同。最大147tokens，不是32K评测。基座全上下文/末32tokens能力门槛：'+json.dumps(gate,ensure_ascii=False),'','|模型|上下文|正确/512|正确率|目标词困惑度|','|---|---|---:|---:|---:|']
    for x in records:lines.append(f"|{x['name']}|{x['variant']}|{x['correct']}|{x['accuracy']*100:.2f}%|{x['word_perplexity']:.4f}|")
    lines+=['','预设自然词预测准确率非劣筛查：'+str(passed)+'。','',json.dumps(contrasts,ensure_ascii=False,indent=2),'','首次本项目固定子集评测，来源原始书籍分组不可见，逐题bootstrap可能低估同书相关性；仅两个已有训练种子，未知基座预训练接触。不是官方全测试集，也不是广泛长程能力或原创贡献证明。若基座门槛失败，不运行训练后对照，更不能将其记为稀疏失败。']
    (R/'docs/lambada-natural-retry-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',base_gate=gate,count=count,natural_accuracy_screen_pass=passed,contrasts=contrasts)))
if __name__=='__main__':main()
