"""Audit external-book natural language quality without selecting outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(params):return hashlib.sha256(b''.join(x.numpy().tobytes() for x in params.values())).hexdigest()
def main():
    stage='pg19-quality';a=R/f'exports/{stage}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'))
    assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes'];dest=R/f'results/cloud-{stage}-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        mn=stage+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/f'provenance/{stage}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{stage}-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    manifest=load(dest/'data/pg19-external-v0/manifest.json');windows=np.load(dest/'data/pg19-external-v0/windows.npz')['windows'];assert windows.shape==(16,32769)
    assert len(manifest['selected'])==16 and len({x['book_id'] for x in manifest['selected']})==16
    # Reconstruct all selected token windows locally from archived official raw books.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(R/'data/flashmoba-qwen-precision-v0/model',local_files_only=True)
    for w,book in zip(windows,manifest['selected']):
        f=dest/'data/pg19-external-v0/raw'/(book['book_id']+'.txt');assert sha(f)==book['raw_sha256']
        ids=tok.encode(f.read_bytes().decode('utf-8'),add_special_tokens=False)
        assert np.array_equal(w,np.asarray(ids[4096:36865],dtype=np.int32)) and hashlib.sha256(w.tobytes()).hexdigest()==book['window_sha256']
    out=dest/f'results/{stage}-stage-v0';control=load(out/'result.json')
    assert control['status']=='complete' and control['optimizer_updates']==control['task_predictions']==control['gradient_passes']==0 and control['protocol_sha256']==sha(pp) and len(control['jobs'])==9
    records=[];by={};count=0
    metrics=['full_nll','full_context_tail_nll','short_context_tail_nll','context_benefit_nats']
    for j in p['jobs']:
        d=out/j['name'];v=load(d/'result.json')
        assert v['status']=='complete' and v['job']==j and v['evaluation_k']==0 and v['optimizer_updates']==v['task_predictions']==0 and v['nll_forwards']==52
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_pg19_quality_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu'] and v['weights_unchanged']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        trained=load(mirror/j['training_result']);assert trained['identity']==v['identity']
        cal=next(e['values'] for e in trained['evaluations'] if e['step']==j['step'] and e['split']=='calibration')
        assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        cp=torch.load(mirror/j['path'],map_location='cpu',weights_only=False);params=cp['params']
        if j['restore_qk']:
            abl=load(d/'ablation.json');assert abl==v['ablation'] and abl['selected_count']==48 and abl['unchanged_other_tensors']==144 and abl['restored_to_pretraining_QK']
            selected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']}
            assert set(abl['selected_B_tensors'])==selected
            for n in selected:
                assert hashlib.sha256(params[n].numpy().tobytes()).hexdigest()==abl['prior_B_sha256'][n];params[n].zero_()
        else:assert v['ablation'] is None
        assert digest(params)==v['parameter_digest'];del cp,params
        rows=v['windows'];assert len(rows)==16 and rows==[json.loads(x) for x in (d/'window-nll.jsonl').read_text().splitlines()]
        for i,(row,w) in enumerate(zip(rows,windows)):
            assert row['window']==i and row['input_sha256']==hashlib.sha256(w.tobytes()).hexdigest() and all(np.isfinite(row[m]) for m in metrics)
            assert abs(row['context_benefit_nats']-(row['short_context_tail_nll']-row['full_context_tail_nll']))<1e-12
        by[(j['k'],j['seed'],j['step'],j['restore_qk'])]={m:np.array([x[m] for x in rows]) for m in metrics}
        records.append(dict(name=j['name'],k=j['k'],seed=j['seed'],step=j['step'],restore_qk=j['restore_qk'],means={m:float(np.mean([x[m] for x in rows])) for m in metrics},token_perplexity=float(np.exp(np.mean([x['full_nll'] for x in rows])))))
        count+=52
    assert count==p['expected_nll_forwards']==468
    idx=np.random.default_rng(p['primary']['bootstrap_seed']).integers(0,16,(p['primary']['bootstrap_draws'],16));seeds=[2026091660,2026091661]
    contrasts=[]
    comparisons=[('restored_sparse_vs_original_dense',(32,True),(0,False)),('restored_sparse_vs_restored_dense',(32,True),(0,True)),('original_sparse_vs_original_dense',(32,False),(0,False)),('sparse_restoration_effect',(32,True),(32,False)),('dense_restoration_effect',(0,True),(0,False))]
    for name,(ak,ar),(bk,br) in comparisons:
        for m in metrics:
            ds=np.stack([by[(ak,s,128,ar)][m]-by[(bk,s,128,br)][m] for s in seeds]);delta=ds.mean(axis=0);ci=np.quantile(delta[idx].mean(axis=1),[.025,.975])
            rec=dict(contrast=name,metric=m,difference_nats=float(delta.mean()),book_paired_95ci=ci.tolist(),by_seed_nats=ds.mean(axis=1).tolist())
            if m!='context_benefit_nats':rec.update(ppl_ratio=float(np.exp(delta.mean())),ppl_ratio_95ci=np.exp(ci).tolist())
            contrasts.append(rec)
    primary=[x for x in contrasts if x['contrast']==p['primary']['contrast'].split(' at')[0].replace(' ','_') and x['metric'] in p['primary']['metrics']]
    assert len(primary)==2
    passed=all(x['ppl_ratio_95ci'][1]<=p['primary']['ppl_ratio_noninferiority_upper'] for x in primary)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=0,nll_forwards=count,records=records,contrasts=contrasts,external_sample_joint_pass=passed,scope=p['scope'])
    target=R/'results/pg19-quality-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# PG19 外部真实长篇文本质量检查','','事先固定16本官方测试书籍，每本固定偏移4096后的32769个token。所有模型统一密集评测，固定128步终点；无训练。这里是Qwen分词的16书子集结果，不是官方按词困惑度或完整PG19榜单。','','|模型|完整32K NLL|token困惑度|结尾4K NLL（完整上下文）|长上下文收益nats|','|---|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['name']}|{x['means']['full_nll']:.6f}|{x['token_perplexity']:.4f}|{x['means']['full_context_tail_nll']:.6f}|{x['means']['context_benefit_nats']:.6f}|")
    lines+=['','主要对照：恢复后稀疏训练模型与原密集训练模型。整体与结尾两个指标的困惑度比值95%上限都须不超过1.05；本次通过='+str(passed)+'。']
    for x in primary:lines+=['',json.dumps(x,ensure_ascii=False)]
    lines+=['','区间以16本书为配对抽样单位，先平均两个种子。基座预训练接触未知；两个种子和一个外部语料不能证明全面泛化。零Q/K操作已有前作，不属于原创贡献。所有逐书指标、恢复/未恢复对照、重放和哈希保存在审计JSON。通过本筛查仍不能替代自然任务理解与同质量实际训练成本的完整证据。']
    (R/'docs/pg19-quality-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified',nll_forwards=count,primary=primary,external_sample_joint_pass=passed)))
if __name__=='__main__':main()
