"""Audit prospective common-dense factual confirmation including unsuccessful ability gates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/fresh-word-dense-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-fresh-word-dense-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('fresh-word-dense-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'fresh-word-dense-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/fresh-word-dense-protocol-v0.json';assert sha(pp)==sha(R/'provenance/fresh-word-dense-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    ds=dest/'data/fresh-word-dense-v0';meta=load(ds/'tasks.json');manifest=load(ds/'manifest.json');arr=np.load(ds/'tasks.npz');xs=arr['input_ids'];off=arr['offsets'];assert len(meta)==133 and len(off)==134 and off[-1]==len(xs)
    backgrounds=manifest['backgrounds'];assert len(backgrounds)==32;seen=set();prior=set(manifest['prior_article_hashes'])
    for bg in backgrounds:
        hs={x['article_sha256'] for x in bg['articles']};assert len(hs)==len(bg['articles']) and hs.isdisjoint(seen|prior);seen|=hs
    assert len(seen)==manifest['article_count']
    families=sorted({x['family_id'] for x in meta if x['variant']=='long32768'});assert len(families)==32
    for family in families:
        idx=[i for i,x in enumerate(meta) if x['family_id']==family];assert len(idx)==4 and [meta[i]['gold'] for i in idx]==[0,1,2,3]
        seqs=[xs[off[i]:off[i+1]] for i in idx];assert all(len(z)==32768 for z in seqs) and all(np.count_nonzero(seqs[0]!=z)==1 for z in seqs[1:])
    stage=dest/'results/fresh-word-dense-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs']) in [1,5];records=[];by={};count=0
    for j in p['jobs'][:len(control['jobs'])]:
        d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['evaluation_k']==0 and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_fresh_word_dense_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        old=load(mirror/j['training_result']);assert old['identity']==v['identity'];cal=next(e['values'] for e in old['evaluations'] if e['step']==j['step'] and e['split']=='calibration')
        assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        rows=v['predictions'];assert len(rows)==v['task_predictions'] and rows==[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()]
        for pred,m in zip(rows,meta):
            assert all(pred[k]==value for k,value in m.items()) and len(pred['choice_logits'])==4 and np.isfinite(pred['choice_logits']).all() and pred['prediction']==int(np.argmax(pred['choice_logits']))
            if m['gold'] is not None:assert pred['correct']==(pred['prediction']==m['gold'])
        short=[x for x in rows if x['variant']=='short'];assert len(short)==4 and v['short_gate']['passed']==all(x['correct'] for x in short)
        long=[x for x in rows if x['variant']=='long32768'];matrix=None
        if long:
            assert len(long)==128 and v['long_gate']['passed']==(sum(x['correct'] for x in long)>=96)
            matrix=np.array([[next(x['correct'] for x in long if x['family_id']==f and x['gold']==gold) for gold in range(4)] for f in families],dtype=float)
        records.append(dict(name=j['name'],k=j['k'],seed=j['seed'],step=j['step'],short_correct=sum(x['correct'] for x in short),long_correct=int(matrix.sum()) if matrix is not None else None,long_accuracy=float(matrix.mean()) if matrix is not None else None,families_all_four_correct=int(matrix.all(axis=1).sum()) if matrix is not None else None))
        by[(j['k'],j['seed'],j['step'])]=matrix;count+=len(rows)
    base=load(stage/'base-ability/result.json');gate=base['short_gate']['passed'] and bool(base['long_gate'] and base['long_gate']['passed'])
    assert count==control['task_predictions'] and ((gate and len(records)==5 and count==665) or (not gate and len(records)==1 and count in [5,133]))
    comparison=None
    if gate:
        ds=np.stack([by[(32,seed,128)]-by[(0,seed,128)] for seed in [2026091660,2026091661]])
        delta=ds.mean(axis=(0,2));idx=np.random.default_rng(2026091673).integers(0,32,(20000,32));ci=np.quantile(delta[idx].mean(axis=1)*100,[.025,.975]).tolist()
        comparison=dict(sparse_minus_dense_pp=float(delta.mean()*100),paired_background_95ci_pp=ci,by_seed_pp=(ds.mean(axis=(1,2))*100).tolist(),prospective_margin_pp=5,noninferiority_pass=ci[0]>-5,dense_accuracy=float(np.mean([by[(0,s,128)].mean() for s in [2026091660,2026091661]])),sparse_accuracy=float(np.mean([by[(32,s,128)].mean() for s in [2026091660,2026091661]])))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=count,base_ability_gate_passed=gate,records=records,comparison=comparison,scope=p['scope']);out=R/'results/fresh-word-dense-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 新背景、新答案词：统一密集注意力事实读取检查','',f"原始模型能力门槛：{'通过' if gate else '未通过，停止后续比较'}。新背景由{manifest['article_count']}篇文章构成，32组间无相同文章，排除了此前四批背景池；四个答案词orange/purple/rabbit/planet，每个背景四种事实，只改变一个证据token。所有模型评测均为密集注意力。",'','|检查点|短题正确/4|长题正确/128|四种事实全对的背景数/32|','|---|---:|---:|---:|']
    for x in records:lines.append(f"|{x['name']}|{x['short_correct']}|{x['long_correct']}|{x['families_all_four_correct']}|")
    if comparison:
        x=comparison;lines+=['',f"两种子平均：密集训练{x['dense_accuracy']*100:.2f}%，稀疏训练{x['sparse_accuracy']*100:.2f}%；差{x['sparse_minus_dense_pp']:+.2f}pp，背景配对95%区间{x['paired_background_95ci_pp']}pp。事先固定的5pp非劣门槛：{'通过' if x['noninferiority_pass'] else '未通过'}。"]
    lines+=['','这是已知模板上的新背景/新答案值确认，不是新任务类型、官方RULER或自由生成成绩。模型预训练接触未知，未排除所有近重复；两个已有训练种子，统计区间只覆盖背景抽样，不等同于训练种子总体不确定性。若能力门槛失败，不能将后续未测记为稀疏失败。成本结果来自单独的匹配设置诊断，不将二者合并成已经完成的同质量总成本证明。']
    (R/'docs/fresh-word-dense-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',task_predictions=count,base_gate=gate,comparison=comparison)))
if __name__=='__main__':main()
