"""Independently recompute factual-pair gates and verify the frozen input intervention."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-counterfactual-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-counterfactual-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-counterfactual-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-counterfactual-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-counterfactual-protocol-v0.json';assert sha(pp)==sha(R/'provenance/word-counterfactual-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    stage=dest/'results/word-counterfactual-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
    j=p['jobs'][0];assert len(control['jobs'])==1 and control['jobs'][0]['returncode']==0
    d=stage/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['optimizer_updates']==0
    assert v['eval_source_sha256']==sha(dest/'scripts/eval_word_counterfactual_v0.py')==sha(d/'source.py')
    mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
    train=load(mirror/j['training_result']);assert v['identity']==train['identity'] and v['environment']['gpu']==p['evaluation_gpu']
    assert {n:x for n,x in v['environment'].items() if n!='gpu'}=={n:x for n,x in train['environment'].items() if n!='gpu'}
    cal=next(e['values'] for e in train['evaluations'] if e['step']==0 and e['split']=='calibration');assert max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
    data=dest/'data/word-counterfactual-v0';meta=load(data/'tasks.json');a=np.load(data/'tasks.npz');flat=a['input_ids'];offsets=a['offsets'];groups={}
    for index,x in enumerate(meta):groups.setdefault((x['family_id'],x['variant']),[]).append((x,flat[offsets[index]:offsets[index+1]]))
    assert len(groups)==96 and len(meta)==192
    for (_,variant),rows in groups.items():
        assert len(rows)==2 and rows[0][0]['gold']!=rows[1][0]['gold'];aa,bb=rows[0][1],rows[1][1];assert aa.shape==bb.shape
        assert int(np.count_nonzero(aa!=bb))==(0 if variant=='no_context' else 1)
        if variant=='long32768':assert len(aa)==32768
    predictions=v['predictions'];assert len(predictions)==v['task_predictions']==control['task_predictions']
    assert predictions==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
    for x,y in zip(predictions,meta):
        assert all(x[n]==val for n,val in y.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all()
        assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
    def metrics(variant):
        xs=[x for x in predictions if x['variant']==variant];groups={}
        for x in xs:groups.setdefault(x['family_id'],[]).append(x)
        assert len(xs)==64 and len(groups)==32 and all(len(x)==2 for x in groups.values())
        return dict(correct=sum(x['correct'] for x in xs),total=64,pairs_both_correct=sum(all(y['correct'] for y in x) for x in groups.values()),pairs_total=32)
    def gate(variant,criteria):
        m=metrics(variant);n=metrics('no_context');m['gain_over_no_context']=m['correct']-n['correct'];m['passed']=m['correct']>=criteria['min_correct'] and m['pairs_both_correct']>=criteria['min_pairs_both_correct'] and m['gain_over_no_context']>=criteria['min_gain_over_no_context'];return m
    sg=gate('short',p['short_gate']);assert sg==v['short_gate'] and metrics('no_context')==v['no_context']
    lg=gate('long32768',p['long_gate']) if sg['passed'] else None;assert lg==v['long_gate'] and len(predictions)==(192 if sg['passed'] else 128)
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=len(predictions),short_gate=sg,long_gate=lg,no_context=v['no_context'],control=control,scope=p['scope'])
    out=R/'results/word-counterfactual-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 直接词值反事实：密集底座能力检查','','每组只改一个事实token，正确答案必须随之改变。32组、每组两种答案；在候选词logit中取最大，不是自由生成或正式RULER。0训练更新。','','|输入|正确/64|两版都正确/32|能力门槛|','|---|---:|---:|---|',f"|短原文|{sg['correct']}|{sg['pairs_both_correct']}|{'通过' if sg['passed'] else '未通过'}|",f"|无原文|{v['no_context']['correct']}|{v['no_context']['pairs_both_correct']}|对照|"]
    if lg:lines.append(f"|32K|{lg['correct']}|{lg['pairs_both_correct']}|{'通过' if lg['passed'] else '未通过'}|")
    else:lines.append('|32K|未运行|未运行|短题未通过，按协议停止|')
    lines+=['','这只确定当前底座是否适合本诊断；不证明稀疏质量或论文贡献。全部输入哈希、单token反事实、无原文完全相同、校准NLL和逐题分数已核验；未调模板补考。',f"\n开始UTC：{control['started_utc']}；结束UTC：{control['finished_utc']}；计分前向{len(predictions)}。"]
    (R/'docs/word-counterfactual-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',short_gate=sg,long_gate=lg,no_context=v['no_context'])))
if __name__=='__main__':main()
