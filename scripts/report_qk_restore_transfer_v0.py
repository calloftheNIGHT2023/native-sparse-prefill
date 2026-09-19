"""Audit immutable intervention evidence and all prespecified paired contrasts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    stage='qk-restore-transfer';a=R/f'exports/{stage}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'))
    assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/f'results/cloud-{stage}-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        mn=stage+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
            f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/f'provenance/{stage}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{stage}-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    out=dest/f'results/{stage}-stage-v0';control=load(out/'result.json')
    assert control['status']=='complete' and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp) and len(control['jobs'])==4
    meta=load(dest/'data/fresh-word-dense-v0/tasks.json');families=sorted({x['family_id'] for x in meta if x['variant']=='long32768'})
    assert len(meta)==133 and len(families)==32
    def matrix(rows):
        assert len(rows)==133
        for pred,m in zip(rows,meta):
            assert all(pred[k]==v for k,v in m.items()) and len(pred['choice_logits'])==4 and np.isfinite(pred['choice_logits']).all()
            assert pred['prediction']==int(np.argmax(pred['choice_logits']))
            if m['gold'] is not None:assert pred['correct']==(pred['prediction']==m['gold'])
        return np.array([[next(x['correct'] for x in rows if x['family_id']==f and x['gold']==g) for g in range(4)] for f in families],dtype=float)
    by={};records=[];count=0;short_ok=True
    for j in p['jobs']:
        d=out/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['evaluation_k']==0 and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_qk_restore_transfer_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-expanded76-evidence-v0'
        assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256']
        trained=load(mirror/j['training_result']);assert v['identity']==trained['identity']
        cal=next(e['values'] for e in trained['evaluations'] if e['step']==128 and e['split']=='calibration')
        assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        abl=load(d/'ablation.json');assert v['ablation']==abl and abl['selected_count']==48 and abl['unchanged_other_tensors']==144 and abl['restored_to_pretraining_QK']
        expected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']}
        assert set(abl['selected_B_tensors'])==expected and set(abl['prior_B_sha256'])==expected
        import torch
        cp=torch.load(mirror/j['path'],map_location='cpu',weights_only=False)
        assert all(hashlib.sha256(cp['params'][n].numpy().tobytes()).hexdigest()==abl['prior_B_sha256'][n] for n in expected);del cp
        rows=v['predictions'];assert rows==[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()] and v['task_predictions']==133
        arr=matrix(rows);by[('restored',j['k'],j['seed'])]=arr
        ref=next(x for x in p['references'] if x['k']==j['k'] and x['seed']==j['seed']);old=load(dest/ref['path'])
        assert old['checkpoint_sha256']==j['sha256'] and old['evaluation_k']==0
        before=matrix(old['predictions']);by[('original',j['k'],j['seed'])]=before
        short=sum(x['correct'] for x in rows if x['variant']=='short');assert v['short_gate']['correct']==short
        short_ok &= short==4
        records.append(dict(k=j['k'],seed=j['seed'],short_correct=short,original_correct=int(before.sum()),restored_correct=int(arr.sum()),all_four_correct=int(arr.all(axis=1).sum())))
        count+=133
    assert count==control['task_predictions']==p['expected_task_predictions']
    seeds=[2026091660,2026091661]
    means={(m,k):np.stack([by[(m,k,s)] for s in seeds]).mean(axis=(0,2)) for m in ['original','restored'] for k in [0,32]}
    idx=np.random.default_rng(p['primary']['bootstrap_seed']).integers(0,32,(p['primary']['bootstrap_draws'],32))
    def summary(x):return dict(difference_pp=float(x.mean()*100),paired_background_95ci_pp=(np.quantile(x[idx].mean(axis=1),[.025,.975])*100).tolist())
    contrasts=dict(restored_sparse_vs_original_dense=summary(means['restored',32]-means['original',0]),restored_sparse_vs_restored_dense=summary(means['restored',32]-means['restored',0]),sparse_restore_effect=summary(means['restored',32]-means['original',32]),dense_restore_effect=summary(means['restored',0]-means['original',0]),restoration_interaction=summary((means['restored',32]-means['original',32])-(means['restored',0]-means['original',0])))
    passed=short_ok and contrasts['restored_sparse_vs_original_dense']['paired_background_95ci_pp'][0]>-5
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,task_predictions=count,optimizer_updates=0,calibration_nll_forwards=16,records=records,contrasts=contrasts,development_screen_pass=passed,scope=p['scope'])
    target=R/'results/qk-restore-transfer-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# QK-Restore 在第二批背景和答案词上的开发验证','','这是已知恢复操作的开发验证；本批题此前已暴露，不能称为独立确认。四个原有128步检查点，无新增训练；所有模型统一密集评测。','','|训练K|种子|恢复前正确/128|恢复后正确/128|恢复后短题/4|','|---|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['k']}|{x['seed']}|{x['original_correct']}|{x['restored_correct']}|{x['short_correct']}|")
    lines+=['','主要对照：恢复后的稀疏训练模型，相对于未恢复的密集训练模型。开发筛查通过='+str(passed)+'。','',json.dumps(contrasts,ensure_ascii=False,indent=2),'','区间按32个背景配对抽样，条件于两个训练种子和同一个人工任务模板；非独立泛化证据。原检查点未改写，48个Q/K LoRA B仅在内存清零，144个其他张量逐一检查不变。校准16次，任务预测532次。实际同质量训练总成本与外部自然任务尚待验证，已有QK-Restore不是原创贡献。']
    (R/'docs/qk-restore-transfer-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified',task_predictions=count,contrasts=contrasts,development_screen_pass=passed)))
if __name__=='__main__':main()
