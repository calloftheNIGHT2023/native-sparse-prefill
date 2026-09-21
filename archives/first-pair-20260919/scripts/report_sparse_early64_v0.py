"""Audit early dense checkpoints and the prespecified five-endpoint cost control."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    name='sparse-early64';a=R/f'exports/{name}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes'];dest=R/f'results/cloud-{name}-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        mn=name+'-manifest.json';entries=json.loads(t.extractfile(mn).read())['files'];names=[m.name for m in t];assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{mn}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts;raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/f'provenance/{name}-protocol-v0.json';assert sha(pp)==sha(R/f'provenance/{name}-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    out=dest/f'results/{name}-stage-v0';control=load(out/'result.json');assert control['status']=='complete' and control['optimizer_updates']==0 and control['task_predictions']==2580 and control['protocol_sha256']==sha(pp) and len(control['jobs'])==4
    wm=load(dest/'data/fresh-word-dense-v0/tasks.json');lm=load(dest/'data/lambada-natural-v0/tasks.json');la=np.load(dest/'data/lambada-natural-v0/tasks.npz');pg=np.load(dest/'data/pg19-external-v0/windows.npz')['windows'];families=sorted({x['family_id'] for x in wm if x['variant']=='long32768'})
    def word(rows):
        assert len(rows)==133
        for x,m in zip(rows,wm):
            assert all(x[k]==v for k,v in m.items()) and len(x['choice_logits'])==4 and np.isfinite(x['choice_logits']).all() and x['prediction']==int(np.argmax(x['choice_logits']))
            if m['gold'] is not None:assert x['correct']==(x['prediction']==m['gold'])
        return np.array([[next(x['correct'] for x in rows if x['family_id']==f and x['gold']==g) for g in range(4)] for f in families],dtype=float).mean(axis=1)
    def lambada(rows):
        assert len(rows)==512
        for i,(x,m) in enumerate(zip(rows,lm)):
            assert all(x[k]==v for k,v in m.items()) and x['variant']=='full' and x['context_tokens_used']==m['context_length']
            seq=la['input_ids'][la['offsets'][i]:la['offsets'][i+1]];assert x['target_token_ids']==seq[m['context_length']:].tolist() and x['correct']==(x['predicted_token_ids']==x['target_token_ids']) and len(x['token_nll'])==m['target_length'] and np.isfinite(x['token_nll']).all()
        return np.array([x['correct'] for x in rows],dtype=float)
    def quality(w,l,wn,pr):
        assert len(wn)==9 and np.isfinite(wn).all() and len(pr)==16
        for i,(x,seq) in enumerate(zip(pr,pg)):
            assert x['window']==i and x['input_sha256']==hashlib.sha256(seq.tobytes()).hexdigest() and all(np.isfinite(x[k]) for k in ['full_nll','full_context_tail_nll','short_context_tail_nll'])
        return dict(word=word(w),lambada=lambada(l),wiki=np.array(wn),pg_full=np.array([x['full_nll'] for x in pr]),pg_tail=np.array([x['full_context_tail_nll'] for x in pr]))
    by={};costs=[];records=[];sparse={}
    for seed in [2026091660,2026091661]:
        refs={x['kind']:load(dest/x['path']) for x in p['references'] if x['seed']==seed};v=refs['dense64']
        assert v['status']=='complete' and v['job']['training_k']==0 and v['job']['seed']==seed and v['step']==64 and not v['job']['restore_qk'] and v['evaluation_k']==0
        sparse[seed]=quality(v['word_predictions'],v['predictions'],v['wiki_nll'],v['pg19'])
        sparse[seed]['seconds']=v['job']['early_process_seconds_utc']
    for j in p['jobs']:
        assert j['training_k']==32 and j['identity']['training_k']==32
        d=out/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['identity']==j['identity'] and v['step']==64 and v['evaluation_k']==0 and v['task_predictions']==645 and v['nll_forwards']==61 and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_sparse_early64_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
        mirror=R/'results/cloud-matched-restore-training-evidence-v0';assert sha(mirror/j['path'])==j['sha256']==v['checkpoint_sha256'] and sha(mirror/j['training_result'])==j['training_result_sha256'];tr=load(mirror/j['training_result'])
        assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(v['calibration_values'],j['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
        cp=torch.load(mirror/j['path'],map_location='cpu',weights_only=False);assert cp['step']==cp['data_cursor']==64 and cp['identity']==j['identity'];params=cp['params']
        selected={f'model.layers.{i}.self_attn.{q}.B' for i in range(24) for q in ['q_proj','k_proj']} if j['restore_qk'] else set();assert set(v['selected_restore_B'])==selected and v['other_params_unchanged']
        for n in selected:params[n].zero_()
        if selected:
            artifact=torch.load(d/'restored-adapter.pt',map_location='cpu',weights_only=False);assert artifact['step']==64 and all(torch.equal(artifact['params'][n],params[n]) for n in params)
        assert hashlib.sha256(b''.join(x.numpy().tobytes() for x in params.values())).hexdigest()==v['parameter_digest']
        preds=[json.loads(s) for s in (d/'task-predictions.jsonl').read_text().splitlines()];assert preds==v['word_predictions']+v['predictions'] and v['pg19']==[json.loads(s) for s in (d/'pg19-nll.jsonl').read_text().splitlines()]
        q=quality(v['word_predictions'],v['predictions'],v['wiki_nll'],v['pg19']);by[(j['seed'],j['restore_qk'])]=q
        events=[json.loads(s) for s in (dest/j['timing_events_path']).read_text().splitlines()];e=next(x for x in events if x['event']=='checkpoint_reload_verified' and x['step']==64);seconds=(datetime.fromisoformat(e['utc'])-datetime.fromisoformat(tr['started_utc'])).total_seconds();assert abs(seconds-j['early_process_seconds_utc'])<1e-8 and abs(j['early_step_seconds']-sum(x['seconds'] for x in tr['rows'][:64]))<1e-8
        assert np.isfinite(v['restore_seconds']) and v['restore_seconds']>=0;seconds+=v['restore_seconds'];sp=sparse[j['seed']]['seconds'];costs.append(dict(seed=j['seed'],restore_qk=j['restore_qk'],early_seconds_utc_plus_restore=seconds,dense64_seconds_utc=sp,saving_percent=100*(1-seconds/sp),still_cheaper_with_one_second_penalty=seconds+1<sp))
        records.append(dict(seed=j['seed'],restore_qk=j['restore_qk'],word_accuracy=float(q['word'].mean()),lambada_accuracy=float(q['lambada'].mean()),wiki_ppl=float(np.exp(q['wiki'].mean())),pg19_ppl=float(np.exp(q['pg_full'].mean()))))
    rng=np.random.default_rng(p['primary']['bootstrap_seed']);contrasts=[];screens=[]
    for restore in [False,True]:
        checks={}
        for metric in ['word','lambada','wiki','pg_full','pg_tail']:
            delta=np.stack([by[(s,restore)][metric]-sparse[s][metric] for s in [2026091660,2026091661]]).mean(axis=0);idx=rng.integers(0,len(delta),(p['primary']['draws'],len(delta)));ci=np.quantile(delta[idx].mean(axis=1),p['primary']['quantiles']);x=dict(sparse64_restore=restore,metric=metric,difference=float(delta.mean()),conditional97_5ci=ci.tolist())
            if metric in ['word','lambada']:checks[metric]=bool(ci[0]>-.05)
            else:x.update(ppl_ratio=float(np.exp(delta.mean())),ppl_ratio_ci=np.exp(ci).tolist());checks[metric]=bool(np.exp(ci[1])<=1.05)
            contrasts.append(x)
        checks['cost']=all(x['still_cheaper_with_one_second_penalty'] for x in costs if x['restore_qk']==restore);screens.append(dict(restore_qk=restore,checks=checks,all=all(checks.values())))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=2580,nll_forwards=244,records=records,cost=costs,contrasts=contrasts,candidate_screens=screens,cheaper_sparse_candidate_passes=any(x['all'] for x in screens),scope=p['scope']);target=R/f'results/{name}-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 稀疏64步与密集64步：补齐同预算成本曲线','','相同4090、两种子、64更新，全部按密集注意力评测。使用已保存检查点，未新增训练。原权重与已知QK恢复两个固定候选对比密集64原权重；不是独立确认。','','|种子|恢复QK|长题正确率|LAMBADA正确率|Wiki困惑度|PG19困惑度|','|---|---|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['seed']}|{x['restore_qk']}|{x['word_accuracy']*100:.2f}%|{x['lambada_accuracy']*100:.2f}%|{x['wiki_ppl']:.4f}|{x['pg19_ppl']:.4f}|")
    lines+=['','联合筛查：'+json.dumps(screens,ensure_ascii=False),'','成本：'+json.dumps(costs,ensure_ascii=False),'','两准确率非劣容差5pp、三PPL容差5%，两个候选均用97.5%配对区间；种子先平均，按背景/题目/窗口/书籍重采样。成本双方均按UTC至64检查点重载，稀疏另加实际恢复保存；附加1秒成本仍须更便宜。与128步单调计时不同。没有挑选有利指标，未把未知算相同，未证明最优成本。已知恢复基线不作为原创。']
    (R/'docs/sparse-early64-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',screens=screens,cost=costs)))
if __name__=='__main__':main()
