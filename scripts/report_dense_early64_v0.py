"""Audit early dense checkpoints and the prespecified five-endpoint cost control."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    name='dense-early64';a=R/f'exports/{name}-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes'];dest=R/f'results/cloud-{name}-evidence-v0';dest.mkdir(exist_ok=True)
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
        refs={x['kind']:load(dest/x['path']) for x in p['references'] if x['seed']==seed};v=refs['matched'];lr=refs['lambada'];cc=next(x for x in v['conditions'] if x['name']=='restored')
        assert v['job']['training_k']==32 and v['job']['seed']==seed and v['step']==128 and lr['job']['seed']==seed and lr['job']['restore_qk'] and lr['job']['step']==128 and lr['parameter_digest']==cc['parameter_digest']
        sparse[seed]=quality(cc['predictions'],lr['predictions'],cc['wiki_nll'],cc['pg19']);sparse[seed]['seconds']=v['restored_artifact_seconds']
    for j in p['jobs']:
        d=out/j['name'];v=load(d/'result.json');assert v['status']=='complete' and v['job']==j and v['identity']==j['identity'] and v['step']==64 and v['evaluation_k']==0 and v['task_predictions']==645 and v['nll_forwards']==61 and v['optimizer_updates']==0
        assert v['eval_source_sha256']==sha(dest/'scripts/eval_dense_early64_v0.py')==sha(d/'source.py') and v['environment']['gpu']==p['evaluation_gpu']
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
        assert np.isfinite(v['restore_seconds']) and v['restore_seconds']>=0;seconds+=v['restore_seconds'];sp=sparse[j['seed']]['seconds'];costs.append(dict(seed=j['seed'],restore_qk=j['restore_qk'],early_seconds_utc_plus_restore=seconds,sparse128_seconds_monotonic_plus_restore=sp,saving_percent=100*(1-seconds/sp),still_cheaper_with_one_second_penalty=seconds+1<sp))
        records.append(dict(seed=j['seed'],restore_qk=j['restore_qk'],word_accuracy=float(q['word'].mean()),lambada_accuracy=float(q['lambada'].mean()),wiki_ppl=float(np.exp(q['wiki'].mean())),pg19_ppl=float(np.exp(q['pg_full'].mean()))))
    rng=np.random.default_rng(p['primary']['bootstrap_seed']);contrasts=[];screens=[]
    for restore in [False,True]:
        checks={}
        for metric in ['word','lambada','wiki','pg_full','pg_tail']:
            delta=np.stack([by[(s,restore)][metric]-sparse[s][metric] for s in [2026091660,2026091661]]).mean(axis=0);idx=rng.integers(0,len(delta),(p['primary']['draws'],len(delta)));ci=np.quantile(delta[idx].mean(axis=1),p['primary']['quantiles']);x=dict(dense64_restore=restore,metric=metric,difference=float(delta.mean()),conditional97_5ci=ci.tolist())
            if metric in ['word','lambada']:checks[metric]=bool(ci[0]>-.05)
            else:x.update(ppl_ratio=float(np.exp(delta.mean())),ppl_ratio_ci=np.exp(ci).tolist());checks[metric]=bool(np.exp(ci[1])<=1.05)
            contrasts.append(x)
        checks['cost']=all(x['still_cheaper_with_one_second_penalty'] for x in costs if x['restore_qk']==restore);screens.append(dict(restore_qk=restore,checks=checks,all=all(checks.values())))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,control=control,optimizer_updates=0,task_predictions=2580,nll_forwards=244,records=records,cost=costs,contrasts=contrasts,candidate_screens=screens,cheaper_dense_candidate_passes=any(x['all'] for x in screens),scope=p['scope']);target=R/f'results/{name}-audit-v0';target.mkdir(exist_ok=True);(target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 密集64步早停对照：相同质量是否更便宜','','直接评测已保存同4090密集64步检查点，分别原权重和已知QK恢复；对照已有稀疏128步恢复模型，无新增训练。全部题已暴露，属于开发成本曲线控制。','','|种子|密集64恢复QK|长题正确率|LAMBADA正确率|Wiki困惑度|PG19困惑度|','|---|---|---:|---:|---:|---:|']
    for x in records:lines.append(f"|{x['seed']}|{x['restore_qk']}|{x['word_accuracy']*100:.2f}%|{x['lambada_accuracy']*100:.2f}%|{x['wiki_ppl']:.4f}|{x['pg19_ppl']:.4f}|")
    lines+=['','预定两候选联合筛查：'+json.dumps(screens,ensure_ascii=False),'','成本：'+json.dumps(costs,ensure_ascii=False),'','每个候选同时满足两个准确率最多低5pp、三个PPL最多高5%、两种子更便宜。两候选使用97.5%区间，条件于已有种子和各数据单位；非独立总体证据。64步时长用训练日志UTC到检查点重载的保守时间估计，包含不必执行的重载；128步用单调时钟实测，加恢复保存；报告1秒额外开销敏感性。不同计时来源不能伪称精确同定义。候选都没通过不代表稀疏已达到最优成本。已知恢复基线非原创，固定128步省时事实与同质量最便宜的策略必须区分。']
    (R/'docs/dense-early64-results-2026-09-17.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',screens=screens,cost=costs)))
if __name__=='__main__':main()
