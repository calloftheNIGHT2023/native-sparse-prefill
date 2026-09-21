"""Post-hoc paired analysis of already-audited endpoints; no model calls."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, tarfile
import numpy as np

R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'results/32k-recovery-trajectory-v0';out.mkdir(exist_ok=False)
    sources={};models={};summaries={}
    meta=load(R/'data/32k-expanded-training-v0/tasks.json')
    variants=['short','no_context','long32768'];seeds=[2026091660,2026091661]
    for label in ['expanded76','gentle32k']:
        audit_path=R/f'results/{label}-audit-v0/result.json';audit=load(audit_path)
        assert audit['status']=='verified';sources[audit_path.relative_to(R).as_posix()]=sha(audit_path)
        archive=R/f'exports/{label}-evidence-v0.tar.gz';assert sha(archive)==audit['archive']['sha256']
        with tarfile.open(archive) as t:
            manifest={x['path']:x for x in json.loads(t.extractfile(f'{label}-manifest.json').read())['files']}
        mirror=R/f'results/cloud-{label}-evidence-v0'
        for j in audit['control']['jobs']:
            if j['phase']!='report':continue
            d=mirror/f'results/{label}-stage-v0'/j['name']
            def verified(name):
                f=d/name;n=f.relative_to(mirror).as_posix();assert sha(f)==manifest[n]['sha256'];sources[f.relative_to(R).as_posix()]=sha(f);return load(f)
            result=verified('result.json');pred=verified('task-results.json')['predictions']
            assert len(pred)==len(meta)==192
            for x,y in zip(pred,meta):
                assert all(x[n]==y[n] for n in ['item_id','variant','gold'])
                assert x['prediction']==int(np.argmax(x['choice_logits'])) and x['correct']==(x['prediction']==x['gold'])
            key=(j['k'],j['seed'],result['step']);assert key not in models
            models[key]={v:np.array([float(x['correct']) for x in pred if x['variant']==v]) for v in variants}
        for x in audit['summary']:summaries[(x['k'],x['step'])]=x
    ids={v:[x['item_id'] for x in meta if x['variant']==v] for v in variants}
    assert ids['short']==ids['no_context']==ids['long32768'] and len(set(ids['short']))==64
    articles=[x['article_hash'] for x in meta if x['variant']=='short'];assert len(set(articles))==64
    positions=np.array([x['evidence_fraction'] for x in meta if x['variant']=='long32768'])
    rng=np.random.default_rng(2026091672);indices=rng.integers(0,64,(20000,64))
    def interval(diff):return dict(delta_pp=float(100*diff.mean()),paired_article_ci95_pp=(100*np.quantile(diff[indices].mean(1),[.025,.975])).tolist())
    changes=[];gaps=[];position_rows=[]
    for k in [0,32,48]:
        for variant in variants:
            before=models[(k,seeds[0],0)][variant]
            after=np.array([models[(k,s,128)][variant] for s in seeds]);diff=after.mean(0)-before
            recovered=float(((after==1)&(before==0)).sum(1).mean());lost=float(((after==0)&(before==1)).sum(1).mean())
            assert abs((recovered-lost)/64-diff.mean())<1e-12
            changes.append(dict(k=k,variant=variant,before=float(before.mean()),after=float(after.mean()),recovered_articles_mean=recovered,lost_articles_mean=lost,**interval(diff)))
        for pos in sorted(set(positions)):
            keep=positions==pos;assert keep.sum()==16
            before=models[(k,seeds[0],0)]['long32768'][keep]
            after=np.array([models[(k,s,128)]['long32768'][keep] for s in seeds])
            position_rows.append(dict(k=k,evidence_fraction=float(pos),articles=16,before=float(before.mean()),after=float(after.mean()),change_pp=float(100*(after.mean()-before.mean()))))
    for k in [32,48]:
        for variant in variants:
            before=models[(k,seeds[0],0)][variant]-models[(0,seeds[0],0)][variant]
            after=np.mean([models[(k,s,128)][variant]-models[(0,s,128)][variant] for s in seeds],axis=0)
            gaps.append(dict(k=k,variant=variant,before_gap_pp=float(100*before.mean()),after_gap_pp=float(100*after.mean()),gap_change=interval(after-before)))
    # The intermediate calibration curve is useful for scheduling, not held-out task evidence.
    calibration=[]
    for k in [0,32,48]:
        label='gentle32k' if k==48 else 'expanded76'
        for seed in seeds:
            name=f'seed{seed}' if k==48 else f'k{k}-seed{seed}'
            f=R/f'results/cloud-{label}-evidence-v0/results/{label}-stage-v0/{name}/result.json'
            v=load(f);sources[f.relative_to(R).as_posix()]=sha(f)
            for e in v['evaluations']:calibration.append(dict(k=k,seed=seed,step=e['step'],mean_nll=e['mean_nll']))
    result=dict(status='verified_posthoc_analysis',utc=datetime.now(timezone.utc).isoformat(),new_optimizer_updates=0,new_model_forwards=0,changes=changes,gaps=gaps,positions=position_rows,calibration=calibration,sources_sha256=sources,scope='Reanalysis of reused64 development articles and two reused seeds; one deterministic zero-adapter baseline per attention mode. All intervals conditional on these endpoints, no new independent confirmation or isolated causal mechanism.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    lines=['# 最新32K结果重新对齐：训练修复了多少差距？','',
           '本轮只分析已有预测、校准日志和训练记录，0新增训练、0模型前向。先前“补目标块、选头、分阶段改密集”的诊断已经做过，不能重复把漏块假说当成新发现或已证实根因。','',
           '## 同一题的训练前后变化','','|配置|零步长题|128步长题|净变化pp|平均修复题数|平均新增错题数|','|---|---:|---:|---:|---:|---:|']
    for x in changes:
        if x['variant']=='long32768':lines.append(f"|K{x['k']}|{x['before']:.2%}|{x['after']:.2%}|{x['delta_pp']:+.2f}|{x['recovered_articles_mean']:.1f}|{x['lost_articles_mean']:.1f}|")
    lines+=['','K0指密集。题数先分别计算两颗种子的变化再平均；零步LoRA B矩阵为零，基线只有一份，不能把复用的基线计成两次独立测量。','',
            '|稀疏相对密集|零步差距pp|128步差距pp|差距改善pp|配对95%区间pp|','|---|---:|---:|---:|---|']
    for x in gaps:
        if x['variant']=='long32768':
            g=x['gap_change'];lines.append(f"|K{x['k']}|{x['before_gap_pp']:+.2f}|{x['after_gap_pp']:+.2f}|{g['delta_pp']:+.2f}|{g['paired_article_ci95_pp']}|")
    lines+=['','差距缩小同时包含稀疏模型进步与密集模型退步，不等于训练恢复了全部长距离能力。四个校准窗口上的NLL不能替代长题终点。','',
            '## 目标原文位置分组','','每组16篇不同文章；文章内容和位置混在一起，因此这些分组只能定位后续检查，不能归因于位置。','','|配置|目标位置比例|零步|128步|变化pp|','|---|---:|---:|---:|---:|']
    for x in position_rows:lines.append(f"|K{x['k']}|{x['evidence_fraction']:.2f}|{x['before']:.2%}|{x['after']:.2%}|{x['change_pp']:+.2f}|")
    lines+=['','## 对下一步的约束','',
            '1. 训练确实改善了稀疏模型，不能把“最终仍落后”说成“训练毫无作用”；剩余损失也不能用理论信念抹去。',
            '2. “读不到原文”还没有被隔离证明。此前24题强制路由并不代表全部主要退步；但全96题切换密集路径仍未稳定修复，历史负证据必须一起考虑。',
            '3. 优先用现有中间断点测完整开发集曲线，不新增训练；固定展示全部选定断点，不挑最好一次宣称泛化。这样检查剩余差距是在持续修复、已停滞，还是后期又恶化。',
            '4. 若要判断路由机制，后续需要同一篇文章在不同位置的配对干预，且先验证密集模型确实利用原文；当前不同文章的位置分组不满足因果对照。',
            '5. 本轮不证明新方法、非劣性或论文结论。卡仍按上次停机记录处于已Stop；没有在本轮恢复云端训练。']
    (R/'docs/32k-recovery-trajectory-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],long_changes=[x for x in changes if x['variant']=='long32768'],long_gaps=[x for x in gaps if x['variant']=='long32768'])))

if __name__=='__main__':main()
