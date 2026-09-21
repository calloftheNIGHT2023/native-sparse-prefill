"""Verify profiler traces and label their limits separately from timing evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,numpy as np
R=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=R/'exports/word-suffix-profile-evidence-v0.tar.gz';proof=load(a.with_suffix('.json'));assert sha(a)==proof['sha256'] and a.stat().st_size==proof['bytes']
    dest=R/'results/cloud-word-suffix-profile-evidence-v0';dest.mkdir(exist_ok=True)
    with tarfile.open(a) as t:
        entries=json.loads(t.extractfile('word-suffix-profile-manifest.json').read())['files'];names=[m.name for m in t]
        assert len(names)==len(set(names)) and len(entries)==proof['files'] and set(names)=={e['path'] for e in entries}|{'word-suffix-profile-manifest.json'}
        for e in entries:
            m=t.getmember(e['path']);assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes'];f=dest/m.name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
    pp=dest/'provenance/word-suffix-profile-protocol-v0.json';assert sha(pp)==sha(R/'provenance/word-suffix-profile-protocol-v0.json');p=load(pp)
    for n,h in {**p['source_sha256'],**p['data_sha256']}.items():assert sha(dest/n)==h,n
    assert sha(R/'results/word-suffix-cost-audit-v0/result.json')==p['parent_audit_sha256']
    stage=dest/'results/word-suffix-profile-stage-v0';control=load(stage/'result.json');assert control['status']=='complete' and control['task_predictions']==6 and control['optimizer_updates']==0 and control['protocol_sha256']==sha(pp)
    assert len(control['jobs'])==len(p['jobs'])==1;job=p['jobs'][0];d=stage/job['name'];v=load(d/'result.json')
    assert v['status']=='complete' and v['job']==job and v['task_predictions']==6 and v['optimizer_updates']==0 and v['step']==128
    assert v['eval_source_sha256']==sha(dest/'scripts/profile_word_suffix_v0.py')==sha(d/'source.py')
    mirror=R/'results/cloud-expanded76-evidence-v0';assert sha(mirror/job['path'])==job['sha256']==v['checkpoint_sha256'] and sha(mirror/job['training_result'])==job['training_result_sha256']
    train=load(mirror/job['training_result']);assert train['identity']==v['identity'];cal=next(e['values'] for e in train['evaluations'] if e['step']==128 and e['split']=='calibration');assert len(v['calibration_values'])==4 and max(abs(x-y) for x,y in zip(cal,v['calibration_values']))<=1e-6 and v['calibration_max_abs_error']<=1e-6
    refs={x['condition']:load(dest/x['path']) for x in p['references']};preds=v['predictions'];assert preds==[json.loads(x) for x in (d/'task-predictions.jsonl').read_text().splitlines()]
    for x,planned in zip(preds,p['schedule']):
        assert all(x[n]==value for n,value in planned.items());ref=refs[x['mode']];assert ref['environment']==v['environment']
        old=next(y for y in ref['predictions'] if y['variant']=='long32768' and y['item_id']==x['item_id']);assert max(abs(a-b) for a,b in zip(x['choice_logits'],old['choice_logits']))<=1e-6 and x['prediction']==old['prediction']==int(np.argmax(x['choice_logits']))
    assert len(v['profiles'])==3 and {x['mode'] for x in v['profiles']}=={'DD','SS','SD'}
    summary=[]
    for x in v['profiles']:
        f=d/x['trace_file'];assert sha(f)==x['trace_sha256'];events=[e for e in load(f)['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'];total=sum(e.get('dur',0) for e in events)
        assert abs(total-x['cuda_kernel_microseconds_total'])<.01 and abs(sum(z['microseconds'] for z in x['kernels'])-total)<.01
        summary.append(dict(mode=x['mode'],kernel_ms=total/1000,groups_ms={k:v/1000 for k,v in x['grouped_cuda_kernel_microseconds'].items()},top_kernels=x['kernels'][:12]))
    result=dict(status='verified',utc=datetime.now(timezone.utc).isoformat(),archive=proof,optimizer_updates=0,task_predictions=6,summary=summary,control=control,scope=p['scope']);out=R/'results/word-suffix-profile-audit-v0';out.mkdir(exist_ok=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 4090稀疏prefill性能剖析','','一个固定K32训练权重、一个固定32K输入，三种模式各预热一次、剖析一次，6次输出通过1e-6重放。此表是CUDA kernel时间相加，并非端到端wall time；带剖析开销的执行不能替代此前交错计时。','','|模式|kernel总毫秒|池化|路由/排序|注意力|矩阵|其他|','|---|---:|---:|---:|---:|---:|---:|']
    for x in summary:
        g=x['groups_ms'];lines.append('|'+x['mode']+'|'+f"{x['kernel_ms']:.3f}|"+'|'.join(f"{g.get(k,0):.3f}" for k in ['pooling','routing_sorting','attention','matrix','other'])+'|')
    lines+=['','分类来自kernel名称的启发式规则；原始trace和完整kernel列表已归档，可进一步核对。当前结论限制在0.5B模型、32K、单4090；原32K训练峰值39–40GiB，超过24GiB当前卡，后续训练计时需要两组一致启用梯度检查点并单独报告。']
    (R/'docs/word-suffix-profile-results-2026-09-16.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='verified',summary=[{k:v for k,v in x.items() if k!='top_kernels'} for x in summary])))
if __name__=='__main__':main()
