"""Audit completed length128 evidence and write a report without model updates."""
import hashlib,json,math,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/zoology-length128-cpu-v0'
QUERY=ROOT/'results/zoology-length128-query-v0'
VALUES=ROOT/'results/zoology-length128-source-values-v0'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    r=read(RUN/'result.json');q=read(QUERY/'result.json');value_probe=read(VALUES/'result.json')
    assert r['status']=='complete' and r['sequence_length']==128
    out=ROOT/'results/zoology-length128-analysis-v0';out.mkdir(exist_ok=False)
    verified={}
    for folder in [RUN,QUERY,VALUES]:
        manifest=read(folder/'manifest.json')
        for item in manifest:assert sha(folder/item['path'])==item['sha256'],item['path']
        verified[folder.name]=len(manifest)
    assert q['checkpoint_sha256']==sha(RUN/'checkpoint.pt') and q['optimizer_updates']==0
    assert value_probe['checkpoint_sha256']==q['checkpoint_sha256'] and value_probe['optimizer_updates']==0
    cfg=read(RUN/'config.json');assert cfg==read(ROOT/'configs/zoology-length128-v0.json')
    oldcfg=read(ROOT/'results/zoology-basic-cpu-v1/resolved-config.json')
    adjusted=json.loads(json.dumps(cfg));adjusted['model']['max_position_embeddings']=64
    for split in ['train_configs','test_configs']:
        for part in adjusted['data'][split]:part['input_seq_len']=64
    assert adjusted==oldcfg
    e=[json.loads(line) for line in (RUN/'events.jsonl').read_text().splitlines()]
    steps=[x for x in e if x['event']=='optimizer_step'];v=[x for x in e if x['event']=='validation']
    assert [x['step'] for x in steps]==list(range(1,r['updates']+1))
    assert len(v)==r['epochs'] and r['updates']==313*r['epochs']
    assert all(a['utc']<=b['utc'] for a,b in zip(e,e[1:]))
    assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in steps)
    assert all(x['step']==313*(x['epoch']+1) for x in v)
    assert not any(x['valid/accuracy']>.99 for x in v[:-1])
    assert r['epochs']==100 or v[-1]['valid/accuracy']>.99
    assert r['input_tokens']==1280000*r['epochs'] and r['supervised_answers']==40000*r['epochs']
    assert r['examples_seen']==10000*r['epochs']
    epoch_metrics=[dict(epoch=x['epoch']+1,train_batch_mean_loss=float(np.mean([s['loss'] for s in steps if s['epoch']==x['epoch']])),
        validation_batch_mean_loss=x['valid/loss'],validation_accuracy=x['valid/accuracy']) for x in v]
    save(out/'epoch-metrics.json',epoch_metrics)
    for x in steps:
        assert math.isclose(x['learning_rate'],.001*(1+math.cos(math.pi*x['epoch']/100))/2,rel_tol=1e-12,abs_tol=1e-14)
    hashes={};fresh=None
    for name in ['train','validation','fresh-evaluation']:
        data=torch.load(RUN/(name+'-data.pt'),weights_only=True)
        assert data['inputs'].shape[1]==128 and data['labels'].shape==data['inputs'].shape
        rows=[sha_tensor(x) for x in data['inputs']];assert len(rows)==len(set(rows));hashes[name]=set(rows)
        for x,y in zip(data['inputs'],data['labels']):
            mapping={int(x[i]):int(x[i+1]) for i in range(0,8,2)}
            assert len(mapping)==4 and len(set(mapping.values()))==4 and int((y!=-100).sum())==4
            for pos in torch.where(y!=-100)[0]:assert pos>=8 and mapping[int(x[pos])]==int(y[pos])
        if name=='fresh-evaluation':fresh=data
    assert not hashes['train']&hashes['validation'] and not hashes['fresh-evaluation']&(hashes['train']|hashes['validation'])
    preds=np.asarray(r['fresh_evaluation']['predictions']);labels=fresh['labels'][fresh['labels']!=-100].numpy()
    assert len(preds)==4000 and np.mean(preds==labels)==r['fresh_evaluation']['accuracy']
    scores=(preds==labels).reshape(1000,4).mean(1)
    assert np.array_equal(scores,np.asarray(r['fresh_evaluation']['sequence_scores']))
    rng=np.random.default_rng(2026091426)
    samples=scores[rng.integers(0,1000,(4000,1000))].mean(1)
    ci=np.quantile(samples,[.025,.975]).tolist()
    bins={label:dict(count=0,correct=0) for label in ['1-8','9-16','17-32','33-64','65-127']}
    index=0
    for x,y in zip(fresh['inputs'],fresh['labels']):
        source={int(x[i]):i+1 for i in range(0,8,2)}
        for pos in torch.where(y!=-100)[0]:
            gap=int(pos)-source[int(x[pos])]
            group=next(label for label,upper in zip(bins,[8,16,32,64,127]) if gap<=upper)
            bins[group]['count']+=1;bins[group]['correct']+=int(preds[index]==int(y[pos]));index+=1
    for b in bins.values():b['accuracy']=b['correct']/b['count'] if b['count'] else None
    assert sum(b['count'] for b in bins.values())==4000
    assert sum(b['correct'] for b in bins.values())==int(np.sum(preds==labels))
    # Recompute all query intervention summary values from its saved predictions.
    qp=read(QUERY/'predictions.json')
    before=np.asarray(qp['original_predictions']);after=np.asarray(qp['changed_predictions'])
    old=np.asarray(qp['old_labels']);new=np.asarray(qp['new_labels'])
    assert len(old)==1000 and np.all(old!=new)
    for key,value in dict(original_last_query_accuracy=np.mean(before==old),changed_query_accuracy=np.mean(after==new),
        prediction_changed_fraction=np.mean(before!=after),both_correct_fraction=np.mean((before==old)&(after==new))).items():assert value==q[key]
    vp=read(VALUES/'predictions.json');vp_after=np.asarray(vp['changed_predictions']);vp_new=np.asarray(vp['new_labels']);vp_old=np.asarray(vp['old_labels'])
    assert np.array_equal(before,np.asarray(vp['original_predictions'])) and np.array_equal(old,vp_old)
    assert np.all(vp_new!=vp_old) and np.mean(vp_after==vp_new)==value_probe['swapped_values_accuracy']
    assert np.mean(vp_after!=before)==value_probe['prediction_changed_fraction']
    far=np.asarray(vp['gaps'])>64;assert int(far.sum())==value_probe['far_examples']
    if far.any():assert np.mean(vp_after[far]==vp_new[far])==value_probe['far_swapped_values_accuracy']
    now=datetime.now(timezone.utc).isoformat()
    wall=(datetime.fromisoformat(r['finished_utc'])-datetime.fromisoformat(r['started_utc'])).total_seconds()
    passed=bool(r['upstream_stop_threshold_reached'] and r['independent_fresh_at_least_99'])
    audit=dict(utc=now,run=RUN.name,manifest_files_verified=verified,config_only_length_and_required_position_changes=True,
        optimizer_steps=r['updates'],epochs=r['epochs'],step_sequence_and_times_checked=True,cosine_schedule_checked=True,
        first_stopping_threshold_checked=True,training_validation_fresh_rows_disjoint=True,all_labels_checked=True,
        fresh_accuracy=r['fresh_evaluation']['accuracy'],fresh_sequence_bootstrap_95=ci,distance_bins=bins,
        query_metrics_recomputed=True,checkpoint_sha256=q['checkpoint_sha256'],passed=passed,wall_seconds=wall,
        source_run_and_query_unchanged=True,additional_training_updates=0,supplementary_source_values=value_probe)
    save(out/'audit.json',audit)
    oldrun=read(ROOT/'results/zoology-basic-cpu-v1/result.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for result,length in [(oldrun,64),(r,128)]:
        c=result['curves'];axes[0].plot([x['epoch']+1 for x in c],[100*x['valid/accuracy'] for x in c],label=f'{length} tokens')
    axes[0].axhline(99,color='grey',ls='--',lw=1)
    axes[0].set(xlabel='Epoch',ylabel='Validation accuracy (%)',ylim=(0,102));axes[0].legend();axes[0].grid(alpha=.2)
    labels=[k for k,b in bins.items() if b['count']]
    bars=axes[1].bar(labels,[100*bins[k]['accuracy'] for k in labels],color='#357aab')
    for bar,label in zip(bars,labels):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"n={bins[label]['count']}",ha='center',fontsize=8)
    axes[1].set(xlabel='Query minus source-value position',ylabel='Fresh accuracy (%)',ylim=(0,110));axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning-and-distance.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([x['epoch'] for x in epoch_metrics],[x['train_batch_mean_loss'] for x in epoch_metrics],label='Training batch mean')
    ax.plot([x['epoch'] for x in epoch_metrics],[x['validation_batch_mean_loss'] for x in epoch_metrics],label='Validation batch mean')
    ax.set(xlabel='Epoch',ylabel='Cross-entropy loss');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=170);plt.close(fig)
    headline='长度128检查通过，可以继续加入远距离条件' if passed else '长度128未通过预定门槛，先定位问题'
    report=f"""# Zoology长度128：{headline}

本轮已在本机CPU完成{r['epochs']}轮、{r['updates']:,}次更新。最终验证正确率{v[-1]['valid/accuracy']:.3%}，独立1000条新序列/4000个答案正确率{r['fresh_evaluation']['accuracy']:.3%}。训练与评测墙钟{wall:.2f}秒，纯训练日志累计{r['training_seconds']:.2f}秒。本轮新增GPU作业0。

## 相比64长度改了什么

训练/验证/新题长度64改128，绝对位置嵌入表相应扩到128，模型从437760增至445952参数并从头训练。配置逐字段核验：除此之外与已通过的作者basic配置一致，仍为256词表、4组键值、128宽/2层/1头、10000训练/1000验证、batch32、seed123、AdamW学习率1e-3/weight decay0.1、每轮余弦调度、最多100轮。

原验证集参与每轮停止判断；规则仍是验证准确率严格超过99%。参数表大小和随机数消耗变化，不能声称初始化完全配对。长度128每轮处理的token数翻倍，曲线按epoch展示，不代表等计算量比较，也不是原64长度模型直接外推到128。

训练损失与验证损失同时保存于epoch-metrics.json和loss.png。最终一轮训练批均损失{epoch_metrics[-1]['train_batch_mean_loss']:.4f}、验证批均损失{epoch_metrics[-1]['validation_batch_mean_loss']:.4f}；该差距是诊断，不证明某种唯一原因，也不用于事后改变停止规则。

## 结果与检查

独立新题使用预先固定种子2026091425，在训练结束后评估，未用于选择停止轮数。按序列bootstrap的95%区间为{ci[0]:.3%}—{ci[1]:.3%}，不能代表跨训练种子波动。训练、验证、新题整行输入无交叉，所有监督标签均从前方记录核验。与64长度评测采用相同种子，可能共享键值模板，不能将两长度看成独立重复。

| 条件 | 64长度 | 128长度 |
|---|---:|---:|
| 训练轮数 | {oldrun['epochs']} | {r['epochs']} |
| 验证正确率 | {oldrun['curves'][-1]['valid/accuracy']:.3%} | {v[-1]['valid/accuracy']:.3%} |
| 独立新题正确率 | {oldrun['fresh_evaluation']['accuracy']:.3%} | {r['fresh_evaluation']['accuracy']:.3%} |
| 输入tokens（包含重复样本） | {oldrun['input_tokens']:,} | {r['input_tokens']:,} |

只改最后一个查询键、保持前方记录及其他输入不变，1000对题中原最后一题正确率{q['original_last_query_accuracy']:.2%}，改后{q['changed_query_accuracy']:.2%}，预测改变比例{q['prediction_changed_fraction']:.2%}，两题均正确{q['both_correct_fraction']:.2%}。改键可能产生重复查询键，属于探索性干预；0优化更新、检查点SHA不变。

按保存的新题预测统计距离，距离为查询位置减去源值位置；以下分组未用于挑选模型：

| 距离 | 答案数 | 答对数 | 正确率 |
|---|---:|---:|---:|
"""
    for label,b in bins.items():
        acc=f"{b['accuracy']:.2%}" if b['accuracy'] is not None else '无样本'
        report+=f"| {label} | {b['count']} | {b['correct']} | {acc} |\n"
    report+=f"\n补充源值对调检查（训练期间登记、最终新题结果产生前固定）：保持问题不变，只交换前方两条记录的值，改后正确率{value_probe['swapped_values_accuracy']:.2%}，预测改变{value_probe['prediction_changed_fraction']:.2%}。其中距离超过64的{value_probe['far_examples']}题，改后正确率{value_probe['far_swapped_values_accuracy']:.2%}。这直接检查答案能否跟随前方记录变化；复用同一批新题和模型，不是独立确认。记录见results/zoology-length128-source-values-v0/，优化更新0。\n"
    report+=f"""
数据仍采用作者power_a=0.01的位置分布，加长序列并不意味着所有题都变成远距离查询。因此整体高分仍需结合距离分组解读。

## 日志、核验与时间轴

- 训练开始UTC：{r['started_utc']}；结束UTC：{r['finished_utc']}。
- 查询干预UTC：{q['started_utc']} 至 {q['finished_utc']}。
- 原始训练：results/zoology-length128-cpu-v0/；逐步events.jsonl与logs/zoology-length128-cpu-v0.log，保留配置、源码、模型/优化器/随机状态和数据。
- 查询干预：results/zoology-length128-query-v0/；分析：results/zoology-length128-analysis-v0/。
- 输入tokens {r['input_tokens']:,}，受监督答案{r['supervised_answers']:,}，固定10000序列重复训练，不是同量独立数据。
- 38项CPU检查通过；基线清单{verified[RUN.name]}个文件、干预清单{verified[QUERY.name]}个文件SHA核验通过；更新顺序、UTC、有限损失/梯度、余弦学习率与首次停止阈值核验通过。
- 本轮只读分析额外优化更新0；旧64长度结果保持。此前失败记录见TIMELINE.md和对应运行目录。

## 这支持什么，下一步是什么

"""
    if passed:
        report+='已知稠密模型在128长度上通过同一规则的可学性检查。下一步可固定128长度，把查询集中到后半段，明确检验是否必须利用前方记录，再设计同预算的稠密/局部/候选稀疏比较。该后续实验本轮未启动。\n'
    else:
        report+='原定验证加独立新题双门槛未全部达到；不将这次结果称为成功复现或稀疏方法证据。下一步先查看距离分组和学习曲线，区分训练未收敛与远端错误，再制定一次有针对性的检查；不自动追加租卡或扩大。\n'
    report+='\n当前只有一个训练种子、一个合成查找任务，没有完成2K长上下文、语言建模、稀疏训练方法比较或理论证明。关联回忆任务与作者模型已有前作，本次成功与否均不构成原创贡献。\n\n来源：[固定版本basic配置](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/basic_examples/basic.py)；本地源文件与平台补丁见third_party/zoology-1ad20d1/。\n'
    (ROOT/'docs/zoology-length128-results-2026-09-14.md').write_text(report,encoding='utf-8')
    shutil.copy2(__file__,out/Path(__file__).name)
    save(out/'manifest.json',[dict(path=p.name,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()])
    print(json.dumps(dict(passed=passed,epochs=r['epochs'],validation=v[-1]['valid/accuracy'],fresh=r['fresh_evaluation']['accuracy'],ci=ci,query=q['changed_query_accuracy'],distance=bins)),flush=True)

def sha_tensor(tensor):return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

if __name__=='__main__':main()
