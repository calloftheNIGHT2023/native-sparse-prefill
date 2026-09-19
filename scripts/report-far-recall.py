import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o): p.write_text(json.dumps(o,indent=2)+'\n',encoding='utf-8')
def main():
    data=ROOT/'data/far-recall-v1'
    for version in ['far-recall-v0','far-recall-v1']:
        path=ROOT/'data'/version
        for r in json.loads((path/'manifest.json').read_text()): assert sha(path/r['path'])==r['sha256']
    ds=json.loads((data/'audit.json').read_text()); runs=[]; audits=[]
    for name in ['far-recall-cpu-learnability-v0','far-recall-cpu-stream-v2']:
        folder=ROOT/'results'/name; result=json.loads((folder/'result.json').read_text()); assert result['status']=='complete'
        manifest=json.loads((folder/'manifest.json').read_text())
        for rec in manifest: assert sha(folder/rec['path'])==rec['sha256'],rec['path']
        events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
        steps=[x for x in events if x['event']=='optimizer_step']
        assert [x['step'] for x in steps]==list(range(1,result['updates']+1))
        assert all(np.isfinite(x['loss']) and np.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in steps)
        assert all(a['utc']<=b['utc'] for a,b in zip(events,events[1:]))
        cfg=json.loads((folder/'frozen-config.json').read_text())
        if cfg.get('training_mode')=='fresh_families':
            stream=[h for x in steps for h in x['fresh_family_hashes']]
            assert len(stream)==len(set(stream))==cfg['updates']*cfg['batch_size']
            holdout={m['family_sha256'] for split in json.loads((folder/'data-families.json').read_text()).values() for m in split}
            assert not holdout.intersection(stream)
        result['name']=name; result['train_last_100_mean_loss']=float(np.mean([x['loss'] for x in steps[-100:]])); runs.append(result)
        audits.append(dict(name=name,files_verified=len(manifest),updates=len(steps),utc_ordered=True,
            input_tokens=result['input_tokens'],supervised_answers=result['supervised_answer_tokens']))
    probe=ROOT/'results/far-recall-70m-sensitivity-v1'
    for r in json.loads((probe/'manifest.json').read_text()): assert sha(probe/r['path'])==r['sha256']
    pr=json.loads((probe/'result.json').read_text()); assert pr['all_checks_passed']
    failed=ROOT/'results/far-recall-cpu-stream-v1'
    fe=[json.loads(x) for x in (failed/'events.jsonl').read_text().splitlines()]
    fs=[x for x in fe if x['event']=='optimizer_step']
    assert fe[-1]['event']=='failed' and len(fs)==3142
    assert [x['step'] for x in fs]==list(range(1,3143))
    collision=json.loads((ROOT/'logs/far-recall-stream-v1-collision-audit.json').read_text())
    failure=dict(name=failed.name,status='failed',started_utc=fe[0]['utc'],finished_utc=fe[-1]['utc'],
        updates=len(fs),input_tokens=fs[-1]['input_tokens'],supervised_answers=fs[-1]['examples'],
        training_seconds=sum(x['step_seconds'] for x in fs),collision=collision,final_test_available=False)
    # Post-run preservation manifest; not an original pre-run source attestation.
    failure_manifest=[dict(path=str(p.relative_to(failed)),sha256=sha(p)) for p in sorted(failed.rglob('*')) if p.is_file()]
    failure['preserved_files']=failure_manifest
    qp=ROOT/'results/far-recall-query-use-v0'
    for r in json.loads((qp/'manifest.json').read_text()): assert sha(qp/r['path'])==r['sha256']
    query=json.loads((qp/'result.json').read_text())
    folder=ROOT/'results/far-recall-figures'; folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for r,label in zip(runs,['Fixed families / answer-marker','Fresh PCG64 / key-last / new split']):
        axes[0].plot([x['step'] for x in r['development']],[x['accuracy']*100 for x in r['development']],marker='o',label=label)
    axes[0].axhline(80,color='grey',ls='--',lw=1,label='Fixed learnability gate')
    axes[0].set(xlabel='Optimizer updates',ylabel='Development accuracy (%)',ylim=(0,105),title='CPU 128-token learnability controls')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    latest=runs[-1]; labels=['Full / far','Local / far','Full / evidence removed','Local / moved near']
    values=[latest[k]['accuracy']*100 for k in ['dense_test','local_far_test','removed_evidence_dense_test','local_near_test']]
    axes[1].bar(range(4),values,color=['#3576c4','#b0b7c3','#b0b7c3','#de9638'])
    axes[1].set_xticks(range(4),labels,rotation=18,ha='right'); axes[1].set(ylabel='Test accuracy (%)',ylim=(0,110),title='Same final weights; 64 held-out families')
    for i,v in enumerate(values): axes[1].text(i,v+2,f'{v:.1f}%',ha='center')
    fig.tight_layout(); fig.savefig(folder/'learnability.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,2.8)); n=2048
    direct=np.zeros(n); direct[:4]=1; direct[1796:]=1
    all6=np.zeros(n); all6[:4]=1; all6[536:]=1
    evidence=np.zeros(n); evidence[64:448]=1
    ax.imshow(np.stack([direct,all6,evidence]),aspect='auto',interpolation='nearest',cmap='Blues',vmin=0,vmax=1,extent=[0,2048,2.5,-.5])
    ax.set_yticks([0,1,2],['One-layer accessible','Six-layer accessible','Evidence placement'])
    ax.set_xticks([0,64,448,536,1024,1796,2048]); ax.set_xlabel('Input token position (zero based)')
    ax.set_title('Static sink + recent graph: evidence has no path to final output')
    fig.tight_layout(); fig.savefig(folder/'reachability.png',dpi=180); plt.close(fig)
    totals=dict(completed_runs=2,failed_runs=1,optimizer_updates=sum(x['updates'] for x in audits)+failure['updates'],
        input_tokens=sum(x['input_tokens'] for x in audits)+failure['input_tokens'],
        supervised_answers=sum(x['supervised_answers'] for x in audits)+failure['supervised_answers'])
    now=datetime.now(timezone.utc).isoformat(); save(ROOT/'logs/far-recall-final-audit.json',dict(utc=now,data=ds,runs=audits,failed_run=failure,probe=pr,query_probe=query,totals=totals))
    save(ROOT/'results/far-recall-summary.json',dict(utc=now,data=ds,runs=runs,failed_run=failure,probe=pr,query_probe=query,totals=totals))
    lines=['# 远距离信息测试：本地准备与初步结果','',
        '已完成2K数据、跨六层信息路径审计、既有70M模型的CPU输入干预和小模型可学性实验。本轮没有启动GPU作业。测试工具建立在已有的关联回忆任务上；还不是新稀疏方法的结果。','',
        '## 用人话说','',
        '在前面放“编号→数值”的记录，最后问某个编号的数值。每个反事实家族只改变远处答案，附近内容完全相同。这能检查模型是否真的拿到了那条远处信息。','',
        '## 实际成绩','',
        '| CPU训练设置 | 固定最终开发正确率 | 最终测试正确率 | 局部对照测试 | 80%可学性门槛 |',
        '|---|---:|---:|---:|---|']
    for r,label in zip(runs,['重复固定题库 v0','PCG64新题族、末尾键查询 v2']):
        lines.append(f"| {label} | {r['development'][-1]['accuracy']:.2%} | {r['dense_test']['accuracy']:.2%} | {r['local_far_test']['accuracy']:.2%} | {'通过' if r['learnability_gate_passed'] else '未通过'} |")
    lines.extend(['','这两轮是330,752参数、2层、128输入长度的CPU诊断，随机种子41；后一次改变题族生成器、查询格式、数据种子和留出题，是流程迭代，不是单因素对照或独立种子复现。完整模型只对最后一个答案token接受监督，不能把输入token总量当作全部语言建模训练量。',
        '',f"最新模型去掉远处证据后，测试正确率为{latest['removed_evidence_dense_test']['accuracy']:.2%}；同一权重将证据移近并切局部注意力后为{latest['local_near_test']['accuracy']:.2%}。后者同时改变证据位置和掩码，属于分布外干预，不能单独解释成局部架构上限。",'',
        '固定题库的训练损失很低而新题正确率不足，表明原先训练设置未获得足够的泛化。该失败运行、所有中间曲线和最终权重完整保留。流式版本逐样本生成全新家族，64,000个训练家族哈希无重复且与留出家族不交叉；数据生成种子、样本编号、答案变体和哈希均有记录。','',
        '两次完整运行都未达到预设的最终开发/测试均80%门槛。不能将38%的正确率当成已学会关联查找，也不能据此判定索引器优劣。', '',
        '## 为什么继续增加训练量前要停下来检查','',
        f"对最新权重只做CPU前向，在32个开发家族的256个基础输入上遍历4个查询编号，共1024次预测。记录保持不变，只改查询键。在{query['different_answer_query_pairs']}对正确答案不同的查询之间，模型仅{query['prediction_changed_fraction']:.2%}改变预测，两个答案均正确仅{query['both_answers_correct_fraction']:.2%}。这些成组样本不是独立试验。",'',
        f"原开发准确率{query['original_dev_accuracy']:.2%}；忽略查询编号、直接选择记录中最常见数值的规则为{query['original_dev_query_blind_baselines']['most_frequent_value_ties_lowest']:.2%}，总选最末记录数值为{query['original_dev_query_blind_baselines']['last_record_value']:.2%}。它们是看到了远处数值但不会按键查找的参照。当前证据支持优先排查键值绑定，而非据此扩大稀疏训练；不是证明所有模型都学不会。未使用测试集挑检查点。",'',
        '## 保留的生成失败','',
        '流式v1在第3143步生成样本时触发重复家族断言，只完成3142次更新，没有最终测试成绩。两个不同64位种子具有相同低32位，在当前Torch CPU生成器下产生相同题族。改用完整SHA种子驱动的PCG64并加入回归检查；v2全部64,000个训练题族未重复。v1原始日志、失败堆栈、3142步checkpoint和源码快照保留；不把它列为完成运行。','',
        '## 2K数据和70M模型检查','',
        '2K主数据含训练4096条、开发512条、测试1024条；分别来自256/32/64个家族，每族16个答案变体。家族是统计单元。这是离散符号查找任务，不是自然语言指令。',
        '', '当前静态开头+附近对照在六层后只依赖0—3和536—2047位置；关键记录在64—447。因此16个反事实版本在可达位置完全一致，确定性、答案类别内预测的正确率只能为6.25%。完整输入符号查表器全部答对；程序查表分数不能当作神经网络成绩。',
        '', '使用已有10M云训练所得70M权重，仅在CPU前向，最后输出对以下变动的最大logit差：','',
        '| 干预 | 最大输出差 |','|---|---:|'])
    for r in pr['conditions']: lines.append(f"| {r['condition']} | {r['max_logit_change']:.9f} |")
    lines.extend(['','该检查说明无路径时看不到改变、接通路径后会响应；**没有证明70M模型已能答对这道任务**。初次batch=2检查连完全相同输入都有约2.4e-6浮点差，因此严格相等断言失败；保留失败日志后改为同形状batch=1逐条计算，局部远处和删证据两项差均为0，并未通过放宽精度阈值掩盖差异。','',
        '6.25%界仅适用于这个静态局部对照及位置独立的残差/MLP/归一化。内容相关选块、Gated DeltaNet等递归路径不满足该前提；不能外推到Qwen混合架构。','',
        '## 时间轴与证据','',
        '| 运行 | 开始UTC | 结束UTC | 更新数 | 纯训练秒 |','|---|---|---|---:|---:|'])
    for r in sorted([*runs,failure],key=lambda r:r['started_utc']):
        label=r['name']+('（失败）' if r['status']=='failed' else '')
        lines.append(f"| {label} | {r['started_utc']} | {r['finished_utc']} | {r['updates']} | {r['training_seconds']:.2f} |")
    lines.extend(['',f"本轮小模型累计{totals['optimizer_updates']:,}次更新、{totals['input_tokens']:,}输入tokens、{totals['supervised_answers']:,}个受监督答案，含失败运行已执行步骤。不得并入此前70M模型的训练量。",'',
        '- 当前2K数据与生成审计：`data/far-recall-v1/`；旧v0保留；总审计：`logs/far-recall-final-audit.json`。v1改用PCG64和末尾查询键格式。',
        '- 完整训练：`results/far-recall-cpu-learnability-v0/`、`results/far-recall-cpu-stream-v2/`；失败v1另存，均含逐步日志、配置、来源快照、模型和优化器。',
        '- 实际70M前向检查：`results/far-recall-70m-sensitivity-v1/`；v0失败事件和日志保留。',
        '- 查询干预：`results/far-recall-query-use-v0/`，只有开发集CPU前向，无参数更新。',
        '- 图：`results/far-recall-figures/`。29项正确性检查通过；后续流式训练额外执行逐家族去重断言。','',
        '## 前作与下一步','',
        '关联回忆与噪声查找已有 [MAD §3.1/B.1](https://arxiv.org/html/2403.17844v2)、[RULER](https://arxiv.org/html/2404.06654v3) 等前作；[Zoology](https://arxiv.org/abs/2312.04927)本轮仅核对摘要。我们将它们作为诊断依据，不把造这套题宣称为原创贡献。定向原文记录在 `literature/far-recall-check-2026-09-14/`，详细前提见 `docs/far-recall-validation-plan-2026-09-14.md`。','',
        '下一步保留当前任务和固定门槛，先核对已有关联回忆实现的键值绑定结构与监督密度，设计一次有明确区别的本地对照；当前不继续盲目加步数、种子或租卡。只有完整模型在新题上达到门槛，才进入70M/2K复核及后续索引器比较。既有CPU运行器尚非GPU任务运行器。当前尚无新方法收益或Qwen复现结论。旧Pod停止状态本轮未查，未启动新的GPU作业不能等同于旧Pod已停止计费。'])
    (ROOT/'docs/far-recall-results-2026-09-14.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(runs=audits,latest_gate=latest['learnability_gate_passed'])))
if __name__=='__main__': main()
