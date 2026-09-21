"""Verify the first full-backbone pilot and write a bounded, evidence-led decision."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def write(p,x): (ROOT/p).write_text(x,encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()

def main():
    source=ROOT/'results/joint-pilot-v0'; s=read('results/joint-pilot-v0/summary.json'); cfg=read('configs/joint-pilot-v0.json')
    started=utc(); manifest=read('results/joint-pilot-v0/manifest.json')
    for rec in manifest: assert sha(source/rec['path'])==rec['sha256'],rec['path']
    data=read('data/joint-pilot-v0/manifest.json'); assert sha(ROOT/'data/joint-pilot-v0/tokens.pt')==data['tokens_sha256']
    old=[read('data/'+n+'/manifest.json') for n in ['realtext-v0-r1','realtext-calibrated-fresh-v0']]
    for field in ['text_sha256','token_prefix_sha256']:
        new={x[field] for x in data['examples']}; previous={x[field] for m in old for x in m['examples']}
        assert len(new)==len(data['examples']); assert not new.intersection(previous)
    grouping=defaultdict(set); idx_grouping=defaultdict(set); runs=[]; warnings=[]
    for r in s['results']:
        name=f"{r['initialization']}__seed{r['seed']}__{r['method']}"
        events=[json.loads(x) for x in (source/name/'events.jsonl').read_text().splitlines()]
        steps=[x for x in events if x['event']=='optimizer_step']
        assert [x['step'] for x in steps]==list(range(1,cfg['updates']+1))
        assert events[-1]['event']=='complete'; assert steps[-1]['tokens']==r['tokens']
        for a,b in zip(events,events[1:]):
            assert b['monotonic_seconds']>=a['monotonic_seconds']
            if datetime.fromisoformat(b['utc'])<datetime.fromisoformat(a['utc']): warnings.append(name)
        assert all(math.isfinite(x[k]) for x in steps for k in ['lm_loss','aux_loss','lm_grad','indexer_grad'])
        assert all(x['lm_grad']>0 for x in steps)
        dense_steps=sum(x['mode']=='dense' for x in steps)
        expected=cfg['updates'] if r['method']=='dense' else cfg['warmup_updates'] if r['method']=='sparse_warmup' else 0
        assert dense_steps==expected
        assert r['initial_hashes']['backbone']!=r['final_hashes']['backbone']
        if r['method']=='dense': assert r['initial_hashes']['indexers']==r['final_hashes']['indexers']
        else:
            assert r['initial_hashes']['indexers']!=r['final_hashes']['indexers']
            assert all(x['indexer_grad']>0 for x in steps)
        grouping[(r['initialization'],r['seed'])].add(r['initial_hashes']['backbone'])
        idx_grouping[(r['initialization'],r['seed'])].add(r['initial_hashes']['indexers'])
        runs.append({'run':name,'updates':len(steps),'dense_steps':dense_steps,'status':'passed'})
    assert all(len(v)==1 for v in grouping.values()); assert all(len(v)==1 for v in idx_grouping.values())
    audit={'started_utc':started,'finished_utc':utc(),'status':'passed','files_verified':len(manifest),
        'runs':runs,'paired_initialization_groups':len(grouping),'fresh_paragraphs':len(data['examples']),
        'excluded_prior_paragraphs':sum(len(x['examples']) for x in old),'wall_clock_warnings':warnings,
        'language_model_updates':sum(r['updates'] for r in s['results']),'cloud_spend_usd':0}
    write('logs/joint-pilot-v0-audit.json',json.dumps(audit,indent=2)+'\n')
    diagnostic=read('results/joint-routing-diagnostic-v0/summary.json')
    for rec in read('results/joint-routing-diagnostic-v0/manifest.json'):
        assert sha(ROOT/'results/joint-routing-diagnostic-v0'/rec['path'])==rec['sha256']
    cells=[]; timing=[]
    labels={'dense':'稠密','sparse_step0':'第一步稀疏','sparse_warmup':'32步预热后稀疏'}
    for r in s['results']:
        cells.append(f"| {r['initialization']} | {labels[r['method']]} | {r['final_dev']['late_nll']:.5f} | {r['test']['late_nll']:.5f} | {r['final_weights_dense_test']['late_nll']:.5f} |")
        name=f"{r['initialization']}__seed{r['seed']}__{r['method']}"
        timing.append(f"| {name} | {r['started_utc']} | {r['finished_utc']} | {r['elapsed_seconds']:.2f} | {r['updates']} | [日志](../results/joint-pilot-v0/{name}/events.jsonl) |")
    gates=[]; ci=[]; rng=np.random.default_rng(1400)
    for g in s['gates']:
        gates.append(f"- {g['initialization']}：预热追回开发集{g['dev_gap_nats']:+.5f}、测试集{g['test_gap_nats']:+.5f} nats；预定本地定位门槛通过：{g['local_mechanism_gate']}。")
        by={r['method']:r for r in s['results'] if r['initialization']==g['initialization']}
        diff=np.array([a['late_nll']-b['late_nll'] for a,b in zip(by['sparse_step0']['test']['per_example'],by['sparse_warmup']['test']['per_example'])])
        boot=diff[rng.integers(0,len(diff),(5000,len(diff)))].mean(1)
        ci.append({'initialization':g['initialization'],'mean_step0_minus_warmup':float(diff.mean()),
            'paragraph_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
            'scope':'Conditional on one trained model pair; NOT uncertainty over independent training seeds'})
    write('results/joint-pilot-conditional-bootstrap-v0.json',json.dumps(ci,indent=2)+'\n')
    diag_table=[]
    for init in cfg['initializations']:
        for backbone in ['sparse_step0','sparse_warmup']:
            rows={r['rule']:r for r in diagnostic['results'] if r['initialization']==init and r['backbone']==backbone
                  and (r['indexer']==backbone or r['rule'] in ['dense','self_only'] or (r['rule']=='sink_recent' and r['indexer']=='sparse_step0'))}
            diag_table.append(f"| {init} | {backbone} | "+' | '.join(f"{rows[rule]['development']['late_nll']:.5f}" for rule in ['learned','learned_recent','sink_recent','dense','self_only'])+' |')
    plots=ROOT/'results/joint-pilot-figures-v0'; plots.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,init in zip(axes,cfg['initializations']):
        for r in s['results']:
            if r['initialization']!=init: continue
            name=f"{init}__seed{r['seed']}__{r['method']}"
            curve=read(f'results/joint-pilot-v0/{name}/development-curve.json')
            ax.plot([x['step'] for x in curve],[x['late_nll'] for x in curve],marker='o',markersize=3,label=r['method'])
        ax.set(title=init+' initialization',xlabel='Backbone updates',ylabel='Development late-token NLL (nats)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('CPU joint-training pilot: one paired seed, 256-token paragraphs',fontsize=11)
    fig.savefig(plots/'learning-curves.png',dpi=170); plt.close(fig)
    report='''# 首次主干联合训练：完整结果与下一步边界

这次确实训练了语言模型主干：70,426,624参数的GPTNeoX/Pythia纯注意力代理，另295,296个索引器参数。两种起点、三组日程，共6次独立运行；每组256步、65,536训练tokens，共1,536次主干更新。另有之前未保存权重的2步随机token资源预检，不计入科学结果。

早期step1000权重是继续训练，random才是从随机初始化。两种起点不能当成两个训练种子；每个起点仅一个配对种子41。数据为新的304段（256训练/16开发/32测试），排除此前112段的正文与token前缀精确重复。

## 预定主读出

表中是后128个token位置的下一词NLL，越低越好。稀疏组主指标使用原生稀疏掩码；最后一列仅在同一最终权重上切回完整注意力，用来判断即时掩码影响，不能当成稀疏组部署成绩。

| 起点 | 日程 | 最终开发NLL | 最终测试NLL | 同权重改用完整注意力的测试NLL |
|---|---|---:|---:|---:|
'''+ '\n'.join(cells)+'\n\n'+'\n'.join(gates)+'''

预定门槛要求开发和测试都追回至少0.02 nats，才进入进一步本地冷启动定位。这个门槛从来不是付费扩训许可。段落bootstrap另存 `results/joint-pilot-conditional-bootstrap-v0.json`；它只描述固定模型对上的样本波动，不能代替独立训练种子重复。

![学习曲线](../results/joint-pilot-figures-v0/learning-curves.png)

## 排除近邻上下文缺陷：仅开发集、零更新

使用最终权重做固定诊断：原索引器；保证最近完整块在预算内；已知的sink+recent；完整注意力。已强制保留的未完成尾块始终相同。强制最近完整块仍只选8个块，没有偷偷提高注意力预算。索引器跨日程互换也全部保存，不能凭某个组合较好宣称训练方法收益。

| 起点 | 主干 | 原选块 | 预算内保证最近完整块 | sink+recent | 完整注意力 | 仅当前token |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(diag_table)+'''

这属于观察到差距后的探索性开发集诊断，未重新训练，也没有为这组诊断使用测试集。最近块和sink等对照已有前作，不能登记为新贡献。纯GPTNeoX没有Qwen的Gated DeltaNet路径，因此这些诊断不足以判断真实Qwen混合结构从零稀疏训练的效果。

## 工程与证据

20项正确性检查通过：包括GPTNeoX原始稠密前向/反向一致、LM与辅助梯度隔离、严格因果、块/文档/尾块边界、相同预算的近邻规则、真正gather选中KV的前向/反向及教师分布一致，以及“仅当前token”读出确实不依赖此前输入。

已实现 `src/gathered_core.py`，真正只计算所选KV的主分支QK，并接入联合训练工具的可选后端。本批6组训练仍统一使用开跑时冻结的稠密掩码参考，不能把随后新增的gather后端归入这些训练，也未测GPU速度。gather参考会复制每个query的KV，尚不是高效内核；索引器仍计算全部块分数。

每组最终检查点含主干、索引器、AdamW状态、步数和随机状态。初始权重tensor哈希已核对配对相同；主干都有实际更新，纯稠密组的索引器未更新。每步日志记录主干/索引器梯度与UTC/单调时间，预热32步的日程已核对。全部运行文件SHA256见审计。

本项目新云支出0美元，GPU作业0，500美元总上限不变。此阶段没有证明原创方法有效，也没有给出训练或prefill加速比。

## 时间轴与文件

- 固定计划：`docs/joint-pilot-plan-2026-09-14.md`。
- 运行与最终权重：`results/joint-pilot-v0/`；开发集诊断：`results/joint-routing-diagnostic-v0/`。
- 哈希、步骤和梯度审计：`logs/joint-pilot-v0-audit.json`。
- [逐次时间轴](joint-pilot-run-index-2026-09-14.md)。
'''
    write('docs/joint-pilot-results-2026-09-14.md',report)
    write('docs/joint-pilot-run-index-2026-09-14.md','# 主干联合训练逐次时间轴\n\n| 运行 | 开始UTC | 结束UTC | 单调秒 | 主干更新 | 原日志 |\n|---|---|---|---:|---:|---|\n'+'\n'.join(timing)+'\n')
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        f.write('\n## 主干联合训练与预算内局部路由诊断\n\n')
        f.write(f"- {data['started_utc']} 至 {data['finished_utc']}：新的304段token数据准备完成。\n")
        for r in s['results']: f.write(f"- {r['started_utc']} 至 {r['finished_utc']}：{r['initialization']} / {r['method']}，256次主干更新完成。\n")
        f.write(f"- {diagnostic['started_utc']} 至 {diagnostic['finished_utc']}：{diagnostic['conditions']}个开发集路由诊断条件完成，新增主干更新0。\n")
        f.write(f'- {utc()}：完成主干日志、参数更新与数据/运行文件核验。云支出0美元。\n')
    c=read('logs/control-state.json'); c.update(updated_utc=utc(),status='joint_pilot_and_routing_diagnosis_complete',
        active_training_jobs=0,language_model_training_runs=len(s['results']),training_runs=len(s['results']),
        language_model_optimizer_updates=audit['language_model_updates'],language_model_training_tokens=sum(r['tokens'] for r in s['results']),
        correctness_tests_passed=20,joint_real_text_unique_paragraphs=304,joint_routing_diagnostic_conditions=diagnostic['conditions'],
        current_decision_report='docs/joint-pilot-results-2026-09-14.md',run_index='docs/joint-pilot-run-index-2026-09-14.md',
        audit_report='logs/joint-pilot-v0-audit.json',paid_experiment_gate_passed=False,
        result_scope='Frozen indexer diagnostics plus 6 actual CPU full-backbone training runs, one paired seed per initialization',
        next_action='Review joint pilot and local-control evidence before deciding any further local runs or cloud expansion.')
    write('logs/control-state.json',json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'audit':audit,'gates':s['gates'],'report':'docs/joint-pilot-results-2026-09-14.md'}))

if __name__=='__main__': main()
