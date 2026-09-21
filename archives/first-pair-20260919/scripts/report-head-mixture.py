"""Audit closed head-normalizer screening and write its decision without a sweep."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def write(p,x): (ROOT/p).write_text(x,encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()

def main():
    started=utc(); batches=['head-mixture-final-v0','head-mixture-early-v0']; audits=[]; summaries=[]; total_updates=0
    for batch in batches:
        source=ROOT/'results'/batch; summary=read(f'results/{batch}/summary.json'); summaries.append(summary)
        manifest=read(f'results/{batch}/manifest.json')
        for rec in manifest: assert sha(source/rec['path'])==rec['sha256'],rec['path']
        groups=defaultdict(set); warning=[]; updates=0
        for run in summary['results']:
            name=f"layer{run['layer']}__seed{run['seed']}__{run['method']}"; folder=source/name
            events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
            steps=[x for x in events if x['event']=='optimizer_step']
            assert [x['step'] for x in steps]==list(range(1,run['updates']+1))
            assert events[-1]['event']=='complete'
            for before,after in zip(events,events[1:]):
                assert after['monotonic_seconds']>=before['monotonic_seconds']
                if datetime.fromisoformat(after['utc'])<datetime.fromisoformat(before['utc']): warning.append(name)
            assert all(math.isfinite(x['loss']) and math.isfinite(x['gradient_norm']) for x in steps)
            assert steps[-1]['candidate_token_pairs']==run['candidate_token_pairs_logical_only']
            groups[(run['layer'],run['seed'])].add(sha(folder/'initial-indexer.pt'))
            updates+=len(steps)
        assert all(len(v)==1 for v in groups.values())
        assert not summary['local_followup_gate_passed']
        audits.append({'batch':batch,'runs':len(summary['results']),'files_verified':len(manifest),'updates':updates,
            'paired_initialization_groups':len(groups),'wall_clock_warnings':warning})
        total_updates+=updates
    audit={'started_utc':started,'finished_utc':utc(),'status':'passed','batches':audits,'updates':total_updates,
        'runs':sum(x['runs'] for x in audits),'cloud_spend_usd':0}
    write('logs/head-mixture-training-audit.json',json.dumps(audit,indent=2)+'\n')
    table=[]
    labels={'selected_only':'原监督','normalizer_exact':'准确稠密normalizer（诊断）','normalizer_sampled':'两块抽样normalizer',
            'outside_probe':'集合外抽查（旧候选）','selected_budget_match':'原监督多训练（评分预算匹配）'}
    for batch,summary in zip(batches,summaries):
        for layer in [1,4]:
            rows={r['method']:r for r in summary['summary'] if r['layer']==layer}
            for method,label in labels.items():
                table.append(f"| {batch} | 第{layer+1}层 | {label} | {rows[method]['mean']['relative_output_error']:.6f} |")
    report='''# 头间归一化校正：机制成立，学习收益不足，关闭该变体

先在可控计算中确认：每个注意力头分别对选中集合做softmax，再平均各头，与先算完整注意力、再限制到选中集合，得到的教师分布可能不同。使用准确的每头保留质量可以恢复后一种分布。两块抽样能减轻分布偏差，但不能保证路由或语言模型变好。

## 完整训练比较

最终Pythia检查点与早期step1000分别比较五种方式；两个预定层、三个相同初始化的配对种子，共60次索引器训练。两种检查点均使用此前已观察过的留出集，因此属于探索性对照。仅更新索引器，主干冻结。

| 数据/检查点 | 层 | 方法 | 测试注意力输出误差，越低越好 |
|---|---|---|---:|
'''+ '\n'.join(table)+'''

两批均未通过事先规定的门槛：两个层都应超过1%的平均相对改善，胜过原监督和评分预算匹配对照，并在三个种子中均胜过原监督。没有修改阈值、择层报告或继续加抽样数量。主指标失败后不补做NLL寻找正结果。

准确稠密normalizer也没有在两个检查点、两个层上稳定占优。因此不能只归因于两块样本太少，也没有理由按当前变体租卡。

## 结论范围与前作

QSA原文式20是否依赖完整稠密教师的实现细节仍未确认；准确校正可能只是恢复该式的预期语义，不能声称发现官方训练错误。KVpop附录D.3已有稀疏LSE近似稠密normalizer来生成标签的做法。抽样分母无偏不代表softmax或KL无偏；当前比值估计存在偏差。

其他被筛除的宽泛改法包括value-aware路由、输出对齐和因果证据监督路由，均已有直接前作。来源与具体阅读位置见[筛查与固定计划](head-mixture-plan-2026-09-14.md)。[数学核查](head-mixture-estimator-math-2026-09-14.md)只是恒等式、方差与松界，不包装成新理论贡献。

## 接续

关闭均匀抽查和当前头间校正这两个具体变体。接着建立真正更新主干的联合训练对照，确认原问题在我们的可负担模型上是否有明显差距；这属于验证工具，不登记新方法。[联合训练计划](joint-pilot-plan-2026-09-14.md)在其结果出来前保存。

本批新增60次小索引器运行、75144次更新；主干训练0次。云支出0美元。随后新开的联合训练应单独统计，不与这些冻结轨迹诊断合并宣称模型训练次数。

完整步骤、梯度、时间、初始/最终权重及源码均保留；核验见 `logs/head-mixture-training-audit.json`。准确normalizer读取所有候选，计入逻辑评分量，但所有实现均为CPU稠密缓存诊断，不用其秒数宣称稀疏效率。
'''
    write('docs/head-mixture-results-2026-09-14.md',report)
    index=['# 头间校正逐次运行时间轴（UTC）','','均为冻结主干的小索引器运行。','']
    for batch,summary in zip(batches,summaries):
        index += ['## '+batch,'','| 运行 | 开始UTC | 结束UTC | 单调秒 | 更新 | 日志 |','|---|---|---|---:|---:|---|']
        for r in summary['results']:
            name=f"layer{r['layer']}__seed{r['seed']}__{r['method']}"
            index.append(f"| {name} | {r['started_utc']} | {r['finished_utc']} | {r['elapsed_seconds']:.3f} | {r['updates']} | [events](../results/{batch}/{name}/events.jsonl) |")
    write('docs/head-mixture-run-index-2026-09-14.md','\n'.join(index)+'\n')
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        f.write('\n## 头间校正与进入联合训练\n\n')
        for batch,s in zip(batches,summaries): f.write(f"- {s['started_utc']} 至 {s['finished_utc']}：{batch}完成30次小索引器训练，门槛未通过。\n")
        f.write(f'- {utc()}：核验60次训练、75144步、配对初始化和全部清单文件，关闭头间校正变体。\n')
    control=read('logs/control-state.json')
    control.update(updated_utc=utc(),status='joint_backbone_pilot_running',real_text_reference_runs_completed=204,
        real_text_indexer_optimizer_updates=221994,head_mixture_runs=60,head_mixture_optimizer_updates=75144,
        correctness_tests_passed=15,training_implementation_ready=True,active_training_jobs=1,
        current_decision_report='docs/head-mixture-results-2026-09-14.md',run_index='docs/head-mixture-run-index-2026-09-14.md',
        next_action='Finish registered CPU joint LM pilot. No cloud expansion: uniform probing and head correction rejected.',
        joint_training_scope='Differentiable CPU dense-mask pure GPTNeoX reference; not efficient sparse kernel or Qwen hybrid',
        technical_backbone_benchmark_updates=2,paid_experiment_gate_passed=False)
    write('logs/control-state.json',json.dumps(control,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__': main()
