"""Produce the calibration follow-up after both training and NLL finish."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def write(name, text):
    (ROOT/name).write_text(text,encoding='utf-8')


def main():
    filename = 'docs/calibration-followup-results-2026-09-13.md'
    if (ROOT/filename).exists():
        raise FileExistsError('Preserve report')
    diagnostic = read('results/indexer-calibration-v0/summary.json')
    da = read('logs/indexer-calibration-v0-audit.json')
    fresh = read('results/realtext-calibrated-fresh-v0/summary.json')
    nll = read('results/realtext-calibrated-fresh-nll-v0/summary.json')
    audit = read('logs/calibrated-fresh-complete-audit.json')
    assert audit['status'].startswith('passed') or 'audit_passed' in audit['status']
    fresh_health=read('logs/calibrated-fresh-indexer-health.json')
    assert max(r['late_validation_all_zero_query_fraction'] for r in fresh_health['rows']) == 0
    assert max(r['last_100_steps_zero_gradient_fraction'] for r in fresh_health['rows']) == 0
    rows = {(r['layer'],r['method']):r for r in fresh['summary']}
    nll_groups = defaultdict(list)
    for row in nll['results']:
        nll_groups[(row['layer'],row['method'])].append(row)
    table=[]
    methods={'selected_only':'原监督','outside_probe':'集合外抽查','full_supervision':'全监督',
             'selected_budget_match':'原监督多训练（评分预算匹配）'}
    for method,label in methods.items():
        table.append('| '+label+' | '+' | '.join(f"{rows[(layer,method)]['mean']['relative_output_error']:.5f}" for layer in [1,4])+' | '+
            ' | '.join(f"{statistics.mean(r['mean_delta_vs_dense'] for r in nll_groups[(layer,method)]):+.5f}" for layer in [1,4])+' |')
    diag_table=[]
    for r in da['aggregate']:
        if r['method'] != 'full_supervision':
            continue
        m=r['mean_development']
        case='step1000，第2层' if r['case'].endswith('step1000') else 'step10000，第5层'
        diag_table.append(f"| {case} | {r['variant']} | {m['all_zero_query_fraction']*100:.2f}% | {m['relative_output_error']:.5f} |")
    gate_lines=[]
    for gate in fresh['gate']:
        gate_lines.append(f"- 第{gate['layer']+1}层：抽查相对原监督的输出误差减少{gate['relative_reduction_vs_selected']*100:+.2f}%，相对评分预算匹配对照减少{gate['relative_reduction_vs_matched']*100:+.2f}%；三个配对种子方向是否全一致：{gate['all_paired_seeds_better_than_selected']}。")
    passed=fresh['local_followup_gate_passed']
    decision='出现有限的本地筛查信号，仍不能启动大规模训练或声称新方法。' if passed else '抽查仍未通过预先固定的收益门槛，继续暂停此变体的付费扩训。'
    text='''# 继续验证：定位分数尺度问题，并用新文本重比

这轮有了明确进展：上一轮部分零分失败可以通过调整索引器训练分数尺度消除。单独加入可学习RMS增益基本无效；将分数除以sqrt(头维度)后，两处问题设置的全零query比例降为0。这个发现修正了我们对旧实验的解释，但只是实验实现与优化诊断，不是原创论文贡献。

''' + decision + '''

## 先把失效原因拆开

用完全相同投影初始化、数据顺序、维度、位置编码和学习率，固定四组：旧实现、仅可学习RMS增益、仅分数缩放、两者同时。两个此前失效位置、三种监督方式、三个种子、每次400步，共72次。只读取训练和开发集，没有加载测试集。

下表为全监督参照的三个种子均值；注意力输出误差越小越好。新旧两组同样都是400步，不能与旧报告的1200步混作同预算比较。

| 位置 | 变体 | 开发集全零query比例 | 注意力输出误差 |
|---|---|---:|---:|
'''+'\n'.join(diag_table)+'''

证据来自固定因素实验，因此能确认：在这两处缩小代理设置中，分数尺度是导致训练失败的重要因素。不能据此断言真实Qwen会同样失效，也不能保证任意维度、长度或初始化都被修复。

公开推理实现中，统一正比例缩放不改变TopK，但会改变用于训练的softmax和梯度。报告公式、vLLM与NeMo推理参考对统一缩放的写法不同，不能据此判定官方训练温度。可学习RMS增益、温度/尺度控制都不是新思想。来源、代码位置和公式解释见[实现核对](indexer-fidelity-audit-2026-09-13.md)。

## 基线恢复后，用新文本重新比较

使用同一个Pythia step1000底座，预先固定可学习RMS增益与1/sqrt(d)分数缩放，所有方法共用；其余模型与优化设置沿用之前配置。分别在第2、5层比较原监督、抽查、全监督、评分预算匹配的原监督，三个种子，共24次索引器训练。

此次重新选取24训练、8开发、32测试，共64段文本；按完整文本哈希及257-token前缀哈希排除上一轮48段。三组之间也无这些精确重复。数据仍来自WikiText，不能说与底座预训练语料无重叠，也不是换了数据领域或独立重训底座。

24个最终索引器在开发集上全零query比例均为0，最后100步也均无零梯度更新；旧的评分失效在这批配置中未重现，证据在 `logs/calibrated-fresh-indexer-health.json`。

每段256输入tokens，在后128个位置统计注意力输出误差和下一个词预测NLL。NLL由一次只更换一个注意力层得到，底座权重不更新。全部保存的最终索引器均评测；30个条件（24个学习条件+6个固定参照）各32段，共960次模型前向。NLL与注意力误差复用同一新测试集，是两种指标，不能算两次独立复现。

| 方法 | 第2层输出误差 | 第5层输出误差 | 改第2层的NLL增量 | 改第5层的NLL增量 |
|---|---:|---:|---:|---:|
'''+'\n'.join(table)+'\n\n'+'\n'.join(gate_lines)+'''

输出误差门槛要求两层各自超过1%的相对改善、胜过普通原监督和评分预算匹配对照，且三个配对种子相对普通原监督全同向。NLL越低越好；细微均值差异未做显著性主张。评分预算匹配不是GPU同机时或同美元比较。

## 怎么接着做

保留修复后的诊断工具；不再用旧的评分失效参照支持方法优越性。当前额外抽查机制的贡献仍未成立。下一步若继续收窄研究问题，应先检查“教师块分布排名”是否对准最终注意力输出、以及这种偏差是否已有直接前作；仅有相关性或普通温度调参不足以成为新题。此处尚未完成该方向的验证，不把它登记为确定的新选题。

本轮全部使用本地CPU。新增云支出0美元，语言模型底座训练0次。500美元额度保持，不自动开启付费实例。

## 保存和核验

- 72次因素诊断：`results/indexer-calibration-v0/`，含初始/最终权重、逐步梯度/损失、源码和配置。
- 24次新文本训练：`results/realtext-calibrated-fresh-v0/`。
- 30组词预测干预：`results/realtext-calibrated-fresh-nll-v0/`。
- 数据/步骤/权重核验：`logs/indexer-calibration-v0-audit.json`、`logs/calibrated-fresh-complete-audit.json`。
- 11项正确性检查：`logs/correctness-calibration-factory.log`；包括缩放前后初始路由一致、可学习增益有梯度、底座输入无辅助梯度。
- 旧实验原样保留，其解释以本次明确的失效原因补充；前轮UTC回拨记录没有删除。本轮原始时间与核验结果分别保存。
- [本轮逐次时间轴](calibration-run-index-2026-09-13.md)。
'''
    write(filename,text)
    index=['# 本轮逐次运行时间轴（UTC）','','由保存的结果生成；以下均为小索引器运行，非底座训练。','']
    for name,summary in [('indexer-calibration-v0',diagnostic),('realtext-calibrated-fresh-v0',fresh)]:
        index += ['## '+name,'','| run ID | 开始UTC | 结束UTC | 单调耗时秒 | 步数 | 原日志 |','|---|---|---|---:|---:|---|']
        for row in summary['results']:
            run=row.get('run_id') or f"layer{row['layer']}__seed{row['seed']}__{row['method']}"
            index.append(f"| {run} | {row['started_utc']} | {row['finished_utc']} | {row['elapsed_seconds']:.4f} | {row['updates']} | [events](../results/{name}/{run}/events.jsonl) |")
        index.append('')
    index += ['## NLL评测','','[逐条件起止与逐段NLL](../results/realtext-calibrated-fresh-nll-v0/summary.json)。','']
    write('docs/calibration-run-index-2026-09-13.md','\n'.join(index))
    now=datetime.now(timezone.utc).isoformat()
    control=read('logs/control-state.json')
    control.update(updated_utc=now,status='calibration_repaired_fresh_confirmation_complete',
        event='calibration_followup_complete',active_training_jobs=0,calibration_diagnostic_runs=72,
        calibration_optimizer_updates=28800,real_text_reference_runs_completed=144,
        real_text_unique_paragraphs=112,calibrated_fresh_runs=24,calibrated_fresh_paragraphs=64,
        real_text_examples=112,real_text_trace_files=208,
        real_text_indexer_optimizer_updates=116478+sum(r['updates'] for r in fresh['results']),
        next_token_intervention_conditions=66,next_token_intervention_forward_passes=1536,
        local_followup_gate_passed=passed,paid_experiment_gate_passed=False,
        current_decision_report=filename,run_index='docs/calibration-run-index-2026-09-13.md',
        audit_report='logs/calibrated-fresh-complete-audit.json',
        result_scope='Frozen Pythia indexer diagnostics, controlled score-calibration factors, disjoint fresh paragraphs and single-layer NLL; no backbone training',
        correctness_tests_passed=11,
        next_action='No paid expansion. Calibration issue is localized, not novelty. Audit objective/output mismatch literature and falsifiable examples before proposing another sparse-training method.')
    write('logs/control-state.json',json.dumps(control,ensure_ascii=False,indent=2)+'\n')
    lines=['\n## 分数尺度定位与新文本确认\n']
    source_manifest=read('literature/indexer-fidelity-2026-09-13/manifest.json')
    lines.append(f"- {source_manifest[0]['started_utc']} 至 {source_manifest[-1]['finished_utc']}：保存6个固定提交的实现/配置来源文件，只读核对；不执行外部源码。")
    data_manifest=read('data/realtext-calibrated-fresh-v0/manifest.json')
    lines.append(f"- {data_manifest['started_utc']} 至 {data_manifest['finished_utc']}：排除旧48段后准备64段新文本，复用已校验本地模型资产，Q/K注意力重建误差0。")
    for name,summary in [('72次因素诊断',diagnostic),('24次新文本索引器训练',fresh),('30组新文本NLL干预',nll)]:
        lines.append(f"- {summary['started_utc']} 至 {summary['finished_utc']}：{name}完成。")
    lines.append(f"- {now}：本轮数据和运行核验完成，保存尺度问题定位与新文本报告。新文本抽查门槛通过={passed}；付费实验未启动，活动训练0。")
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'report':filename,'fresh_gate':passed,'new_indexer_runs':96,'new_cloud_spend_usd':0}))


if __name__=='__main__':
    main()
