"""Build the 2026-09-13 decision report and run index from preserved evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def write(path, text):
    (ROOT / path).write_text(text, encoding='utf-8')


def main():
    report_path = 'docs/realtext-all-checkpoints-decision-2026-09-13.md'
    if (ROOT / report_path).exists():
        raise FileExistsError('Preserve the dated decision report')
    groups = ['realtext-v0', 'realtext-v1-optimization-check',
              'realtext-early-step1000', 'realtext-early-step10000']
    summaries = {name: read(f'results/{name}/summary.json') for name in groups}
    audit = read('logs/realtext-all-checkpoints-audit.json')
    health = read('logs/indexer-health-all-checkpoints.json')
    now = datetime.now(timezone.utc).isoformat()
    assert sum(s['runs_completed'] for s in summaries.values()) == 120
    assert all(not s['local_followup_gate_passed'] for s in summaries.values())
    updates = sum(r['optimizer_events'] for r in audit['runs'])
    comparisons = []
    table = []
    for name, label in [('realtext-early-step1000', '训练第1,000步'),
                        ('realtext-early-step10000', '训练第10,000步'),
                        ('realtext-v1-optimization-check', '最终公开权重')]:
        summary = summaries[name]
        by = {(r['layer'], r['method']): r['mean']['relative_output_error'] for r in summary['summary']}
        for gate in summary['gate']:
            layer = gate['layer']
            selected, probe, matched = [by[(layer, method)] for method in
                ['selected_only', 'outside_probe', 'selected_budget_match']]
            table.append(f'| {label} | {layer + 1} | {selected:.5f} | {probe:.5f} | {matched:.5f} | '
                         f"{-gate['relative_reduction_vs_selected'] * 100:+.2f}% |")
            comparisons.append({'checkpoint': label, 'run_group': name, **gate})
    report = '''# 均匀集合外抽查：三检查点验证与阶段决定

结论：暂停把“均匀抽查未选块”作为当前付费训练候选。在固定的本地配置下，它没有稳定胜过原监督；将额外候选评分预算用于原监督多训练，通常更有效。这个结果不足以支撑方法论文，也不否定所有原生稀疏训练方案。

## 这次实际做到了哪一步

在之前 24 次合成诊断之外，完成了 120 次真实文本上的小索引器学习运行：最终公开模型的原配置与统一优化修正各 30 次，公开 step1000、step10000 检查点各 30 次。另有 36 个单层稀疏干预条件、576 次完整模型前向，直接检查下一个词的预测损失。这里只训练外接索引器；语言模型底座训练次数仍为 0。

三个检查点都来自 [EleutherAI/Pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)，实际参数量 70,426,624。数据来自 [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext)，版本、模型提交及哈希均落盘。不同检查点使用完全相同、同顺序的 48 段文本：训练24、开发8、测试16，每段256个输入token。三套轨迹不是144段独立文本，三个索引器种子也不是三次底座预训练。

预先固定第2、5层（代码索引1、4）；每4 tokens一块、选8块、额外抽查2块。比较仅选中监督、集合外抽查、全量预热、始终全监督、原监督多训练五组。各组配对初始化相同。成熟模型 v0 的全监督发生零分/零梯度异常后，所有方法统一改为学习率1e-4、1,200次更新；匹配评分预算的对照为1,462次。早期检查点沿用此配置，没有按候选或种子单独调参。

## 主要结果

指标是稀疏注意力输出相对原稠密输出的平方误差比，统计后128个位置，表内为3种子均值，越低越好。“抽查误差变化”相对普通原监督计算；正号表示变差。

| 冻结的底座检查点 | 层 | 原监督 | 均匀抽查 | 原监督多训练 | 抽查误差变化 |
|---|---:|---:|---:|---:|---:|
''' + '\n'.join(table) + '''

第1,000步的第2层有均值改善，但种子方向不一致，第5层同时变差。第10,000步第5层误差明显增加。最终模型两层也没有一致收益。三个检查点均未通过事先记录的门槛：两层各自相对普通原监督及评分预算匹配对照均降低超过1%，且3个配对种子相对普通原监督方向一致。

匹配的是逻辑候选 query-token 评分数量；不等于完整 FLOPs、租卡机时或美元成本。本实现仍计算稠密CPU矩阵，没有实测GPU加速。

## 对实际词预测有没有用

在最终公开模型上，每次只将第2层或第5层替换为保存的稀疏索引方案，其余层和所有底座权重不变。原稠密模型平均 NLL 为4.22840 nats。改第2层时，普通原监督和抽查的 NLL 增量分别为0.03220、0.03215，几乎相同；改第5层时分别为0.02383、0.03133，抽查更差。所有保存的方法均评测，没有按NLL选择检查点。

这没有显示一致的词预测收益。测试只有16段，并在优化检查中复用，属于探索性诊断；NLL是同一数据的第二种读出，不能当成独立复现。完整五组数值与图见[最终检查点报告](realtext-final-checkpoint-results-2026-09-13.md)。

## 为什么不能把阴性结果说得过头

只读检查120个已保存索引器发现：部分参照确实优化失效。第1,000步第2层的全监督/全量预热，对开发集后半段约99%的query给所有可见块打零分。第10,000步第5层，抽查也有约15%的query全部零分，且可见评分约91%为零；不能把其误差增加全部归因于抽查机制本身。证据见 `logs/indexer-health-all-checkpoints.json`；该核查没有重新训练。

因此本轮决定是“当前实现和配置不值得付费扩展”，不是“理论上抽查必然无效”。没有继续扫学习率或换种子寻找阳性。缩小的QSA式索引器使用参数无关RMS归一化，省略完整架构的一些细节；Pythia也不是Qwen，早期快照分别冻结，不等于索引器与变化中的底座联合训练。序列256也没有检验长上下文收益。

## 已保存的证据和运行时间

10项正确性检查通过。模型原生注意力与提取Q/K重建在保存轨迹上的最大误差为0；单层稠密干预也与原缓存NLL一致。全量核验覆盖25个下载资产、144个轨迹文件、624个运行文件、40个NLL干预文件，并检查更新步号、单调累计训练时间和配对初始权重。

原日志保留一次约1.34秒的UTC墙钟回拨：最终模型v1，第5层、seed33、dense_warmup，第670到671步。审计明确记录该异常；未改写原时间，运行顺序依据步号和单调计时。新运行增加逐事件单调时钟字段。

所有训练条目的起止、步骤数与原始日志链接见[逐次运行时间轴](run-index-2026-09-13.md)，整体过程见[项目时间轴](../TIMELINE.md)。完整审计：`logs/realtext-all-checkpoints-audit.json`；配置、代码快照、损失曲线、初始/最终权重均在各自 `results/` 目录。首次准备失败记录也保留。

## 下一步如何避免继续花冤枉钱

当前均匀抽查变体停止扩大训练，首笔30美元云实验不启动。整个本地诊断新增云支出0美元，500美元总预算未由本项目动用；历史RunPod实例是否仍计费未核实。

现有工具保留为后续稀疏训练研究的诊断底座。再次开展方法实验前，先核对官方索引器的归一化、可学习缩放与打分方式，再区分“评分器没学好”与“监督目标没有对准最终注意力输出”这两个问题。这只是待筛查的问题，不是已经确认的创新点。只有完成相近贡献核查、修复失效参照并在新留出数据出现稳定改善，才考虑联合训练或租GPU；这批结果本身不能证明减少稠密预热或全程稀疏训练可行。
'''
    write(report_path, report)
    # One table for every real-text run; filenames link to immutable raw events.
    index = ['# 逐次运行时间轴（UTC）', '',
        '本表由保存的结果生成。索引器运行不是语言模型训练。原日志未改写；墙钟回拨情况见总报告。', '']
    for name in groups:
        index += [f'## {name}', '',
                  '| 层 | 种子 | 方法 | 开始 UTC | 结束 UTC | 更新数 | 单调计时秒 | 日志 |',
                  '|---:|---:|---|---|---|---:|---:|---|']
        for r in summaries[name]['results']:
            run_id = f"layer{r['layer']}__seed{r['seed']}__{r['method']}"
            index.append(f"| {r['layer']+1} | {r['seed']} | {r['method']} | {r['started_utc']} | "
                         f"{r['finished_utc']} | {r['updates']} | {r['elapsed_seconds']:.4f} | "
                         f"[events](../results/{name}/{run_id}/events.jsonl) |")
        index.append('')
    index += ['## 其他记录', '',
        '- 之前24次合成运行：[逐次结果](../results/mechanism-v0/summary.json)。',
        '- 36个单层干预条件：[逐条件结果](../results/realtext-v1-nll-intervention/summary.json)。',
        '- 120个索引器的只读健康检查：[记录](../logs/indexer-health-all-checkpoints.json)。', '']
    write('docs/run-index-2026-09-13.md', '\n'.join(index))
    control = read('logs/control-state.json')
    control.update(updated_utc=now, event='real_text_diagnostics_complete_uniform_probe_no_go',
        status='uniform_probe_frozen_pending_reframing', dataset_ready=True,
        result_scope='Frozen Pythia traces across 3 checkpoints; indexer-only learning and final-model one-layer NLL interventions',
        real_text_reference_runs_completed=120, real_text_indexer_optimizer_updates=updates,
        real_text_checkpoints=3, real_text_unique_paragraphs=48, real_text_trace_files=144,
        next_token_intervention_conditions=36, next_token_intervention_forward_passes=576,
        next_action='No paid expansion of uniform probing. Audit official indexer fidelity and target/output alignment before any new-method experiment; novelty remains unresolved.',
        paid_experiment_gate_passed=False, local_followup_gate_passed=False,
        current_decision_report=report_path, audit_report='logs/realtext-all-checkpoints-audit.json',
        run_index='docs/run-index-2026-09-13.md', active_training_jobs=0)
    write('logs/control-state.json', json.dumps(control, ensure_ascii=False, indent=2) + '\n')
    write('results/realtext-checkpoint-comparison.json', json.dumps({
        'created_utc': now, 'comparisons': comparisons, 'decision': 'no_paid_expansion',
        'real_text_runs': 120, 'optimizer_updates': updates,
        'cloud_jobs_started': 0, 'new_cloud_spend_usd': 0}, ensure_ascii=False, indent=2) + '\n')
    timeline = ['\n## 真实文本验证收尾\n']
    for name in groups[1:]:
        s = summaries[name]
        timeline.append(f"- {s['started_utc']} 至 {s['finished_utc']}：{name} 完成30次索引器学习；预定门槛未通过。")
    nll = read('results/realtext-v1-nll-intervention/summary.json')
    timeline.append(f"- {nll['started_utc']} 至 {nll['finished_utc']}：完成36个单层干预条件、576次完整模型前向，未见一致NLL收益。")
    for name in groups[2:]:
        events = [json.loads(line) for line in (ROOT / f'data/{name}/events.jsonl').read_text().splitlines()]
        timeline.append(f"- {events[0]['utc']} 至 {events[-1]['utc']}：{name} 保存48条轨迹；与最终模型使用相同输入和划分，注意力重建误差0。")
    timeline.sort()  # Raw UTC start times provide the stage chronology.
    heading = '\n## 真实文本验证收尾\n\n'
    entries = [line for line in timeline if line.startswith('- ')]
    entries += [f"- {audit['started_utc']} 至 {audit['finished_utc']}：25个下载资产、144轨迹、624运行文件与40干预文件完成哈希核验；120条运行步骤和初始化配对检查通过，保留一次UTC回拨记录。",
        f"- {health['started_utc']} 至 {health['finished_utc']}：只读检查全部120个最终索引器的开发集评分和已存梯度；新增更新0，记录部分参照失效。",
        f'- {now}：保存三检查点决定报告、逐次运行时间轴和状态。当前均匀抽查候选暂停扩大训练；本项目新增云支出0美元、底座训练0次、活动训练作业0。']
    with (ROOT / 'TIMELINE.md').open('a', encoding='utf-8') as stream:
        stream.write(heading + '\n'.join(entries) + '\n')
    print(json.dumps({'report': report_path, 'run_index_entries': 120,
                      'real_text_updates': updates, 'current_decision': 'no_paid_expansion'}))


if __name__ == '__main__':
    main()
