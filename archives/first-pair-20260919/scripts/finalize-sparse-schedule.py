"""Materialize honest decision report and timeline after verified full analysis."""
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    now=datetime.now(timezone.utc).isoformat();run=ROOT/'results/schedule-screen-cloud-v0'
    a=read(ROOT/'results/schedule-screen-analysis-v0/analysis.json');allr=read(run/'result.json')
    assert allr['status']=='complete' and len(allr['runs'])==5 and a['data_labels_and_split_disjointness_checked']
    checkpoint_audit=read(ROOT/'results/schedule-checkpoint-audit-v0/full-fresh-recomputation.json')
    assert checkpoint_audit['passed'] and checkpoint_audit['total_answers']==20000
    passes=[g['candidate'] for g in a['frozen_screen_gates'] if g['pass_to_independent_confirmation']]
    names=dict(dense='始终稠密',native='从第一步稀疏',all_warm4='两层各预热4轮',first_warm8='第一层预热8轮',second_warm8='第二层预热8轮')
    lines=['| 条件 | 新题正确率 | 首次验证≥99%的轮数 | 到99%的累计连接（百万） | 总墙钟秒 |','|---|---:|---:|---:|---:|']
    probes=['| 条件 | 改查询后正确率 | 调换源值后正确率 | 间隔超过32个token的源值调换 |','|---|---:|---:|---:|']
    for r in a['runs']:
        t=r['milestones']['0.99'];p=r['probe'];epoch=t['epoch'] if t else '未达到';edges=f"{t['edges']/1e6:.2f}" if t else '未达到'
        lines.append(f"| {names[r['method']]} | {r['accuracy']:.3%} | {epoch} | {edges} | {r['wall_seconds']:.2f} |")
        probes.append(f"| {names[r['method']]} | {p['changed_query_accuracy']:.2%} | {p['swapped_values_accuracy']:.2%} | {p['far_swapped_values_accuracy']:.2%}（n={p['far_examples']}） |")
    conclusion=('以下候选达到首轮固定门槛，进入独立训练种子确认：'+', '.join(passes)+'。这不是论文贡献确认。') if passes else '两种单层预热均未达到首轮预设门槛，当前证据不支持把“单层早期稠密预热”作为新方法继续扩大。保留负结果，关闭这个具体候选；不能靠事后改轮数或门槛包装正结果。'
    total_seconds=sum(r['wall_seconds'] for r in a['runs']);updates=sum(r['updates'] for r in a['runs']);tokens=sum(r['input_tokens'] for r in a['runs'])
    report=f'''# 稀疏训练预热分配：五组完整结果

更新：{now}。{conclusion}

{chr(10).join(lines)}

这是一张A40上完成的5次训练，每组固定40轮/12520更新，同初始化、相同训练数据与逐轮学习率；仅注意力日程不同。参数437760、长度64、词表256、4组键值、10000训练序列、1000验证序列、seed123。三种预热均额外使用127680000条连接；第9轮起都执行相同top-8注意力（包含最近2个token）。

累计{updates:,}次科学更新、{tokens:,}个输入tokens、8000000个监督答案；同一10000条训练数据重复使用，不是这些数量的独立样本。完成训练/验证/新题评测墙钟合计{total_seconds:.2f}秒（{total_seconds/60:.2f}分钟）。环境预检另16次技术更新；两次准备错误0更新，日志保留。

## 不是靠不看记录的捷径吗

{chr(10).join(probes)}

干预使用同一最终模型和预先生成的新题，0次更新；原始权重哈希保持一致。修改最后查询可能产生分布外重复查询，所以源值调换是补充检查。各条件CPU重算与GPU原始预测的一致率见分析JSON。它们是机制诊断，不是额外训练种子。

## 怎么判断，而不是只挑好看的数字

固定门槛：单层预热的新题正确率同时比直接稀疏、均匀预热高至少5个百分点；或首次达到99%的累计连接至少减少20%，且新题终点不差超过1个百分点。结果见 results/schedule-screen-analysis-v0/analysis.json 的 frozen_screen_gates。所有曲线和同题配对的序列bootstrap区间均保存；单训练种子还不能量化跨种子波动。

本实现仍计算全部QK和掩码AV。表中“累计连接”只是保留连接数，不是实测FLOPs，更不是加速比。不得从本表宣称省了多少GPU费用。实际价格和账单未核实；总预算500美元、首阶段30美元不变。30分钟上限停止的是本批训练进程，不会自动停止Pod计费。

## 与论文目标的距离

没有证明新的理论，也没有完成QSA压缩索引器、Qwen混合结构、真实文本或prefill内核实验。[MSA的预热](https://arxiv.org/html/2606.13392v1)、[Cerebras的层敏感性选择](https://www.cerebras.ai/blog/compressing-kv-cache-memory-by-half-with-sparse-attention)、[KSA的无稠密预热训练](https://github.com/awni/k_sparse_attention)、[Zucchet等的学习平台期与induction层分工](https://arxiv.org/html/2505.17863v2)均属于前作。来源、原文位置和下载哈希见 literature/sparse-schedule-2026-09-14/。当前结果不能被写成这些方向的首次发现。

## 启动器与终端日志异常

启动包装脚本最终返回125，原始science-console.log只保留到最后一组第28轮；原因尚未查明，不能记成正常退出。与此分开的5份events.jsonl共有完整62600步及200条逐轮验证，所有运行文件SHA核验通过，5个最终权重在本机重新计算全部20000个新题答案，与云端保存预测逐项100%一致，0训练更新。原始异常日志不覆盖；从events重建的逐轮摘要明确标为派生文件，存于 results/schedule-checkpoint-audit-v0/。科学结果据这些完整证据确认，不依赖启动器退出码。

## 完整证据

- 原始结果、数据、初始权重、最终权重、源码快照、每步UTC日志：results/schedule-screen-cloud-v0/。
- 审计、配对区间、干预输入/预测、图：results/schedule-screen-analysis-v0/。
- 云端安装、错误、CUDA预检、启动/结束日志：logs/schedule-screen-cloud-v0/。
- 固定计划：docs/paper-evidence-program-2026-09-14.md；解释边界：docs/sparse-schedule-mechanism-boundaries-2026-09-14.md。
- 逐组起止时间已加入TIMELINE.md。代码包与云端校验相同，运行清单{a['manifest_files_verified']}个文件SHA256核验通过。
'''
    (ROOT/'docs/sparse-schedule-screen-results-2026-09-14.md').write_text(report,encoding='utf-8')
    s=read(ROOT/'logs/control-state.json');s.update(updated_utc=now,active_training_jobs=0,status='sparse_schedule_screen_complete',
        current_decision_report='docs/sparse-schedule-screen-results-2026-09-14.md',
        next_action='Independent seed confirmation for passing candidates only; otherwise close single-layer warmup hypothesis and do not expand it.',
        confirmed_original_contributions=0)
    s['sparse_schedule'].update(status='complete_with_wrapper_log_anomaly',scientific_runs_completed=5,updates=updates,input_tokens=tokens,supervised_answers=8000000,
        wall_seconds=total_seconds,screen_passing_candidates=passes,scientific_result_scope='Single-seed exact-score top-k mechanism screen; not efficient sparse training or Qwen result',
        local_results='results/schedule-screen-cloud-v0',analysis='results/schedule-screen-analysis-v0',wrapper_exit_code=125,
        wrapper_anomaly='results/schedule-checkpoint-audit-v0/wrapper-anomaly.json',full_checkpoint_predictions_verified=20000)
    s['gpu_jobs_started_scope']='Historical field refers to prior 70M jobs; sparse_schedule holds the new five-run batch separately.'
    save(ROOT/'logs/control-state.json',s)
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        for r in allr['runs']:f.write(f"\n- {r['started_utc']} → {r['finished_utc']} A40/{r['method']}完成40轮/{r['updates']}更新，新题{r['fresh_accuracy']:.3%}，墙钟{r['wall_seconds']:.2f}秒；原始日志results/schedule-screen-cloud-v0/{r['method']}/events.jsonl。")
        f.write(f"\n- {now} 五组全部本机备份及SHA审计完成，0更新干预完成。{conclusion}\n")
    state=f'''# 当前状态：五组稀疏预热实验已完成并备份

更新：{now}。活动训练0。

{conclusion}

报告：docs/sparse-schedule-screen-results-2026-09-14.md。结果：results/schedule-screen-cloud-v0/；审计/干预/图：results/schedule-screen-analysis-v0/。5组40轮、{updates}更新、{tokens}输入tokens，另16次技术更新。科学更新与准备错误分别计数，完整UTC时间轴在TIMELINE.md。启动器退出125、终端文本缺尾已单独记录；完整结构化日志与最终权重重算20000个答案一致，详见 results/schedule-checkpoint-audit-v0/。

仍未确认论文级原创贡献、真实文本效果或Qwen/QSA全程稀疏可行性。旧均匀集合外抽查、头间normalizer候选仍关闭。本轮是精确QK+掩码AV机制参照，不能称高效稀疏内核。

总预算500美元，首阶段30美元；实际费率和发票未核实。Pod可能仍在计费，训练结束不等于停止Pod。原始数据和所有完成权重已备份在本机。
'''
    (ROOT/'STATE.md').write_text(state,encoding='utf-8')
    print(json.dumps(dict(passing_candidates=passes,updates=updates,input_tokens=tokens,wall_seconds=total_seconds)))
if __name__=='__main__':main()
