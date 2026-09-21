"""Persist audited results and the next boundary after the 128-token run."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    r=read('results/zoology-length128-cpu-v0/result.json')
    a=read('results/zoology-length128-analysis-v0/audit.json')
    q=read('results/zoology-length128-query-v0/result.json')
    value_probe=read('results/zoology-length128-source-values-v0/result.json')
    assert r['status']=='complete'
    for folder in ['results/zoology-length128-cpu-v0','results/zoology-length128-query-v0','results/zoology-length128-source-values-v0','results/zoology-length128-analysis-v0']:
        for f in read(folder+'/manifest.json'):assert sha(ROOT/folder/f['path'])==f['sha256']
    launch=read('logs/zoology-length128-launch.json');assert launch['exit_code']==0
    now=datetime.now(timezone.utc).isoformat();passed=a['passed']
    outcome='长度128也已跑通，下一步加入明确的远距离条件' if passed else '长度128未过双门槛，下一步先定位错误'
    next_step=('固定128长度，将查询集中在后半段，明确其对前方记录的依赖；之后再设计稠密、局部与候选稀疏比较。'
        if passed else '先依据距离分组与学习曲线定位误差，提出有针对性的后续检查；不扩大或追加租卡。')
    handoff=ROOT/'provenance/zoology-length128-handoff-2026-09-14';handoff.mkdir(exist_ok=False)
    changed=['STATE.md','README.md','TIMELINE.md','logs/control-state.json']
    for rel in changed:
        p=handoff/'before'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    state=f'''# 当前状态：{outcome}

更新：{now}。本轮CPU训练、独立新题、查询干预与结果核验全部结束，活动训练0，新增GPU作业0。总500美元/首阶段30美元不变；旧Pod账单和停止状态未核实。

## 最新结果

Zoology长度128、445952参数、seed123，从头训练{r['epochs']}轮、{r['updates']}更新；最终验证{r['curves'][-1]['valid/accuracy']:.3%}，独立1000序列/4000答案{r['fresh_evaluation']['accuracy']:.3%}，序列bootstrap区间{a['fresh_sequence_bootstrap_95'][0]:.3%}—{a['fresh_sequence_bootstrap_95'][1]:.3%}。改最后查询键后{q['changed_query_accuracy']:.2%}正确，{q['prediction_changed_fraction']:.2%}预测改变。训练评测墙钟{a['wall_seconds']:.2f}秒。

相比64基线，只修改序列长度和必要的位置嵌入表；其他配置逐项一致。相同种子不代表不同长度的参数初始化逐位配对。两长度独立新题使用相同种子，可能共享键值模板，不是独立训练重复。

## 下一步与边界

{next_step}后续训练本轮尚未启动。

目前仅一个训练种子和已知合成关联回忆任务。没有完成2K/Qwen、语言建模、稀疏方法优势或新理论证明，没有确认原创贡献。均匀集合外抽查、头间normalizer两个旧候选继续关闭，不换名包装。静态局部信息支持界仍不直接适用于动态选块或Qwen Gated DeltaNet。

## 日志和计数

- 最新报告：docs/zoology-length128-results-2026-09-14.md；固定计划：docs/zoology-length128-plan-2026-09-14.md。
- 原始训练、配置、源码、数据、权重、逐步events.jsonl：results/zoology-length128-cpu-v0/；控制台日志logs/zoology-length128-cpu-v0.log。
- 查询干预results/zoology-length128-query-v0/；曲线/距离/审计results/zoology-length128-analysis-v0/；开始结束见TIMELINE.md。
- 38项CPU检查通过；运行、查询及分析清单SHA核验通过；标签/数据去重、连续更新、UTC、损失梯度有限、余弦调度及首次停止规则检查通过。
- 本轮1次CPU训练，输入tokens {r['input_tokens']}，受监督答案{r['supervised_answers']}，固定10000序列重复训练，不能称为同量独立样本。
- 公开Zoology系列累计2次训练完成、{7199+r['updates']}更新、{14720000+r['input_tokens']}输入tokens、{920000+r['supervised_answers']}监督答案；64长度基线仍为99.125%验证/99.1%新题。此前另1次准备失败0更新，2次技术单步更新另计。
- 历史自制330752参数小模型仍6次完成、1次中止，27142更新/55586816输入tokens/626272监督答案；不能将与Zoology的差异归因于单一因素。
- 历史70M训练仍8次（6CPU+2GPU），6930更新/11440128预测tokens，与toy实验分开。模型及日志已备份。
'''
    (ROOT/'STATE.md').write_text(state,encoding='utf-8')
    with (ROOT/'STATE.md').open('a',encoding='utf-8') as f:
        f.write(f'\n补充源值对调（保持所有查询不变）：正确率{value_probe["swapped_values_accuracy"]:.2%}；其中距离超过64的{value_probe["far_examples"]}题为{value_probe["far_swapped_values_accuracy"]:.2%}。0更新，与主评测共用新题，不是独立重复。结果results/zoology-length128-source-values-v0/。\n')
    c=read('logs/control-state.json')
    c.update(updated_utc=now,status='zoology_length128_complete_audited',active_training_jobs=0,
        current_turn_gpu_jobs_started=0,correctness_tests_passed=38,
        correcteness_tests_note='38 CPU tests passed; 4 CUDA tests belong to prior cloud stage',
        current_decision_report='docs/zoology-length128-results-2026-09-14.md',
        audit_report='results/zoology-length128-analysis-v0/audit.json',next_action=next_step,
        zoology_length128_gate_passed=passed,
        zoology_length128=dict(completed_runs=1,parameters=445952,sequence_length=128,training_seeds=[123],
            epochs=r['epochs'],optimizer_updates=r['updates'],input_tokens=r['input_tokens'],supervised_answers=r['supervised_answers'],
            validation_accuracy=r['curves'][-1]['valid/accuracy'],fresh_accuracy=r['fresh_evaluation']['accuracy'],
            query_intervention_accuracy=q['changed_query_accuracy'],wall_seconds=a['wall_seconds']),
        zoology_series=dict(completed_runs=2,setup_failures=1,optimizer_updates=7199+r['updates'],
            input_tokens=14720000+r['input_tokens'],supervised_answers=920000+r['supervised_answers'],technical_instrumentation_updates=2))
    c['zoology_length128']['source_value_swap_accuracy']=value_probe['swapped_values_accuracy']
    save(ROOT/'logs/control-state.json',c)
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8')
    old=next(x for x in s.splitlines() if x.startswith('最新已在CPU跑通'))
    s=s.replace(old,f'最新完成Zoology长度128的CPU检查：{r["epochs"]}轮、独立新题{r["fresh_evaluation"]["accuracy"]:.2%}，换查询后{q["changed_query_accuracy"]:.2%}。38项检查通过，全部训练结束。本轮没有GPU作业。{next_step}尚无新的稀疏方法贡献，详见STATE.md。')
    s=s.replace('- [公开最小基线结果]','- [最新128长度结果](docs/zoology-length128-results-2026-09-14.md)：与64基线对照、新题、距离分组与查询干预。\n- [128长度预定计划](docs/zoology-length128-plan-2026-09-14.md)。\n- [公开最小基线结果]')
    s=s.replace('36项CPU检查','38项CPU检查')
    s=s.replace('公开64长度基线已可靠学会按键查找，下一步逐项加回远距离约束。',f'64长度基线已跑通，128长度检查完成，最新判断见STATE.md。')
    p.write_text(s,encoding='utf-8')
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## {now} Zoology长度128结果与收尾\n\n- {r["started_utc"]} 至 {r["finished_utc"]}：CPU从头训练{r["epochs"]}轮/{r["updates"]}更新，{r["input_tokens"]}输入tokens/{r["supervised_answers"]}监督答案，纯训练{r["training_seconds"]:.2f}秒、训练评测墙钟{a["wall_seconds"]:.2f}秒；验证{r["curves"][-1]["valid/accuracy"]:.3%}，独立新题{r["fresh_evaluation"]["accuracy"]:.3%}，预定双门槛通过={passed}。\n- {q["started_utc"]} 至 {q["finished_utc"]}：1000对改查询键检查，改后正确率{q["changed_query_accuracy"]:.2%}，0更新。\n- {a["utc"]}：清单SHA、仅长度/位置配置差异、数据去重/标签、连续更新、有限损失/梯度、UTC及原停止规则核验通过。38项CPU检查通过。\n- {now}：报告、状态、源码、曲线及交接快照保存；活动训练0、新GPU作业0。旧Pod计费状态未核实，后续训练尚未启动。\n')
    artifacts=changed+['docs/zoology-length128-results-2026-09-14.md','docs/zoology-length128-plan-2026-09-14.md',
        'configs/zoology-length128-v0.json','logs/zoology-length128-correctness-v0.log','logs/zoology-length128-launch.json',
        'results/zoology-length128-cpu-v0/manifest.json','results/zoology-length128-query-v0/manifest.json',
        'results/zoology-length128-analysis-v0/manifest.json','results/zoology-length128-analysis-v0/audit.json',
        'src/zoology_entry.py','src/run_zoology_baseline.py','scripts/probe-zoology-query.py',
        'scripts/start-zoology-length128.py','scripts/report-zoology-length128.py','scripts/finalize-zoology-length128.py']
    artifacts+=['scripts/probe-zoology-source-values.py','results/zoology-length128-source-values-v0/manifest.json','results/zoology-length128-source-values-v0/result.json']
    with (ROOT/'TIMELINE.md').open('a',encoding='utf-8') as f:
        f.write(f'\n- {value_probe["started_utc"]} 至 {value_probe["finished_utc"]}：补充源值对调检查完成，正确率{value_probe["swapped_values_accuracy"]:.2%}，距离超过64子集{value_probe["far_examples"]}题，0优化更新。此项目为按登记时间补记，不覆盖上方训练结束时间。\n')
    for rel in artifacts:
        target=handoff/'after'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    manifest=[dict(path=p.relative_to(handoff).as_posix(),sha256=sha(p)) for p in sorted(handoff.rglob('*')) if p.is_file()]
    save(handoff/'manifest.json',manifest)
    for f in manifest:assert sha(handoff/f['path'])==f['sha256']
    print(json.dumps(dict(status='complete',passed=passed,active_training_jobs=0,handoff_files=len(manifest))),flush=True)

if __name__=='__main__':main()
